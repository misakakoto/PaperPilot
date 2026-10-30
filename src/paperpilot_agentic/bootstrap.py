"""One authoritative composition root for CLI, Web and tests."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

from paperpilot_retrieval import FastEmbedder, OnnxCrossEncoderReranker, PaperKnowledgeStore
from paperpilot_retrieval.reranking import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_FILE

from .adapters import LocalCorpusSearchAdapter, PaperPilotKnowledgeSearchAdapter, make_vision_adapter
from .config import RuntimeSettings
from .orchestrator import ResearchAgentRuntime
from .providers import OpenAICompatibleLLM
from .sandbox import SandboxExperimentRunner
from .scholarly import ScholarlySearchAdapter


def build_runtime(settings: RuntimeSettings | None = None, *,
                  llm_factory=None, vision_factory=None) -> ResearchAgentRuntime:
    """Build the complete Agent from one explicit settings object.

    Expensive semantic models are lazy and opt-in. Existing BM25 indexes remain
    usable when the optional model packages are unavailable.
    Provider factories run before tool registration: swapping a runtime attribute
    afterwards cannot replace the adapter captured by a tool handler.
    """

    settings = settings or RuntimeSettings.from_env()
    settings.workspace.mkdir(parents=True, exist_ok=True)
    settings.runtime_db.parent.mkdir(parents=True, exist_ok=True)

    embedder = None
    dense_status = {'state': 'DISABLED_BY_CONFIGURATION'}
    reranker = None
    cache = settings.workspace / "model-cache"
    embedding_cache = Path(
        os.getenv("PAPERPILOT_EMBEDDING_CACHE", str(cache / "embedding"))
    ).expanduser().resolve()
    reranker_cache = Path(
        os.getenv("PAPERPILOT_RERANKER_CACHE", str(cache / "reranker"))
    ).expanduser().resolve()
    if _enabled("PAPERPILOT_ENABLE_DENSE", default=False):
        try:
            embedder = FastEmbedder(
                cache_dir=embedding_cache,
                model_name=os.getenv(
                    "PAPERPILOT_EMBEDDING_MODEL",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ),
                allow_download=_enabled("PAPERPILOT_ALLOW_MODEL_DOWNLOAD", default=False),
            )
            # Constructor is lazy. Check storage identity before advertising the
            # optional adapter; otherwise a lexical query can fail at first use.
            identity = getattr(embedder, 'cache_identity', None)
            if callable(identity):
                identity = identity()
            dense_status = {'state': 'MODEL_IDENTITY_AVAILABLE', 'identity': identity}
        except RuntimeError as exc:
            embedder = None
            dense_status = {'state': 'UNAVAILABLE', 'reason': str(exc),
                            'fallback': 'BM25; existing vectors retained and not reused across unknown versions'}
    if _enabled("PAPERPILOT_ENABLE_RERANKER", default=False):
        try:
            reranker = OnnxCrossEncoderReranker(
                cache_dir=reranker_cache,
                model_name=os.getenv(
                    "PAPERPILOT_RERANKER_MODEL", DEFAULT_RERANKER_MODEL
                ),
                model_file=os.getenv("PAPERPILOT_RERANKER_FILE", DEFAULT_RERANKER_FILE),
                revision=os.getenv("PAPERPILOT_RERANKER_REVISION", "main"),
                allow_download=_enabled("PAPERPILOT_ALLOW_MODEL_DOWNLOAD", default=False),
            )
        except RuntimeError:
            reranker = None

    if settings.memory_db and settings.memory_db.is_file():
        search_adapter = PaperPilotKnowledgeSearchAdapter(
            PaperKnowledgeStore(settings.memory_db, namespace=settings.memory_namespace),
            embedder=embedder,
            reranker=reranker,
            paper_roots=settings.paper_roots,
            rerank_stage=os.getenv("PAPERPILOT_RERANK_STAGE", "final"),
        )
    else:
        search_adapter = LocalCorpusSearchAdapter(settings.paper_roots)

    search_adapter.retrieval_status = {'dense': dense_status,
        'reranker_configured': reranker is not None}

    retrieval_identity = hashlib.sha256(json.dumps({
        'stage': getattr(search_adapter, 'rerank_stage', 'local'),
        'lexical': getattr(getattr(search_adapter, 'store', None), 'lexical_query_policy', None),
        'embedding': getattr(embedder, 'name', None),
        'reranker': getattr(reranker, 'name', None),
        'reranker_revision': getattr(reranker, 'requested_revision', None),
        'reranker_file': getattr(reranker, 'model_file', None),
    }, sort_keys=True).encode()).hexdigest()[:16]
    llm = (llm_factory or OpenAICompatibleLLM)(settings) if settings.api_key else None
    from .model_policy import policy_identity
    model_policy_sha = policy_identity(settings.provider, settings.model, settings.fallback_models,
                                       settings.base_url, 'daily')['sha256'][:16]
    vision = None
    qwen_key_file = os.getenv("PAPERPILOT_QWEN_API_KEY_FILE") or os.getenv(
        "PAPERPILOT_API_KEY_FILE"
    )
    if settings.provider == "qwen" and settings.api_key:
        os.environ.setdefault("DASHSCOPE_API_KEY", settings.api_key)
    try:
        if (os.getenv('PAPERPILOT_STRUCTURED_PARSER', 'legacy') != 'legacy' or
                os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or qwen_key_file):
            vision = (vision_factory or make_vision_adapter)(api_key_file=qwen_key_file)
    except (OSError, RuntimeError, ValueError):
        if vision_factory is not None or os.getenv('PAPERPILOT_STRUCTURED_PARSER', 'legacy') != 'legacy':
            raise  # A failed accounting factory must not fall back to an unwrapped adapter.
        vision = None
    if vision_factory is not None and vision is None:
        raise ValueError('Explicit visual provider factory did not supply an adapter')

    scholarly = (
        ScholarlySearchAdapter(settings.workspace / "scholarly-cache")
        if _enabled("PAPERPILOT_ALLOW_EXTERNAL_SCHOLARLY", default=False)
        else None
    )
    return ResearchAgentRuntime(
        db_path=settings.runtime_db,
        workspace=settings.workspace,
        search_adapter=search_adapter,
        scholarly_adapter=scholarly,
        vision_adapter=vision,
        llm=llm,
        experiment_runner=SandboxExperimentRunner(),
        text_reader_protocol=(f'answer-audit-v36-object-query:{settings.provider}:{settings.model}'
            f':retrieval-final-v1-{retrieval_identity}:model-policy-{model_policy_sha}:packet-v1'),
    )


def _enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}
