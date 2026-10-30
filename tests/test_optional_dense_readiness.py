from types import SimpleNamespace

import pytest

from paperpilot_agentic import bootstrap
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.models import PaperChunk, PaperDocument


@pytest.mark.parametrize("ready", [False, True])
def test_optional_dense_identity_is_checked_before_query_and_lexical_stays_usable(
    tmp_path, monkeypatch, ready
):
    class LazyEmbedding:
        name = "fixture-embedding"

        def __init__(self, **kwargs):
            pass

        @property
        def cache_identity(self):
            if not ready:
                raise RuntimeError("embedding model provenance is incomplete")
            return "fixture-frozen-model-v1"

        def embed_query(self, query):
            return [1.0, 0.0]

    monkeypatch.setattr(bootstrap, "FastEmbedder", LazyEmbedding)
    monkeypatch.setenv("PAPERPILOT_ENABLE_DENSE", "true")
    monkeypatch.setenv("PAPERPILOT_ENABLE_RERANKER", "false")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fixture-no-network")
    db = tmp_path / "papers.sqlite3"
    store = PaperKnowledgeStore(db, namespace="test")
    store.sync_documents(
        [
            PaperDocument(
                id="P1",
                file_name="paper.pdf",
                sha256="a" * 64,
                title="Alpha method",
                page_count=1,
                processed_pages=0,
                pages=[],
            )
        ],
        [
            PaperChunk(
                id="P1:p1:c1",
                paper_id="P1",
                page_start=1,
                page_end=1,
                section="Method",
                text="Alpha original method.",
            )
        ],
    )
    settings = RuntimeSettings(
        provider="qwen",
        api_key=None,
        model="fixture",
        base_url="https://example.invalid",
        thinking_enabled=False,
        temperature=0,
        max_tokens=100,
        timeout_seconds=1,
        workspace=tmp_path,
        runtime_db=tmp_path / "runs.sqlite3",
        memory_db=db,
        memory_namespace="test",
        paper_roots=(tmp_path,),
    )
    runtime = bootstrap.build_runtime(settings, vision_factory=lambda **kwargs: SimpleNamespace())
    assert runtime.search_adapter.search("Alpha original method", top_k=1)
    assert (runtime.search_adapter.embedder is not None) == ready
    assert runtime.llm_runtime is None
