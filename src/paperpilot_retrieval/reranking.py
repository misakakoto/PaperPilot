"""Local cross-encoder reranking with explicit model provenance and bounded input."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Protocol

from .retrieval import ScoredChunk

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_FILE = "onnx/model_quint8_avx2.onnx"


class TextReranker(Protocol):
    """Minimal interface used by both the paper agent and public benchmarks."""

    @property
    def name(self) -> str: ...

    def warmup(self) -> None: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class OnnxCrossEncoderReranker:
    """CPU-only Hugging Face ONNX cross-encoder without a PyTorch dependency.

    Only a public ``organization/model`` identifier and a repository-relative
    ONNX file are accepted. The resolved snapshot commit and model SHA-256 are
    exposed for benchmark manifests, so ``main`` never becomes silent provenance.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model_file: str = DEFAULT_RERANKER_FILE,
        revision: str = "main",
        batch_size: int = 16,
        max_length: int = 512,
        allow_download: bool = True,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", model_name):
            raise ValueError("reranker model must be a public organization/model ID")
        if not re.fullmatch(r"[A-Za-z0-9._/-]+\.onnx", model_file):
            raise ValueError("reranker model file must be a repository-relative ONNX path")
        if model_file.startswith(("/", "\\")) or ".." in Path(model_file).parts:
            raise ValueError("reranker model file cannot escape the model repository")
        if not re.fullmatch(r"[A-Za-z0-9._/-]{1,160}", revision):
            raise ValueError("reranker revision contains unsupported characters")
        if not 1 <= batch_size <= 128:
            raise ValueError("reranker batch_size must be between 1 and 128")
        if not 32 <= max_length <= 512:
            raise ValueError("reranker max_length must be between 32 and 512")
        self._name = model_name
        self._model_file = model_file
        self._requested_revision = revision
        self._cache_dir = cache_dir.resolve()
        self._batch_size = batch_size
        self._max_length = max_length
        self._allow_download = allow_download
        self._tokenizer: object | None = None
        self._session: object | None = None
        self._resolved_revision: str | None = None
        self._model_sha256: str | None = None
        self._tokenizer_sha256: str | None = None
        self._load_lock = Lock()
        self._score_lock = Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def requested_revision(self) -> str:
        return self._requested_revision

    @property
    def resolved_revision(self) -> str | None:
        return self._resolved_revision

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    @property
    def tokenizer_sha256(self) -> str | None:
        return self._tokenizer_sha256

    @property
    def model_file(self) -> str:
        return self._model_file

    def warmup(self) -> None:
        self.score("scientific evidence", ["scientific evidence"])

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("reranker query cannot be empty")
        normalized_passages = [str(value).strip()[:20_000] for value in passages]
        if any(not value for value in normalized_passages):
            raise ValueError("reranker passages cannot be empty")
        if not normalized_passages:
            return []
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._session is not None

        import numpy as np

        output: list[float] = []
        with self._score_lock:
            for start in range(0, len(normalized_passages), self._batch_size):
                batch = normalized_passages[start : start + self._batch_size]
                encodings = self._tokenizer.encode_batch(  # type: ignore[union-attr]
                    [(normalized_query, passage) for passage in batch]
                )
                input_rows = {
                    "input_ids": np.asarray(
                        [encoding.ids for encoding in encodings], dtype=np.int64
                    ),
                    "attention_mask": np.asarray(
                        [encoding.attention_mask for encoding in encodings],
                        dtype=np.int64,
                    ),
                    "token_type_ids": np.asarray(
                        [encoding.type_ids for encoding in encodings], dtype=np.int64
                    ),
                }
                model_inputs: dict[str, object] = {}
                for item in self._session.get_inputs():  # type: ignore[union-attr]
                    if item.name not in input_rows:
                        raise RuntimeError(
                            f"unsupported reranker ONNX input: {item.name}"
                        )
                    model_inputs[item.name] = input_rows[item.name]
                raw = self._session.run(None, model_inputs)[0]  # type: ignore[union-attr]
                scores = np.asarray(raw, dtype=np.float32).reshape(-1).tolist()
                if len(scores) != len(batch) or not all(
                    math.isfinite(float(value)) for value in scores
                ):
                    raise RuntimeError("reranker returned invalid score dimensions")
                output.extend(float(value) for value in scores)
        return output

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            try:
                from huggingface_hub import hf_hub_download
                from onnxruntime import InferenceSession, SessionOptions
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "ONNX reranking requires huggingface-hub, onnxruntime, tokenizers, "
                    "and numpy; install with: pip install -e '.[benchmark]'"
                ) from exc

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs = {
                "repo_id": self._name,
                "revision": self._requested_revision,
                "cache_dir": str(self._cache_dir),
                "local_files_only": not self._allow_download,
            }
            try:
                model_download = Path(
                    hf_hub_download(filename=self._model_file, **kwargs)
                )
                # Hugging Face snapshot files are commonly symlinks into
                # ``blobs/``.  Capture the commit before ``resolve()`` erases
                # the ``snapshots/<commit>`` provenance from the path, then pin
                # the tokenizer download to that same immutable revision.
                model_revision = _snapshot_revision(model_download)
                tokenizer_kwargs = dict(kwargs)
                if model_revision is not None:
                    tokenizer_kwargs["revision"] = model_revision
                tokenizer_download = Path(
                    hf_hub_download(filename="tokenizer.json", **tokenizer_kwargs)
                )
            except Exception as exc:
                mode = "本地缓存" if not self._allow_download else "Hugging Face"
                raise RuntimeError(f"无法从{mode}加载重排模型 {self._name}: {exc}") from exc

            tokenizer_revision = _snapshot_revision(tokenizer_download)
            if model_revision and tokenizer_revision and model_revision != tokenizer_revision:
                raise RuntimeError("reranker model and tokenizer resolved to different revisions")
            model_path = model_download.resolve()
            tokenizer_path = tokenizer_download.resolve()
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=self._max_length)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            options = SessionOptions()
            session = InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._resolved_revision = model_revision or tokenizer_revision
            self._model_sha256 = _sha256_file(model_path)
            self._tokenizer_sha256 = _sha256_file(tokenizer_path)
            self._tokenizer = tokenizer
            self._session = session


def rerank_scored_chunks(
    query: str,
    candidates: Sequence[ScoredChunk],
    reranker: TextReranker,
    *,
    top_k: int,
) -> list[ScoredChunk]:
    """Rerank an already bounded candidate set without changing its membership."""

    if top_k <= 0 or not candidates:
        return []
    passages = [
        f"{item.chunk.section}\n{item.chunk.text}".strip() for item in candidates
    ]
    scores = reranker.score(query, passages)
    if len(scores) != len(candidates):
        raise RuntimeError("reranker score count does not match candidate count")
    ranked = sorted(
        enumerate(zip(candidates, scores, strict=True)),
        key=lambda item: (-float(item[1][1]), item[0]),
    )
    return [
        ScoredChunk(
            chunk=candidate.chunk,
            score=float(score),
            lexical_score=candidate.lexical_score,
            semantic_score=candidate.semantic_score,
            channels=tuple(dict.fromkeys((*candidate.channels, "cross_encoder"))),
        )
        for _, (candidate, score) in ranked[:top_k]
    ]


def _snapshot_revision(path: Path) -> str | None:
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    value = parts[index + 1]
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
