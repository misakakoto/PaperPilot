"""Hybrid BM25 and local multilingual dense retrieval for paper chunks."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from array import array
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol

from .models import PaperChunk, PaperDocument

_LATIN_OR_NUMBER = re.compile(r"[a-z][a-z0-9_+.#/-]{1,}|\d+(?:\.\d+)?", re.IGNORECASE)
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")
DEFAULT_MULTILINGUAL_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_RRF_K = 60


class TextEmbedder(Protocol):
    """Small interface that keeps retrieval independent from one embedding library."""

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Lazy, CPU-only FastEmbed adapter with a content-addressed SQLite cache."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        model_name: str = DEFAULT_MULTILINGUAL_MODEL,
        allow_download: bool = True,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", model_name):
            raise ValueError("embedding model must be a public organization/model identifier")
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "semantic retrieval requires FastEmbed; install with: "
                "pip install -e '.[semantic]' or pass --no-semantic-retrieval"
            ) from exc
        self._text_embedding_type = TextEmbedding
        self._name = model_name
        self._cache_dir = cache_dir.resolve()
        self._model_cache = self._cache_dir / "models"
        self._database = self._cache_dir / "vectors.sqlite3"
        self._allow_download = allow_download
        self._model: object | None = None
        self._resolved_revision: str | None = None
        self._source_repository: str | None = None
        self._model_file: str | None = None
        self._model_sha256: str | None = None
        self._tokenizer_sha256: str | None = None
        try:
            self._fastembed_version = package_version("fastembed")
        except PackageNotFoundError:  # pragma: no cover - guarded by the import above
            self._fastembed_version = "unknown"
        self._backend_identity: str | None = None
        self._lock = Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def resolved_revision(self) -> str | None:
        return self._resolved_revision

    @property
    def source_repository(self) -> str | None:
        return self._source_repository

    @property
    def model_file(self) -> str | None:
        return self._model_file

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    @property
    def tokenizer_sha256(self) -> str | None:
        return self._tokenizer_sha256

    @property
    def cache_identity(self) -> str:
        """Immutable storage key for vectors persisted outside this adapter."""

        self._ensure_model()
        identity = self._cache_identity()
        if identity is None:
            raise RuntimeError(
                "embedding model provenance is incomplete; durable vector caching "
                "is disabled to prevent cross-version reuse"
            )
        return identity

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            return []
        self._ensure_model()
        cache_identity = self._cache_identity()
        if cache_identity is None:
            return self._embed(normalized_texts)
        digests = [_text_digest(text) for text in normalized_texts]
        cached = self._load_cached(digests, cache_identity)
        missing_positions = [
            index for index, digest in enumerate(digests) if digest not in cached
        ]
        if missing_positions:
            missing_texts = [normalized_texts[index] for index in missing_positions]
            vectors = self._embed(missing_texts)
            additions = {
                digests[index]: vector
                for index, vector in zip(missing_positions, vectors, strict=True)
            }
            self._save_cached(additions, cache_identity)
            cached.update(additions)
        return [list(cached[digest]) for digest in digests]

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        if not vectors:
            raise RuntimeError("embedding model returned no query vector")
        return vectors[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_model()
        with self._lock:
            assert self._model is not None
            raw_vectors = list(self._model.embed(list(texts)))  # type: ignore[union-attr]
        vectors = [_normalize_vector(vector) for vector in raw_vectors]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding model returned an unexpected vector count")
        return vectors

    def _ensure_model(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._model_cache.mkdir(parents=True, exist_ok=True)
            self._model = self._text_embedding_type(
                model_name=self._name,
                cache_dir=str(self._model_cache),
                local_files_only=not self._allow_download,
            )
            self._capture_model_provenance()

    def _cache_identity(self) -> str | None:
        if (
            self._model_sha256 is None
            or self._tokenizer_sha256 is None
            or self._backend_identity is None
        ):
            return None
        return (
            f"{self._name}@model-sha256:{self._model_sha256}"
            f"@tokenizer-sha256:{self._tokenizer_sha256}"
            f"@fastembed:{self._fastembed_version}"
            f"@backend:{self._backend_identity}"
        )

    def _capture_model_provenance(self) -> None:
        """Capture the immutable FastEmbed ONNX snapshot after lazy loading."""

        backend = getattr(self._model, "model", None)
        if backend is not None:
            backend_type = type(backend)
            self._backend_identity = (
                f"{backend_type.__module__}.{backend_type.__qualname__}"
            )
        model_dir_value = getattr(backend, "_model_dir", None)
        description = getattr(backend, "model_description", None)
        sources = getattr(description, "sources", None)
        source_repository = getattr(sources, "hf", None)
        model_file_value = getattr(description, "model_file", None)
        if model_dir_value is None or not isinstance(model_file_value, str):
            return
        model_dir = Path(model_dir_value)
        relative_model = Path(model_file_value)
        if relative_model.is_absolute() or ".." in relative_model.parts:
            return
        model_path = model_dir / relative_model
        try:
            resolved_model = model_path.resolve(strict=True)
        except OSError:
            return
        # Snapshot entries can legitimately be symlinks into Hugging Face's
        # sibling ``blobs`` directory, so validate the repository-relative
        # lexical path above and hash the resolved file here.
        if not resolved_model.is_file():
            return
        self._resolved_revision = _cache_snapshot_revision(model_dir)
        self._source_repository = (
            source_repository
            if isinstance(source_repository, str) and source_repository.strip()
            else None
        )
        self._model_file = model_file_value
        self._model_sha256 = _sha256_path(resolved_model)
        tokenizer_path = model_dir / "tokenizer.json"
        try:
            resolved_tokenizer = tokenizer_path.resolve(strict=True)
        except OSError:
            return
        if resolved_tokenizer.is_file():
            self._tokenizer_sha256 = _sha256_path(resolved_tokenizer)

    def _connection(self) -> sqlite3.Connection:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_embeddings (
                model TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model, content_sha256)
            )
            """
        )
        return connection

    def _load_cached(
        self,
        digests: Sequence[str],
        model_identity: str,
    ) -> dict[str, list[float]]:
        if not self._database.is_file() or not digests:
            return {}
        found: dict[str, list[float]] = {}
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connection()
            for digest in dict.fromkeys(digests):
                row = connection.execute(
                    "SELECT dimension, vector FROM paper_embeddings "
                    "WHERE model = ? AND content_sha256 = ?",
                    (model_identity, digest),
                ).fetchone()
                if row is None:
                    continue
                dimension, blob = row
                vector = array("f")
                vector.frombytes(blob)
                if dimension <= 0 or len(vector) != dimension:
                    continue
                values = [float(value) for value in vector]
                if all(math.isfinite(value) for value in values):
                    found[digest] = values
        except (OSError, sqlite3.DatabaseError, ValueError):
            return {}
        finally:
            if connection is not None:
                connection.close()
        return found

    def _save_cached(
        self,
        values: dict[str, list[float]],
        model_identity: str,
    ) -> None:
        if not values:
            return
        connection = self._connection()
        try:
            connection.executemany(
                "INSERT OR REPLACE INTO paper_embeddings "
                "(model, content_sha256, dimension, vector) VALUES (?, ?, ?, ?)",
                [
                    (
                        model_identity,
                        digest,
                        len(vector),
                        array("f", vector).tobytes(),
                    )
                    for digest, vector in values.items()
                ],
            )
            connection.commit()
        finally:
            connection.close()


@dataclass(frozen=True)
class ScoredChunk:
    chunk: PaperChunk
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalTrace:
    """Ranked results plus the independent channel leaders used for gating."""

    results: tuple[ScoredChunk, ...]
    lexical_top_id: str | None = None
    semantic_top_id: str | None = None


class PaperIndex:
    """In-memory hybrid index suitable for tens to hundreds of papers."""

    def __init__(
        self,
        chunks: list[PaperChunk],
        *,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self._texts = [f"{chunk.section} {chunk.text}".strip() for chunk in chunks]
        self._tokens = [_tokenize(text) for text in self._texts]
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._average_length = (
            sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 1.0
        )
        self._dense_vectors: list[list[float]] | None = None
        self._dense_matrix: object | None = None

    @property
    def retrieval_mode(self) -> str:
        return "hybrid_rrf" if self.embedder is not None else "bm25"

    @property
    def semantic_model(self) -> str | None:
        return self.embedder.name if self.embedder is not None else None

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        paper_ids: set[str] | None = None,
        mode: Literal["auto", "bm25", "dense", "hybrid"] = "auto",
    ) -> list[ScoredChunk]:
        return list(
            self.search_with_trace(
                query,
                top_k=top_k,
                paper_ids=paper_ids,
                mode=mode,
            ).results
        )

    def search_with_trace(
        self,
        query: str,
        *,
        top_k: int = 8,
        paper_ids: set[str] | None = None,
        mode: Literal["auto", "bm25", "dense", "hybrid"] = "auto",
    ) -> RetrievalTrace:
        """Search once and expose channel agreement without duplicating retrieval.

        The trace intentionally contains IDs and scores only. It is used by the
        selective reranker to decide whether a costly second-stage model is useful;
        it does not bypass the normal ranked-result path.
        """

        if top_k <= 0:
            return RetrievalTrace(results=())
        effective_mode = (
            "hybrid" if mode == "auto" and self.embedder is not None else mode
        )
        if effective_mode == "auto":
            effective_mode = "bm25"
        lexical = (
            self._lexical_scores(query, paper_ids=paper_ids)
            if effective_mode in {"bm25", "hybrid"}
            else []
        )
        lexical_top_id = lexical[0].chunk.id if lexical else None
        if effective_mode == "bm25":
            return RetrievalTrace(
                results=tuple(lexical[:top_k]),
                lexical_top_id=lexical_top_id,
            )
        if self.embedder is None:
            raise ValueError(f"retrieval mode {effective_mode!r} requires an embedder")
        semantic = self._semantic_scores(query, paper_ids=paper_ids)
        semantic_top_id = semantic[0].chunk.id if semantic else None
        if effective_mode == "dense":
            return RetrievalTrace(
                results=tuple(semantic[:top_k]),
                semantic_top_id=semantic_top_id,
            )
        if effective_mode != "hybrid":
            raise ValueError(f"unsupported retrieval mode: {mode}")
        candidate_count = max(top_k * 4, 20)
        lexical = lexical[:candidate_count]
        semantic = semantic[:candidate_count]
        combined: dict[str, dict[str, object]] = {}
        for channel, ranked in (("bm25", lexical), ("dense", semantic)):
            for rank, item in enumerate(ranked, start=1):
                entry = combined.setdefault(
                    item.chunk.id,
                    {
                        "chunk": item.chunk,
                        "score": 0.0,
                        "lexical": 0.0,
                        "semantic": 0.0,
                        "channels": [],
                    },
                )
                entry["score"] = float(entry["score"]) + 1.0 / (_RRF_K + rank)
                entry["channels"].append(channel)  # type: ignore[union-attr]
                if channel == "bm25":
                    entry["lexical"] = item.score
                else:
                    entry["semantic"] = item.score
        results = [
            ScoredChunk(
                chunk=entry["chunk"],  # type: ignore[arg-type]
                score=float(entry["score"]),
                lexical_score=float(entry["lexical"]),
                semantic_score=float(entry["semantic"]),
                channels=tuple(entry["channels"]),  # type: ignore[arg-type]
            )
            for entry in combined.values()
        ]
        results.sort(key=lambda item: (-item.score, item.chunk.id))
        return RetrievalTrace(
            results=tuple(results[:top_k]),
            lexical_top_id=lexical_top_id,
            semantic_top_id=semantic_top_id,
        )

    def analysis_context(
        self,
        documents: list[PaperDocument],
        *,
        question: str | None = None,
        max_chars: int = 60_000,
        chunks_per_query: int = 3,
    ) -> list[PaperChunk]:
        """Select broad, balanced context instead of dumping full papers into the LLM."""

        queries = [
            "abstract research problem motivation contribution 摘要 问题 贡献",
            "method architecture model module algorithm 方法 架构 模型",
            "training objective loss optimization hyperparameter 训练 损失 优化",
            "experiment dataset baseline metric result ablation 实验 数据 指标 消融",
            "limitation discussion conclusion future 局限 讨论 结论",
        ]
        if question:
            queries.insert(0, question)

        selected: dict[str, PaperChunk] = {}
        for document in documents:
            paper_ids = {document.id}
            first = next((chunk for chunk in self.chunks if chunk.paper_id == document.id), None)
            if first:
                selected[first.id] = first
            for query in queries:
                for item in self.search(query, top_k=chunks_per_query, paper_ids=paper_ids):
                    selected[item.chunk.id] = item.chunk

        ordered = sorted(
            selected.values(), key=lambda item: (int(item.paper_id[1:]), item.page_start, item.id)
        )
        accepted: list[PaperChunk] = []
        used = 0
        for chunk in ordered:
            if used + len(chunk.text) > max_chars:
                remaining = max_chars - used
                if remaining >= 500:
                    accepted.append(chunk.model_copy(update={"text": chunk.text[:remaining]}))
                break
            accepted.append(chunk)
            used += len(chunk.text)
        return accepted

    def _lexical_scores(
        self, query: str, *, paper_ids: set[str] | None
    ) -> list[ScoredChunk]:
        query_tokens = list(dict.fromkeys(_tokenize(query)))
        if not query_tokens:
            return []
        count = len(self.chunks)
        scores: list[ScoredChunk] = []
        for chunk, tokens, frequencies in zip(
            self.chunks, self._tokens, self._term_counts, strict=True
        ):
            if paper_ids is not None and chunk.paper_id not in paper_ids:
                continue
            score = 0.0
            length_ratio = len(tokens) / max(self._average_length, 1.0)
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                doc_frequency = self._document_frequency[token]
                inverse = math.log(1 + (count - doc_frequency + 0.5) / (doc_frequency + 0.5))
                score += inverse * (frequency * 2.2) / (
                    frequency + 1.2 * (1 - 0.75 + 0.75 * length_ratio)
                )
            if score > 0:
                scores.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=score,
                        lexical_score=score,
                        channels=("bm25",),
                    )
                )
        scores.sort(key=lambda item: (-item.score, item.chunk.id))
        return scores

    def _semantic_scores(
        self, query: str, *, paper_ids: set[str] | None
    ) -> list[ScoredChunk]:
        assert self.embedder is not None
        if not query.strip() or not self.chunks:
            return []
        if self._dense_vectors is None:
            self._dense_vectors = self.embedder.embed_documents(self._texts)
            if len(self._dense_vectors) != len(self.chunks):
                raise RuntimeError("semantic index vector count does not match paper chunks")
        query_vector = _normalize_vector(self.embedder.embed_query(query))
        similarities = _dense_similarities(
            self._dense_vectors,
            query_vector,
            cached_matrix=self._dense_matrix,
        )
        if self._dense_matrix is None:
            self._dense_matrix = similarities.matrix
        scores: list[ScoredChunk] = []
        for chunk, similarity in zip(
            self.chunks,
            similarities.values,
            strict=True,
        ):
            if paper_ids is not None and chunk.paper_id not in paper_ids:
                continue
            scores.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(similarity),
                    semantic_score=float(similarity),
                    channels=("dense",),
                )
            )
        scores.sort(key=lambda item: (-item.score, item.chunk.id))
        return scores


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _cache_snapshot_revision(path: Path) -> str | None:
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    value = parts[index + 1]
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_vector(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise ValueError("embedding vector has zero norm")
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class _DenseSimilarities:
    values: list[float]
    matrix: object | None


def _dense_similarities(
    vectors: Sequence[Sequence[float]],
    query: Sequence[float],
    *,
    cached_matrix: object | None,
) -> _DenseSimilarities:
    """Use exact NumPy matrix multiplication when available, with a safe fallback.

    FastEmbed already depends on NumPy, but keeping the import lazy preserves the
    small protocol-based test embedders and the BM25-only base installation. The
    vectors are normalized before this function, so matrix multiplication remains
    mathematically equivalent to the previous per-document cosine loop.
    """

    try:
        import numpy as np
    except ImportError:
        return _DenseSimilarities(
            values=[_dot(query, vector) for vector in vectors],
            matrix=None,
        )

    matrix = cached_matrix
    if matrix is None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(vectors):
            raise ValueError("embedding matrix must be two-dimensional")
    query_array = np.asarray(query, dtype=np.float32)
    if query_array.ndim != 1 or matrix.shape[1] != query_array.shape[0]:
        raise ValueError("embedding dimensions do not match")
    raw = matrix @ query_array
    return _DenseSimilarities(
        values=[float(value) for value in raw.tolist()],
        matrix=matrix,
    )


def _tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    tokens = [item for item in _LATIN_OR_NUMBER.findall(lowered) if len(item) > 1]
    for sequence in _CHINESE.findall(lowered):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens
