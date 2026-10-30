"""Persistent, evidence-governed memory for local-paper research.

The store deliberately separates two concepts:

* paper chunks are durable source evidence identified by a content-derived ID;
* memories are derived claims that are only used as retrieval hints and always keep
  links back to one or more durable source chunks.

Run-local evidence IDs such as ``P1:p3:c1`` remain useful in reports, but they are
not stable when a corpus is reordered.  This module therefore anchors persistence to
the paper SHA-256, page, chunk ordinal, and chunk text hash instead.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import sqlite3
from .lexical_query import content_query
from array import array
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from .models import (
    EvidencePoint,
    PaperAnalysis,
    PaperAnswer,
    PaperChunk,
    PaperDocument,
)
from .render import audit_paper_analysis, audit_paper_answer
from .reranking import TextReranker
from .retrieval import TextEmbedder

MemoryStatus = Literal["active", "superseded", "stale", "tombstoned"]
MemoryRelation = Literal["supports", "contradicts", "supersedes"]
RetrievalMode = Literal["auto", "bm25", "dense", "hybrid"]

_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CHUNK_INDEX = re.compile(r":c(?P<ordinal>\d+)$")
_LATIN_OR_NUMBER = re.compile(
    r"[a-z][a-z0-9_+.#/-]{1,}|\d+(?:\.\d+)?", re.IGNORECASE
)
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")
_RUN_CITATION = re.compile(r"\[?P\d+:p\d+(?:-p\d+)?:c\d+\]?")
_WHITESPACE = re.compile(r"\s+")
_RRF_K = 60
_SCHEMA_VERSION = 1
_MEMORY_REDACTION = "[来源已删除，派生记忆已撤回]"
_TITLE_GENERIC_TERMS = {
    "analysis",
    "approach",
    "based",
    "change",
    "data",
    "dataset",
    "deep",
    "detection",
    "evidence",
    "experiment",
    "framework",
    "image",
    "learning",
    "method",
    "model",
    "network",
    "paper",
    "remote",
    "results",
    "sensing",
    "study",
    "using",
}
_TITLE_QUERY_LEADING_SKIP = {
    "are",
    "compare",
    "describe",
    "does",
    "explain",
    "how",
    "is",
    "please",
    "the",
    "what",
    "where",
    "which",
    "why",
}


class _ClosingConnection(sqlite3.Connection):
    """Make ``with connection`` close file handles as well as commit/rollback."""

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, traceback))
        finally:
            self.close()


@dataclass(frozen=True)
class LibrarySyncResult:
    namespace: str
    papers_seen: int
    papers_added: int
    papers_updated: int
    papers_unchanged: int
    chunks_written: int
    embeddings_written: int


@dataclass(frozen=True)
class StoredChunkHit:
    stable_id: str
    paper_sha256: str
    file_name: str
    title: str
    page_start: int
    page_end: int
    ordinal: int
    section: str
    text: str
    token_count_estimate: int
    score: float = 0.0
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperRouteHit:
    """One paper-level candidate aggregated from independent chunk queries."""

    paper_sha256: str
    file_name: str
    title: str
    score: float
    matched_query_indexes: tuple[int, ...]
    best_stable_ids: tuple[str, ...]


@dataclass(frozen=True)
class PaperRoutingResult:
    queries: tuple[str, ...]
    hits: tuple[PaperRouteHit, ...]
    query_hit_counts: tuple[int, ...]


@dataclass(frozen=True)
class MultiQueryRetrievalResult:
    queries: tuple[str, ...]
    hits: tuple[StoredChunkHit, ...]
    paper_hits: tuple[PaperRouteHit, ...]
    recalled_memory_ids: tuple[str, ...]
    query_hit_counts: tuple[int, ...]
    memory_representative_count: int = 0
    multi_query_expansion_count: int = 0
    reranked_query_count: int = 0
    rerank_skipped_query_count: int = 0


@dataclass(frozen=True)
class MemoryEvidenceRef:
    stable_id: str
    paper_sha256: str
    file_name: str
    title: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    kind: str
    content: str
    confidence: str
    status: MemoryStatus
    source_run_id: str | None
    score: float
    evidence: tuple[MemoryEvidenceRef, ...]


@dataclass(frozen=True)
class MemoryWriteResult:
    namespace: str
    gate_passed: bool
    inserted: int
    duplicates: int
    skipped: int
    reason: str = ""


@dataclass(frozen=True)
class ForgetResult:
    namespace: str
    paper_sha256: str
    papers_removed: int
    chunks_removed: int
    memories_tombstoned: int


@dataclass(frozen=True)
class _MemoryCandidate:
    kind: str
    content: str
    evidence_ids: tuple[str, ...]
    confidence: str


class PaperKnowledgeStore:
    """SQLite-backed incremental corpus and evidence-derived long-term memory."""

    def __init__(self, path: Path, *, namespace: str = "default",
                 lexical_query_policy: str = "content_v1") -> None:
        if lexical_query_policy not in {'raw', 'content_v1'}:
            raise ValueError('Unknown lexical query policy')
        self.lexical_query_policy = lexical_query_policy
        self.path = path.resolve()
        self.namespace = _validate_namespace(namespace)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    processed_pages INTEGER NOT NULL,
                    warnings_json TEXT NOT NULL,
                    chunk_fingerprint TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(namespace, sha256)
                );

                CREATE INDEX IF NOT EXISTS papers_namespace_active
                    ON papers(namespace, active, updated_at);

                CREATE TABLE IF NOT EXISTS chunks (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    paper_row_id INTEGER NOT NULL REFERENCES papers(row_id) ON DELETE CASCADE,
                    stable_id TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    token_count_estimate INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(namespace, stable_id)
                );

                CREATE INDEX IF NOT EXISTS chunks_namespace_paper
                    ON chunks(namespace, paper_row_id, page_start, ordinal);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    search_text,
                    content='chunks',
                    content_rowid='row_id'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunk_fts(rowid, search_text)
                    VALUES (new.row_id, new.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunk_fts(chunk_fts, rowid, search_text)
                    VALUES ('delete', old.row_id, old.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunk_fts(chunk_fts, rowid, search_text)
                    VALUES ('delete', old.row_id, old.search_text);
                    INSERT INTO chunk_fts(rowid, search_text)
                    VALUES (new.row_id, new.search_text);
                END;

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_row_id INTEGER NOT NULL REFERENCES chunks(row_id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(chunk_row_id, model)
                );

                CREATE INDEX IF NOT EXISTS embeddings_model
                    ON chunk_embeddings(model, chunk_row_id);

                CREATE TABLE IF NOT EXISTS memories (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    normalized_sha256 TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_run_id TEXT,
                    supersedes_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(namespace, memory_id)
                );

                CREATE INDEX IF NOT EXISTS memories_namespace_status
                    ON memories(namespace, status, updated_at);
                CREATE INDEX IF NOT EXISTS memories_normalized
                    ON memories(namespace, normalized_sha256, status);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    search_text,
                    content='memories',
                    content_rowid='row_id'
                );

                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memory_fts(rowid, search_text)
                    VALUES (new.row_id, new.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, search_text)
                    VALUES ('delete', old.row_id, old.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memory_fts(memory_fts, rowid, search_text)
                    VALUES ('delete', old.row_id, old.search_text);
                    INSERT INTO memory_fts(rowid, search_text)
                    VALUES (new.row_id, new.search_text);
                END;

                CREATE TABLE IF NOT EXISTS memory_evidence (
                    memory_row_id INTEGER NOT NULL REFERENCES memories(row_id) ON DELETE CASCADE,
                    chunk_row_id INTEGER NOT NULL REFERENCES chunks(row_id) ON DELETE CASCADE,
                    PRIMARY KEY(memory_row_id, chunk_row_id)
                );

                CREATE INDEX IF NOT EXISTS memory_evidence_chunk
                    ON memory_evidence(chunk_row_id, memory_row_id);

                CREATE TABLE IF NOT EXISTS memory_relations (
                    namespace TEXT NOT NULL,
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, source_memory_id, target_memory_id, relation)
                );

                CREATE TABLE IF NOT EXISTS memory_events (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS memory_events_namespace
                    ON memory_events(namespace, created_at);
                """
            )
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

    def sync_documents(
        self,
        documents: Sequence[PaperDocument],
        chunks: Sequence[PaperChunk],
        *,
        embedder: TextEmbedder | None = None,
    ) -> LibrarySyncResult:
        """Idempotently add or refresh documents, then index missing vectors."""

        document_by_id = {document.id: document for document in documents}
        if len(document_by_id) != len(documents):
            raise ValueError("paper document IDs must be unique during memory sync")
        grouped: dict[str, list[PaperChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.paper_id not in document_by_id:
                raise ValueError(f"chunk {chunk.id} references an unknown paper")
            grouped[chunk.paper_id].append(chunk)

        added = 0
        updated = 0
        unchanged = 0
        chunks_written = 0
        stable_ids: list[str] = []
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for document in documents:
                paper_chunks = sorted(
                    grouped.get(document.id, []),
                    key=lambda item: (item.page_start, _chunk_ordinal(item), item.id),
                )
                fingerprint = _chunk_fingerprint(document.sha256, paper_chunks)
                existing = connection.execute(
                    "SELECT row_id, chunk_fingerprint, active FROM papers "
                    "WHERE namespace = ? AND sha256 = ?",
                    (self.namespace, document.sha256),
                ).fetchone()
                if existing is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO papers(
                            namespace, sha256, file_name, title, authors_json,
                            page_count, processed_pages, warnings_json,
                            chunk_fingerprint, active, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            self.namespace,
                            document.sha256,
                            document.file_name,
                            document.title,
                            json.dumps(document.authors, ensure_ascii=False),
                            document.page_count,
                            document.processed_pages,
                            json.dumps(document.warnings, ensure_ascii=False),
                            fingerprint,
                            now,
                            now,
                        ),
                    )
                    paper_row_id = int(cursor.lastrowid)
                    added += 1
                else:
                    paper_row_id = int(existing["row_id"])
                    same_chunks = existing["chunk_fingerprint"] == fingerprint
                    if same_chunks and bool(existing["active"]):
                        unchanged += 1
                    else:
                        self._mark_memories_stale_for_paper(connection, paper_row_id)
                        connection.execute(
                            "DELETE FROM chunks WHERE paper_row_id = ?", (paper_row_id,)
                        )
                        updated += 1
                    connection.execute(
                        """
                        UPDATE papers SET file_name = ?, title = ?, authors_json = ?,
                            page_count = ?, processed_pages = ?, warnings_json = ?,
                            chunk_fingerprint = ?, active = 1, updated_at = ?
                        WHERE row_id = ?
                        """,
                        (
                            document.file_name,
                            document.title,
                            json.dumps(document.authors, ensure_ascii=False),
                            document.page_count,
                            document.processed_pages,
                            json.dumps(document.warnings, ensure_ascii=False),
                            fingerprint,
                            now,
                            paper_row_id,
                        ),
                    )

                rows_exist = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE paper_row_id = ?",
                    (paper_row_id,),
                ).fetchone()[0]
                if rows_exist == 0 and paper_chunks:
                    for chunk in paper_chunks:
                        stable_id = stable_chunk_id(document.sha256, chunk)
                        text_sha256 = _sha256_text(chunk.text)
                        connection.execute(
                            """
                            INSERT INTO chunks(
                                namespace, paper_row_id, stable_id, page_start,
                                page_end, ordinal, section, text, search_text,
                                text_sha256, token_count_estimate, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                self.namespace,
                                paper_row_id,
                                stable_id,
                                chunk.page_start,
                                chunk.page_end,
                                _chunk_ordinal(chunk),
                                chunk.section,
                                chunk.text,
                                _search_text(f"{chunk.section} {chunk.text}"),
                                text_sha256,
                                chunk.token_count_estimate,
                                now,
                            ),
                        )
                        chunks_written += 1
                        stable_ids.append(stable_id)
                else:
                    stable_ids.extend(
                        row[0]
                        for row in connection.execute(
                            "SELECT stable_id FROM chunks WHERE paper_row_id = ?",
                            (paper_row_id,),
                        ).fetchall()
                    )
                self._event(
                    connection,
                    "paper.synced",
                    document.sha256,
                    {
                        "file_name": document.file_name,
                        "chunks": len(paper_chunks),
                        "fingerprint": fingerprint,
                    },
                )
            connection.commit()

        embeddings_written = 0
        if embedder is not None and stable_ids:
            embeddings_written = self.ensure_embeddings(embedder, stable_ids=stable_ids)
        return LibrarySyncResult(
            namespace=self.namespace,
            papers_seen=len(documents),
            papers_added=added,
            papers_updated=updated,
            papers_unchanged=unchanged,
            chunks_written=chunks_written,
            embeddings_written=embeddings_written,
        )

    def ensure_embeddings(
        self,
        embedder: TextEmbedder,
        *,
        stable_ids: Sequence[str] | None = None,
        batch_size: int = 64,
    ) -> int:
        """Persist normalized embeddings missing for one model without holding all vectors."""

        if not 1 <= batch_size <= 512:
            raise ValueError("embedding batch_size must be between 1 and 512")
        model_identity = _embedding_storage_identity(embedder)
        params: list[object] = [model_identity, self.namespace]
        where = "e.chunk_row_id IS NULL AND c.namespace = ? AND p.active = 1"
        if stable_ids:
            unique = list(dict.fromkeys(stable_ids))
            placeholders = ",".join("?" for _ in unique)
            where += f" AND c.stable_id IN ({placeholders})"
            params.extend(unique)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.row_id, c.stable_id, c.section, c.text
                FROM chunks c
                JOIN papers p ON p.row_id = c.paper_row_id
                LEFT JOIN chunk_embeddings e
                    ON e.chunk_row_id = c.row_id AND e.model = ?
                WHERE {where}
                ORDER BY c.row_id
                """,
                params,
            ).fetchall()
        written = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [f"{row['section']} {row['text']}".strip() for row in batch]
            vectors = embedder.embed_documents(texts)
            if len(vectors) != len(batch):
                raise RuntimeError("embedding model returned an unexpected vector count")
            payload = []
            for row, vector in zip(batch, vectors, strict=True):
                normalized = _normalize_vector(vector)
                payload.append(
                    (
                        int(row["row_id"]),
                        model_identity,
                        len(normalized),
                        array("f", normalized).tobytes(),
                        _now(),
                    )
                )
            with self._lock, self._connect() as connection:
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO chunk_embeddings"
                    "(chunk_row_id, model, dimension, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    payload,
                )
                written += connection.total_changes - before
        return int(written)

    def search_chunks(
        self,
        query: str,
        *,
        top_k: int = 12,
        embedder: TextEmbedder | None = None,
        paper_sha256s: set[str] | None = None,
        mode: RetrievalMode = "auto",
        max_chunks_per_paper: int | None = None,
        reranker: TextReranker | None = None,
    ) -> list[StoredChunkHit]:
        """Search the durable corpus with FTS5 BM25 and bounded-memory dense scoring."""

        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if mode not in {"auto", "bm25", "dense", "hybrid"}:
            raise ValueError(f"unsupported retrieval mode: {mode}")
        if max_chunks_per_paper is not None and not 1 <= max_chunks_per_paper <= 20:
            raise ValueError("max_chunks_per_paper must be between 1 and 20")
        effective_mode = "hybrid" if mode == "auto" and embedder is not None else mode
        if effective_mode == "auto":
            effective_mode = "bm25"
        if effective_mode in {"dense", "hybrid"} and embedder is None:
            raise ValueError(f"retrieval mode {effective_mode!r} requires an embedder")
        candidate_count = max(
            top_k * (8 if max_chunks_per_paper is not None else 4),
            50 if max_chunks_per_paper is not None else 20,
        )
        lexical = (
            self._lexical_chunk_rows(query, candidate_count, paper_sha256s)
            if effective_mode in {"bm25", "hybrid"}
            else []
        )
        title_scope = self._distinctive_title_scope(query, paper_sha256s)
        title_lexical = (
            self._lexical_chunk_rows(query, candidate_count, title_scope)
            if title_scope and effective_mode in {"bm25", "hybrid"}
            else []
        )
        dense = (
            self._dense_chunk_rows(
                query,
                embedder,
                candidate_count,
                paper_sha256s,
            )
            if effective_mode in {"dense", "hybrid"} and embedder is not None
            else []
        )
        ranked_channels: list[tuple[str, list[tuple[str, float]], float]] = []
        if lexical:
            ranked_channels.append(
                (
                    "bm25",
                    [(row["stable_id"], float(row["raw_score"])) for row in lexical],
                    1.0,
                )
            )
        if title_lexical:
            ranked_channels.append(
                (
                    "title_exact",
                    [
                        (row["stable_id"], float(row["raw_score"]))
                        for row in title_lexical
                    ],
                    3.0,
                )
            )
        if dense:
            ranked_channels.append(("dense", dense, 1.0))
        if not ranked_channels:
            return []
        combined: dict[str, dict[str, object]] = {}
        for channel, ranked, weight in ranked_channels:
            for rank, (stable_id, raw_score) in enumerate(ranked, 1):
                entry = combined.setdefault(
                    stable_id,
                    {"score": 0.0, "lexical": 0.0, "semantic": 0.0, "channels": []},
                )
                entry["score"] = float(entry["score"]) + weight / (_RRF_K + rank)
                entry["channels"].append(channel)  # type: ignore[union-attr]
                if channel in {"bm25", "title_exact"}:
                    entry["lexical"] = raw_score
                else:
                    entry["semantic"] = raw_score
        ranked_ids = sorted(
            combined,
            key=lambda value: (-float(combined[value]["score"]), value),
        )
        row_limit = (
            candidate_count
            if reranker is not None or max_chunks_per_paper is not None
            else top_k
        )
        rows = self._chunk_rows_by_stable_ids(ranked_ids[:row_limit])
        by_id = {row["stable_id"]: row for row in rows}
        if reranker is not None:
            lexical_top_id = next(
                (
                    ranked[0][0]
                    for channel, ranked, _weight in ranked_channels
                    if channel == "bm25" and ranked
                ),
                None,
            )
            semantic_top_id = next(
                (
                    ranked[0][0]
                    for channel, ranked, _weight in ranked_channels
                    if channel == "dense" and ranked
                ),
                None,
            )
            if (
                lexical_top_id is not None
                and lexical_top_id == semantic_top_id
            ):
                for stable_id in ranked_ids[:row_limit]:
                    entry = combined[stable_id]
                    channels = entry["channels"]
                    if "rerank_skipped_top1_agreement" not in channels:  # type: ignore[operator]
                        channels.append(  # type: ignore[union-attr]
                            "rerank_skipped_top1_agreement"
                        )
            else:
                candidate_ids = [
                    stable_id
                    for stable_id in ranked_ids[:row_limit]
                    if stable_id in by_id
                ]
                scores = reranker.score(
                    query,
                    [
                        f"{by_id[stable_id]['section']}\n{by_id[stable_id]['text']}".strip()
                        for stable_id in candidate_ids
                    ],
                )
                if len(scores) != len(candidate_ids):
                    raise RuntimeError(
                        "reranker score count does not match persistent candidates"
                    )
                if not all(math.isfinite(float(score)) for score in scores):
                    raise RuntimeError("reranker returned a non-finite persistent score")
                original_rank = {
                    stable_id: index for index, stable_id in enumerate(candidate_ids)
                }
                rerank_scores = dict(zip(candidate_ids, scores, strict=True))
                ranked_ids = sorted(
                    candidate_ids,
                    key=lambda stable_id: (
                        -float(rerank_scores[stable_id]),
                        original_rank[stable_id],
                    ),
                )
                for stable_id, score in rerank_scores.items():
                    entry = combined[stable_id]
                    entry["score"] = float(score)
                    channels = entry["channels"]
                    if "cross_encoder" not in channels:  # type: ignore[operator]
                        channels.append("cross_encoder")  # type: ignore[union-attr]
        ordered_ids: list[str] = []
        per_paper: dict[str, int] = {}
        for stable_id in ranked_ids:
            row = by_id.get(stable_id)
            if row is None:
                continue
            paper_sha = str(row["sha256"])
            if (
                max_chunks_per_paper is not None
                and per_paper.get(paper_sha, 0) >= max_chunks_per_paper
            ):
                continue
            ordered_ids.append(stable_id)
            per_paper[paper_sha] = per_paper.get(paper_sha, 0) + 1
            if len(ordered_ids) >= top_k:
                break
        results = []
        for stable_id in ordered_ids:
            row = by_id.get(stable_id)
            if row is None:
                continue
            entry = combined[stable_id]
            results.append(
                _stored_chunk_hit(
                    row,
                    score=float(entry["score"]),
                    lexical_score=float(entry["lexical"]),
                    semantic_score=float(entry["semantic"]),
                    channels=tuple(entry["channels"]),  # type: ignore[arg-type]
                )
            )
        return results

    def _distinctive_title_scope(
        self,
        query: str,
        paper_sha256s: set[str] | None,
    ) -> set[str]:
        """Find papers named explicitly by a rare Latin identifier in the query.

        Chunk FTS deliberately stores source text rather than mutable metadata.  A
        model or method name such as ``ChangeMamba`` can therefore be absent from a
        relevant method page. This lightweight metadata channel only considers the
        first substantive Latin term, verifies that the query or stored metadata
        spells it like a model identifier, and gives matching chunks an extra RRF
        vote; it never removes the global retrieval anchor.
        """

        raw_terms = [
            match.group(0)
            for match in _LATIN_OR_NUMBER.finditer(query)
            if len(match.group(0)) >= 4
            and match.group(0).casefold() not in _TITLE_QUERY_LEADING_SKIP
        ]
        if not raw_terms:
            return set()
        leading_term = raw_terms[0]
        query_term = leading_term.casefold()
        if query_term in _TITLE_GENERIC_TERMS:
            return set()
        where = "namespace = ? AND active = 1"
        params: list[object] = [self.namespace]
        if paper_sha256s is not None:
            if not paper_sha256s:
                return set()
            placeholders = ",".join("?" for _ in paper_sha256s)
            where += f" AND sha256 IN ({placeholders})"
            params.extend(sorted(paper_sha256s))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT sha256, file_name, title FROM papers WHERE {where}",
                params,
            ).fetchall()
        if not rows:
            return set()
        matches: set[str] = set()
        for row in rows:
            metadata = f"{row['title']} {row['file_name']}"
            metadata_terms = [
                match.group(0) for match in _LATIN_OR_NUMBER.finditer(metadata)
            ]
            matching_terms = [
                term for term in metadata_terms if term.casefold() == query_term
            ]
            if matching_terms and (
                _looks_like_named_identifier(leading_term)
                or any(_looks_like_named_identifier(term) for term in matching_terms)
            ):
                matches.add(str(row["sha256"]))
        if not matches:
            return set()
        return matches if len(matches) <= 8 else set()

    def resolve_explicit_title_scope(
        self,
        query: str,
        paper_sha256s: set[str] | None = None,
    ) -> set[str]:
        """Resolve a named paper/model in ``query`` to a small paper SHA set.

        This public wrapper lets higher-level research orchestration distinguish a
        user-selected source from a mere lexical ranking hint.  It intentionally
        reuses the conservative identifier rules above and returns no scope when a
        term is generic or ambiguously matches more than eight papers.
        """

        # A planner often writes claims as "The ablation results in ... of the
        # FlowMamba paper". Looking only at the first substantive Latin token
        # misses the named method and leaves a numeric Claim scoped to every PDF
        # named by the parent question. Resolve every identifier-shaped token,
        # while retaining the same exact metadata match and small-scope guard.
        named_terms = list(
            dict.fromkeys(
                match.group(0)
                for match in _LATIN_OR_NUMBER.finditer(query)
                if len(match.group(0)) >= 4
                and match.group(0).casefold() not in _TITLE_GENERIC_TERMS
                and _looks_like_named_identifier(match.group(0))
            )
        )[:16]
        resolved: set[str] = set()
        for term in named_terms:
            resolved.update(self._distinctive_title_scope(term, paper_sha256s))
        if resolved:
            return resolved if len(resolved) <= 8 else set()
        return set(self._distinctive_title_scope(query, paper_sha256s))

    def paper_metadata(self, paper_sha256s: set[str]) -> list[dict[str, object]]:
        """Return non-secret corpus metadata needed to resolve configured local files."""

        values = sorted({str(value) for value in paper_sha256s if str(value)})[:100]
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sha256,file_name,title,page_count FROM papers "
                f"WHERE namespace=? AND active=1 AND sha256 IN ({placeholders}) "
                "ORDER BY title,sha256",
                (self.namespace, *values),
            ).fetchall()
        return [dict(row) for row in rows]

    def route_papers(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 20,
        embedder: TextEmbedder | None = None,
        paper_sha256s: set[str] | None = None,
    ) -> PaperRoutingResult:
        """Select papers before context retrieval using query-level rank fusion.

        Each query is executed independently.  A paper contributes at most once per
        query, so one paper with many similar chunks cannot gain routing score merely
        by repeating itself in the chunk ranking.
        """

        normalized = _normalize_queries(queries)
        if not 1 <= top_k <= 50:
            raise ValueError("paper routing top_k must be between 1 and 50")
        candidate_limit = min(100, max(40, top_k * 8))
        combined: dict[str, dict[str, object]] = {}
        query_hit_counts: list[int] = []
        for query_index, query in enumerate(normalized, 1):
            query_weight = 2.0 if query_index == 1 else 1.0
            hits = self.search_chunks(
                query,
                top_k=candidate_limit,
                embedder=embedder,
                paper_sha256s=paper_sha256s,
            )
            query_hit_counts.append(len(hits))
            seen_papers: set[str] = set()
            paper_rank = 0
            for hit in hits:
                if hit.paper_sha256 in seen_papers:
                    continue
                seen_papers.add(hit.paper_sha256)
                paper_rank += 1
                entry = combined.setdefault(
                    hit.paper_sha256,
                    {
                        "file_name": hit.file_name,
                        "title": hit.title,
                        "score": 0.0,
                        "query_indexes": [],
                        "stable_ids": [],
                    },
                )
                entry["score"] = float(entry["score"]) + query_weight / (
                    _RRF_K + paper_rank
                )
                entry["query_indexes"].append(query_index)  # type: ignore[union-attr]
                entry["stable_ids"].append(hit.stable_id)  # type: ignore[union-attr]
        ranked = sorted(
            combined,
            key=lambda value: (-float(combined[value]["score"]), value),
        )[:top_k]
        return PaperRoutingResult(
            queries=normalized,
            hits=tuple(
                PaperRouteHit(
                    paper_sha256=sha256,
                    file_name=str(combined[sha256]["file_name"]),
                    title=str(combined[sha256]["title"]),
                    score=float(combined[sha256]["score"]),
                    matched_query_indexes=tuple(
                        dict.fromkeys(combined[sha256]["query_indexes"])
                    ),
                    best_stable_ids=tuple(
                        dict.fromkeys(combined[sha256]["stable_ids"])
                    ),
                )
                for sha256 in ranked
            ),
            query_hit_counts=tuple(query_hit_counts),
        )

    def refresh_lexical_index(self, *, apply: bool = False) -> dict[str, object]:
        """Explicit namespace-local refresh of derived FTS text; dry-run by default.

        Original text, stable IDs, embeddings and memory status are preserved.
        Existing indexes are never rewritten just by opening the store. Run this
        after a tokenizer change; syncing unchanged papers alone does not reindex.
        """
        counts = {'chunks': 0, 'memories': 0}
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
            for table, expression, condition in (
                ('chunks', "section || ' ' || text", ''),
                ('memories', 'content', " AND status = 'active'"),
            ):
                rows = connection.execute(
                    f'SELECT row_id, {expression} AS original, search_text FROM {table} '
                    f'WHERE namespace = ?{condition}', (self.namespace,)
                ).fetchall()
                for row in rows:
                    search_text = _search_text(row['original'])
                    if search_text == row['search_text']:
                        continue
                    counts[table] += 1
                    if apply:
                        connection.execute(f'UPDATE {table} SET search_text = ? WHERE row_id = ?',
                                           (search_text, row['row_id']))
            connection.commit()
        return {'namespace': self.namespace, 'applied': apply, 'changed_rows': counts}

    def search_memories(self, query: str, *, top_k: int = 8) -> list[MemoryHit]:
        """Search only active, evidence-linked memories inside this namespace."""

        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        match = _fts_query(query)
        if not match:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*, bm25(memory_fts) AS fts_rank
                FROM memory_fts
                JOIN memories m ON m.row_id = memory_fts.rowid
                WHERE memory_fts MATCH ? AND m.namespace = ? AND m.status = 'active'
                  AND EXISTS (
                      SELECT 1 FROM memory_evidence me
                      JOIN chunks c ON c.row_id = me.chunk_row_id
                      JOIN papers p ON p.row_id = c.paper_row_id
                      WHERE me.memory_row_id = m.row_id AND p.active = 1
                  )
                ORDER BY fts_rank, m.memory_id
                LIMIT ?
                """,
                (match, self.namespace, top_k),
            ).fetchall()
            results: list[MemoryHit] = []
            for rank, row in enumerate(rows, 1):
                evidence_rows = connection.execute(
                    """
                    SELECT c.stable_id, p.sha256, p.file_name, p.title,
                           c.page_start, c.page_end
                    FROM memory_evidence me
                    JOIN chunks c ON c.row_id = me.chunk_row_id
                    JOIN papers p ON p.row_id = c.paper_row_id
                    WHERE me.memory_row_id = ? AND p.active = 1
                    ORDER BY p.sha256, c.page_start, c.ordinal
                    """,
                    (row["row_id"],),
                ).fetchall()
                evidence = tuple(
                    MemoryEvidenceRef(
                        stable_id=item["stable_id"],
                        paper_sha256=item["sha256"],
                        file_name=item["file_name"],
                        title=item["title"],
                        page_start=int(item["page_start"]),
                        page_end=int(item["page_end"]),
                    )
                    for item in evidence_rows
                )
                if not evidence:
                    continue
                results.append(
                    MemoryHit(
                        memory_id=row["memory_id"],
                        kind=row["kind"],
                        content=row["content"],
                        confidence=row["confidence"],
                        status=row["status"],
                        source_run_id=row["source_run_id"],
                        score=1.0 / (_RRF_K + rank),
                        evidence=evidence,
                    )
                )
        return results

    def get_chunks(self, stable_ids: Sequence[str]) -> list[StoredChunkHit]:
        """Read selected active source chunks by stable ID, preserving caller order."""

        unique = list(dict.fromkeys(stable_ids))
        rows = self._chunk_rows_by_stable_ids(unique)
        by_id = {row["stable_id"]: row for row in rows}
        return [
            _stored_chunk_hit(by_id[stable_id])
            for stable_id in unique
            if stable_id in by_id
        ]

    def paper_outline(self, paper_sha256: str) -> list[dict]:
        """Expose source headings and chunk handles, without treating them as evidence."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT c.section, c.stable_id, length(c.text) AS chars
                   FROM chunks c JOIN papers p ON p.row_id = c.paper_row_id
                   WHERE c.namespace = ? AND p.active = 1 AND p.sha256 = ?
                   ORDER BY c.page_start, c.ordinal, c.row_id""",
                [self.namespace, paper_sha256],
            ).fetchall()
        sections: dict[str, dict] = {}
        for row in rows:
            section = sections.setdefault(row['section'], {
                'section_id': f'section_{len(sections)}', 'title': row['section'],
                'chunk_ids': [], 'chars': 0})
            section['chunk_ids'].append(row['stable_id'])
            section['chars'] += row['chars']
        return list(sections.values())

    def retrieve_with_memory(
        self,
        query: str,
        *,
        top_k: int = 12,
        embedder: TextEmbedder | None = None,
        paper_sha256s: set[str] | None = None,
        max_chunks_per_paper: int | None = None,
    ) -> tuple[list[StoredChunkHit], list[str]]:
        """Fuse direct corpus retrieval with memory-linked source evidence.

        Returned objects are always source chunks.  Memory text never becomes a
        citable source on its own.
        """

        direct = self.search_chunks(
            query,
            top_k=max(top_k, 8),
            embedder=embedder,
            paper_sha256s=paper_sha256s,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        memories = self.search_memories(query, top_k=max(4, min(12, top_k)))
        combined: dict[str, dict[str, object]] = {}
        direct_by_id = {item.stable_id: item for item in direct}
        for rank, item in enumerate(direct, 1):
            combined[item.stable_id] = {
                "score": 1.0 / (_RRF_K + rank),
                "channels": list(item.channels),
            }
        memory_ids: list[str] = []
        memory_stable_ids: list[str] = []
        for memory_rank, memory in enumerate(memories, 1):
            accepted = False
            for evidence_rank, evidence in enumerate(memory.evidence, 1):
                if paper_sha256s is not None and evidence.paper_sha256 not in paper_sha256s:
                    continue
                accepted = True
                entry = combined.setdefault(
                    evidence.stable_id,
                    {"score": 0.0, "channels": []},
                )
                entry["score"] = float(entry["score"]) + 1.0 / (
                    _RRF_K + memory_rank + evidence_rank - 1
                )
                if "memory_evidence" not in entry["channels"]:
                    entry["channels"].append("memory_evidence")  # type: ignore[union-attr]
                memory_stable_ids.append(evidence.stable_id)
            if accepted:
                memory_ids.append(memory.memory_id)
        missing_ids = [item for item in dict.fromkeys(memory_stable_ids) if item not in direct_by_id]
        for row in self._chunk_rows_by_stable_ids(missing_ids):
            direct_by_id[row["stable_id"]] = _stored_chunk_hit(row)
        ordered = sorted(
            combined,
            key=lambda value: (-float(combined[value]["score"]), value),
        )
        hits: list[StoredChunkHit] = []
        per_paper: dict[str, int] = {}
        for stable_id in ordered:
            base = direct_by_id.get(stable_id)
            if base is None:
                continue
            if (
                max_chunks_per_paper is not None
                and per_paper.get(base.paper_sha256, 0) >= max_chunks_per_paper
            ):
                continue
            entry = combined[stable_id]
            hits.append(
                StoredChunkHit(
                    **{
                        **base.__dict__,
                        "score": float(entry["score"]),
                        "channels": tuple(dict.fromkeys(entry["channels"])),
                    }
                )
            )
            per_paper[base.paper_sha256] = per_paper.get(base.paper_sha256, 0) + 1
            if len(hits) >= top_k:
                break
        return hits, list(dict.fromkeys(memory_ids))

    def retrieve_queries_with_memory(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 12,
        embedder: TextEmbedder | None = None,
        paper_sha256s: set[str] | None = None,
        max_chunks_per_paper: int | None = None,
        reranker: TextReranker | None = None,
    ) -> MultiQueryRetrievalResult:
        """Fuse independent queries without letting memory displace direct evidence.

        Direct corpus hits occupy the requested ``top_k``. Evidence referenced by
        past, quality-gated memories is appended as a small, bounded hint set. This
        keeps memory useful for rediscovery without allowing an old answer to
        overwrite the current query's direct source ranking.
        """

        normalized = _normalize_queries(queries)
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        per_query_k = min(100, max(20, top_k * 3))
        combined: dict[str, dict[str, object]] = {}
        paper_combined: dict[str, dict[str, object]] = {}
        base_by_id: dict[str, StoredChunkHit] = {}
        memory_ids: list[str] = []
        memory_evidence: dict[str, dict[str, object]] = {}
        query_hit_counts: list[int] = []
        reranked_query_count = 0
        rerank_skipped_query_count = 0
        anchor_stable_ids: list[str] = []
        for query_index, query in enumerate(normalized, 1):
            query_weight = 2.0 if query_index == 1 else 1.0
            hits = self.search_chunks(
                query,
                top_k=top_k if query_index == 1 else per_query_k,
                embedder=embedder,
                paper_sha256s=paper_sha256s,
                max_chunks_per_paper=max_chunks_per_paper,
                reranker=reranker,
            )
            if any("cross_encoder" in hit.channels for hit in hits):
                reranked_query_count += 1
            elif any(
                "rerank_skipped_top1_agreement" in hit.channels for hit in hits
            ):
                rerank_skipped_query_count += 1
            memories = self.search_memories(query, top_k=min(8, per_query_k))
            query_hit_counts.append(len(hits))
            if query_index == 1:
                anchor_stable_ids = [hit.stable_id for hit in hits]
            memory_ids.extend(memory.memory_id for memory in memories)
            for memory_rank, memory in enumerate(memories, 1):
                for evidence in memory.evidence:
                    if (
                        paper_sha256s is not None
                        and evidence.paper_sha256 not in paper_sha256s
                    ):
                        continue
                    entry = memory_evidence.setdefault(
                        evidence.stable_id,
                        {"score": 0.0, "channels": []},
                    )
                    entry["score"] = float(entry["score"]) + 1.0 / (
                        _RRF_K + memory_rank
                    )
                    channel = f"memory_query_{query_index}"
                    if channel not in entry["channels"]:  # type: ignore[operator]
                        entry["channels"].append(channel)  # type: ignore[union-attr]
            seen_papers: set[str] = set()
            paper_rank = 0
            for rank, hit in enumerate(hits, 1):
                if hit.paper_sha256 not in seen_papers:
                    seen_papers.add(hit.paper_sha256)
                    paper_rank += 1
                    paper_entry = paper_combined.setdefault(
                        hit.paper_sha256,
                        {
                            "file_name": hit.file_name,
                            "title": hit.title,
                            "score": 0.0,
                            "query_indexes": [],
                            "stable_ids": [],
                        },
                    )
                    paper_entry["score"] = float(paper_entry["score"]) + query_weight / (
                        _RRF_K + paper_rank
                    )
                    paper_entry["query_indexes"].append(  # type: ignore[union-attr]
                        query_index
                    )
                    paper_entry["stable_ids"].append(  # type: ignore[union-attr]
                        hit.stable_id
                    )
                base_by_id.setdefault(hit.stable_id, hit)
                entry = combined.setdefault(
                    hit.stable_id,
                    {
                        "score": 0.0,
                        "lexical": 0.0,
                        "semantic": 0.0,
                        "channels": [],
                    },
                )
                entry["score"] = float(entry["score"]) + 1.0 / (_RRF_K + rank)
                entry["lexical"] = max(
                    float(entry["lexical"]), hit.lexical_score
                )
                entry["semantic"] = max(
                    float(entry["semantic"]), hit.semantic_score
                )
                channels = entry["channels"]
                for channel in (*hit.channels, f"query_{query_index}"):
                    if channel not in channels:  # type: ignore[operator]
                        channels.append(channel)  # type: ignore[union-attr]
        ordered = sorted(
            combined,
            key=lambda value: (-float(combined[value]["score"]), value),
        )
        accepted: list[StoredChunkHit] = []
        per_paper: dict[str, int] = {}
        accepted_ids: set[str] = set()

        def accept_direct(stable_id: str, *, anchor: bool) -> bool:
            base = base_by_id[stable_id]
            if (
                max_chunks_per_paper is not None
                and per_paper.get(base.paper_sha256, 0) >= max_chunks_per_paper
            ):
                return False
            entry = combined[stable_id]
            channels = list(entry["channels"])
            if anchor and "anchor_query" not in channels:
                channels.append("anchor_query")
            accepted.append(
                replace(
                    base,
                    score=float(entry["score"]),
                    lexical_score=float(entry["lexical"]),
                    semantic_score=float(entry["semantic"]),
                    channels=tuple(channels),  # type: ignore[arg-type]
                )
            )
            accepted_ids.add(stable_id)
            per_paper[base.paper_sha256] = per_paper.get(base.paper_sha256, 0) + 1
            return True

        for stable_id in anchor_stable_ids:
            accept_direct(stable_id, anchor=True)
        expansion_budget = (
            min(6, max(0, 50 - len(accepted)), max(1, top_k // 3))
            if len(normalized) > 1
            else 0
        )
        expansion_count = 0
        if expansion_budget:
            for stable_id in ordered:
                if stable_id in accepted_ids:
                    continue
                if accept_direct(stable_id, anchor=False):
                    expansion_count += 1
                if expansion_count >= expansion_budget:
                    break
        memory_budget = min(
            4,
            max(0, 50 - len(accepted)),
            max(1, top_k // 4),
        )
        memory_ranked_ids = [
            stable_id
            for stable_id in sorted(
                memory_evidence,
                key=lambda value: (
                    -float(memory_evidence[value]["score"]),
                    value,
                ),
            )
            if stable_id not in accepted_ids
        ]
        memory_hits = self.get_chunks(memory_ranked_ids)
        memory_representatives = 0
        for hit in memory_hits:
            if (
                max_chunks_per_paper is not None
                and per_paper.get(hit.paper_sha256, 0) >= max_chunks_per_paper
            ):
                continue
            entry = memory_evidence[hit.stable_id]
            accepted.append(
                replace(
                    hit,
                    score=float(entry["score"]),
                    channels=(
                        "memory_hint",
                        *tuple(entry["channels"]),  # type: ignore[arg-type]
                    ),
                )
            )
            per_paper[hit.paper_sha256] = per_paper.get(hit.paper_sha256, 0) + 1
            memory_representatives += 1
            if memory_representatives >= memory_budget:
                break
        return MultiQueryRetrievalResult(
            queries=normalized,
            hits=tuple(accepted),
            paper_hits=tuple(
                PaperRouteHit(
                    paper_sha256=sha256,
                    file_name=str(paper_combined[sha256]["file_name"]),
                    title=str(paper_combined[sha256]["title"]),
                    score=float(paper_combined[sha256]["score"]),
                    matched_query_indexes=tuple(
                        dict.fromkeys(paper_combined[sha256]["query_indexes"])
                    ),
                    best_stable_ids=tuple(
                        dict.fromkeys(paper_combined[sha256]["stable_ids"])
                    ),
                )
                for sha256 in sorted(
                    paper_combined,
                    key=lambda value: (
                        -float(paper_combined[value]["score"]),
                        value,
                    ),
                )[:50]
            ),
            recalled_memory_ids=tuple(dict.fromkeys(memory_ids)),
            query_hit_counts=tuple(query_hit_counts),
            memory_representative_count=memory_representatives,
            multi_query_expansion_count=expansion_count,
            reranked_query_count=reranked_query_count,
            rerank_skipped_query_count=rerank_skipped_query_count,
        )

    def remember_analysis(
        self,
        analysis: PaperAnalysis,
        documents: Sequence[PaperDocument],
        chunks: Sequence[PaperChunk],
        *,
        run_id: str,
    ) -> MemoryWriteResult:
        """Write structured claims only after recomputing the evidence quality gate."""

        recomputed = audit_paper_analysis(analysis, list(chunks))
        if not recomputed.valid:
            return MemoryWriteResult(
                namespace=self.namespace,
                gate_passed=False,
                inserted=0,
                duplicates=0,
                skipped=0,
                reason="analysis evidence/content quality gate did not pass",
            )
        candidates = _analysis_candidates(analysis)
        return self._write_candidates(candidates, documents, chunks, run_id=run_id)

    def remember_answer(
        self,
        answer: PaperAnswer,
        documents: Sequence[PaperDocument],
        chunks: Sequence[PaperChunk],
        *,
        run_id: str,
    ) -> MemoryWriteResult:
        """Persist a supported answer and its key points as retrieval hints."""

        recomputed = audit_paper_answer(answer, list(chunks))
        if not recomputed.valid:
            return MemoryWriteResult(
                namespace=self.namespace,
                gate_passed=False,
                inserted=0,
                duplicates=0,
                skipped=0,
                reason="answer evidence quality gate did not pass",
            )
        candidates = [
            _MemoryCandidate(
                kind="qa_answer",
                content=answer.answer,
                evidence_ids=tuple(answer.evidence_ids),
                confidence="medium",
            ),
            *(
                _MemoryCandidate(
                    kind="qa_key_point",
                    content=point.text,
                    evidence_ids=tuple(point.evidence_ids),
                    confidence=point.confidence,
                )
                for point in answer.key_points
            ),
        ]
        return self._write_candidates(candidates, documents, chunks, run_id=run_id)

    def relate_memories(
        self,
        source_memory_id: str,
        target_memory_id: str,
        relation: MemoryRelation,
    ) -> None:
        """Record an explicit support/conflict/supersession decision without guessing it."""

        if relation not in {"supports", "contradicts", "supersedes"}:
            raise ValueError(f"unsupported memory relation: {relation}")
        if source_memory_id == target_memory_id:
            raise ValueError("a memory cannot relate to itself")
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT memory_id FROM memories WHERE namespace = ? "
                "AND memory_id IN (?, ?)",
                (self.namespace, source_memory_id, target_memory_id),
            ).fetchall()
            if {row["memory_id"] for row in rows} != {
                source_memory_id,
                target_memory_id,
            }:
                raise KeyError("both memories must exist in the selected namespace")
            connection.execute(
                "INSERT OR IGNORE INTO memory_relations"
                "(namespace, source_memory_id, target_memory_id, relation, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    self.namespace,
                    source_memory_id,
                    target_memory_id,
                    relation,
                    now,
                ),
            )
            if relation == "supersedes":
                connection.execute(
                    "UPDATE memories SET status = 'superseded', updated_at = ? "
                    "WHERE namespace = ? AND memory_id = ? AND status = 'active'",
                    (now, self.namespace, target_memory_id),
                )
                connection.execute(
                    "UPDATE memories SET supersedes_id = ?, updated_at = ? "
                    "WHERE namespace = ? AND memory_id = ?",
                    (target_memory_id, now, self.namespace, source_memory_id),
                )
            self._event(
                connection,
                f"memory.{relation}",
                source_memory_id,
                {"target": target_memory_id},
            )
            connection.commit()

    def forget_paper(self, paper_sha256: str) -> ForgetResult:
        """Permanently delete indexed source text and tombstone all derived memories."""

        if not re.fullmatch(r"[a-f0-9]{64}", paper_sha256):
            raise ValueError("paper SHA-256 must contain exactly 64 lowercase hex characters")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            paper = connection.execute(
                "SELECT row_id FROM papers WHERE namespace = ? AND sha256 = ?",
                (self.namespace, paper_sha256),
            ).fetchone()
            if paper is None:
                return ForgetResult(self.namespace, paper_sha256, 0, 0, 0)
            paper_row_id = int(paper["row_id"])
            chunks_removed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE paper_row_id = ?", (paper_row_id,)
                ).fetchone()[0]
            )
            memory_rows = connection.execute(
                """
                SELECT DISTINCT m.row_id
                FROM memories m
                JOIN memory_evidence me ON me.memory_row_id = m.row_id
                JOIN chunks c ON c.row_id = me.chunk_row_id
                WHERE m.namespace = ? AND c.paper_row_id = ?
                """,
                (self.namespace, paper_row_id),
            ).fetchall()
            memory_ids = [int(row["row_id"]) for row in memory_rows]
            if memory_ids:
                placeholders = ",".join("?" for _ in memory_ids)
                connection.execute(
                    f"UPDATE memories SET status = 'tombstoned', content = ?, "
                    f"search_text = '', updated_at = ? WHERE row_id IN ({placeholders})",
                    [_MEMORY_REDACTION, _now(), *memory_ids],
                )
            connection.execute("DELETE FROM papers WHERE row_id = ?", (paper_row_id,))
            self._event(
                connection,
                "paper.forgotten",
                paper_sha256,
                {
                    "chunks_removed": chunks_removed,
                    "memories_tombstoned": len(memory_ids),
                },
            )
            connection.commit()
        return ForgetResult(
            namespace=self.namespace,
            paper_sha256=paper_sha256,
            papers_removed=1,
            chunks_removed=chunks_removed,
            memories_tombstoned=len(memory_ids),
        )

    def stats(self) -> dict[str, object]:
        """Return namespace-isolated, non-sensitive corpus and memory counters."""

        with self._connect() as connection:
            papers = int(
                connection.execute(
                    "SELECT COUNT(*) FROM papers WHERE namespace = ? AND active = 1",
                    (self.namespace,),
                ).fetchone()[0]
            )
            chunks = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM chunks c JOIN papers p ON p.row_id = c.paper_row_id
                    WHERE c.namespace = ? AND p.active = 1
                    """,
                    (self.namespace,),
                ).fetchone()[0]
            )
            memories = {
                row["status"]: int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM memories "
                    "WHERE namespace = ? GROUP BY status",
                    (self.namespace,),
                ).fetchall()
            }
            embedding_rows = connection.execute(
                """
                SELECT e.model, COUNT(*) AS count
                FROM chunk_embeddings e
                JOIN chunks c ON c.row_id = e.chunk_row_id
                JOIN papers p ON p.row_id = c.paper_row_id
                WHERE c.namespace = ? AND p.active = 1
                GROUP BY e.model ORDER BY e.model
                """,
                (self.namespace,),
            ).fetchall()
        return {
            "schema_version": _SCHEMA_VERSION,
            "namespace": self.namespace,
            "papers": papers,
            "chunks": chunks,
            "memories": memories,
            "embeddings": {row["model"]: int(row["count"]) for row in embedding_rows},
        }

    def corpus_snapshot(
        self, paper_sha256s: set[str] | None = None
    ) -> dict[str, object]:
        """Return a text-free inventory and fingerprint for a declared corpus scope."""

        digest = hashlib.sha256()
        selected = sorted(paper_sha256s) if paper_sha256s is not None else None
        if selected is not None:
            scope_sql = (
                f" AND p.sha256 IN ({','.join('?' for _ in selected)})"
                if selected
                else " AND 0"
            )
            scope_params: list[object] = [*selected]
        else:
            scope_sql = ""
            scope_params = []
        with self._connect() as connection:
            paper_rows = connection.execute(
                f"""
                SELECT p.sha256, p.file_name, p.title, p.page_count,
                       p.processed_pages, COUNT(c.row_id) AS chunk_count
                FROM papers p
                LEFT JOIN chunks c ON c.paper_row_id = p.row_id
                WHERE p.namespace = ? AND p.active = 1{scope_sql}
                GROUP BY p.row_id
                ORDER BY p.sha256
                """,
                [self.namespace, *scope_params],
            ).fetchall()
            stable_rows = connection.execute(
                f"""
                SELECT c.stable_id
                FROM chunks c JOIN papers p ON p.row_id = c.paper_row_id
                WHERE c.namespace = ? AND p.active = 1{scope_sql}
                ORDER BY c.stable_id
                """,
                [self.namespace, *scope_params],
            )
            for row in paper_rows:
                digest.update(f"paper:{row['sha256']}\n".encode())
            chunk_count = 0
            for row in stable_rows:
                digest.update(f"chunk:{row['stable_id']}\n".encode())
                chunk_count += 1
        return {
            "schema_version": _SCHEMA_VERSION,
            "namespace": self.namespace,
            "fingerprint": digest.hexdigest(),
            "paper_count": len(paper_rows),
            "chunk_count": chunk_count,
            "papers": [
                {
                    "sha256": row["sha256"],
                    "file_name": row["file_name"],
                    "title": row["title"],
                    "page_count": int(row["page_count"]),
                    "processed_pages": int(row["processed_pages"]),
                    "chunk_count": int(row["chunk_count"]),
                }
                for row in paper_rows
            ],
        }

    def materialize_hits(
        self, hits: Sequence[StoredChunkHit]
    ) -> tuple[list[PaperDocument], list[PaperChunk]]:
        """Convert persistent hits into run-local P#/evidence IDs for audited answers."""

        paper_hashes = list(dict.fromkeys(hit.paper_sha256 for hit in hits))
        paper_ids = {sha256: f"P{index}" for index, sha256 in enumerate(paper_hashes, 1)}
        with self._connect() as connection:
            metadata = {
                row["sha256"]: row
                for row in connection.execute(
                    f"SELECT * FROM papers WHERE namespace = ? AND active = 1 AND sha256 IN "
                    f"({','.join('?' for _ in paper_hashes)})",
                    [self.namespace, *paper_hashes],
                ).fetchall()
            } if paper_hashes else {}
        documents: list[PaperDocument] = []
        for sha256 in paper_hashes:
            row = metadata.get(sha256)
            if row is None:
                continue
            documents.append(
                PaperDocument(
                    id=paper_ids[sha256],
                    file_name=row["file_name"],
                    sha256=sha256,
                    title=row["title"],
                    authors=json.loads(row["authors_json"]),
                    page_count=int(row["page_count"]),
                    processed_pages=0,
                    pages=[],
                    warnings=["本次问答从持久化论文库读取检索片段，未重新解析 PDF"],
                )
            )
        chunks: list[PaperChunk] = []
        for hit in hits:
            paper_id = paper_ids.get(hit.paper_sha256)
            if paper_id is None:
                continue
            range_label = (
                f"p{hit.page_start}"
                if hit.page_start == hit.page_end
                else f"p{hit.page_start}-p{hit.page_end}"
            )
            chunks.append(
                PaperChunk(
                    id=f"{paper_id}:{range_label}:c{hit.ordinal}",
                    paper_id=paper_id,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    section=hit.section,
                    text=hit.text,
                    token_count_estimate=hit.token_count_estimate,
                )
            )
        return documents, chunks

    def stable_ids_for_run(
        self,
        documents: Sequence[PaperDocument],
        chunks: Sequence[PaperChunk],
    ) -> dict[str, str]:
        document_by_id = {document.id: document for document in documents}
        return {
            stable_chunk_id(document_by_id[chunk.paper_id].sha256, chunk): chunk.id
            for chunk in chunks
            if chunk.paper_id in document_by_id
        }

    def _write_candidates(
        self,
        candidates: Iterable[_MemoryCandidate],
        documents: Sequence[PaperDocument],
        chunks: Sequence[PaperChunk],
        *,
        run_id: str,
    ) -> MemoryWriteResult:
        document_by_id = {document.id: document for document in documents}
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        stable_by_run_id = {
            chunk.id: stable_chunk_id(document_by_id[chunk.paper_id].sha256, chunk)
            for chunk in chunks
            if chunk.paper_id in document_by_id
        }
        inserted = 0
        duplicates = 0
        skipped = 0
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for candidate in candidates:
                content = _clean_memory_text(candidate.content)
                evidence_ids = tuple(dict.fromkeys(candidate.evidence_ids))
                if len(content) < 2 or not evidence_ids:
                    skipped += 1
                    continue
                if any(item not in chunk_by_id or item not in stable_by_run_id for item in evidence_ids):
                    skipped += 1
                    continue
                stable_ids = [stable_by_run_id[item] for item in evidence_ids]
                placeholders = ",".join("?" for _ in stable_ids)
                evidence_rows = connection.execute(
                    f"""
                    SELECT c.row_id, c.stable_id
                    FROM chunks c JOIN papers p ON p.row_id = c.paper_row_id
                    WHERE c.namespace = ? AND p.active = 1
                      AND c.stable_id IN ({placeholders})
                    """,
                    [self.namespace, *stable_ids],
                ).fetchall()
                rows_by_id = {row["stable_id"]: int(row["row_id"]) for row in evidence_rows}
                if set(rows_by_id) != set(stable_ids):
                    skipped += 1
                    continue
                memory_id = _memory_id(candidate.kind, content, stable_ids)
                existing = connection.execute(
                    "SELECT row_id, status FROM memories WHERE namespace = ? AND memory_id = ?",
                    (self.namespace, memory_id),
                ).fetchone()
                if existing is not None:
                    if existing["status"] == "stale":
                        connection.execute(
                            "UPDATE memories SET status = 'active', content = ?, search_text = ?, "
                            "source_run_id = ?, updated_at = ? WHERE row_id = ?",
                            (
                                content,
                                _search_text(content),
                                run_id,
                                now,
                                int(existing["row_id"]),
                            ),
                        )
                        for stable_id in stable_ids:
                            connection.execute(
                                "INSERT OR IGNORE INTO memory_evidence"
                                "(memory_row_id, chunk_row_id) VALUES (?, ?)",
                                (int(existing["row_id"]), rows_by_id[stable_id]),
                            )
                        inserted += 1
                    else:
                        duplicates += 1
                    continue
                normalized_sha = _sha256_text(_normalize_memory_text(content))
                cursor = connection.execute(
                    """
                    INSERT INTO memories(
                        namespace, memory_id, kind, content, search_text,
                        normalized_sha256, confidence, status, source_run_id,
                        supersedes_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL, ?, ?)
                    """,
                    (
                        self.namespace,
                        memory_id,
                        candidate.kind,
                        content,
                        _search_text(content),
                        normalized_sha,
                        candidate.confidence,
                        run_id,
                        now,
                        now,
                    ),
                )
                memory_row_id = int(cursor.lastrowid)
                connection.executemany(
                    "INSERT INTO memory_evidence(memory_row_id, chunk_row_id) VALUES (?, ?)",
                    [(memory_row_id, rows_by_id[item]) for item in stable_ids],
                )
                inserted += 1
            self._event(
                connection,
                "memory.batch_written",
                run_id,
                {"inserted": inserted, "duplicates": duplicates, "skipped": skipped},
            )
            connection.commit()
        return MemoryWriteResult(
            namespace=self.namespace,
            gate_passed=True,
            inserted=inserted,
            duplicates=duplicates,
            skipped=skipped,
        )

    def _lexical_chunk_rows(
        self,
        query: str,
        limit: int,
        paper_sha256s: set[str] | None,
    ) -> list[sqlite3.Row]:
        lexical_query = content_query(query) if self.lexical_query_policy == 'content_v1' else query
        match = _fts_query(lexical_query)
        if not match:
            return []
        where = "chunk_fts MATCH ? AND c.namespace = ? AND p.active = 1"
        params: list[object] = [match, self.namespace]
        if paper_sha256s is not None:
            if not paper_sha256s:
                return []
            placeholders = ",".join("?" for _ in paper_sha256s)
            where += f" AND p.sha256 IN ({placeholders})"
            params.extend(sorted(paper_sha256s))
        params.append(limit)
        with self._connect() as connection:
            return connection.execute(
                f"""
                SELECT c.stable_id, bm25(chunk_fts) AS raw_score
                FROM chunk_fts
                JOIN chunks c ON c.row_id = chunk_fts.rowid
                JOIN papers p ON p.row_id = c.paper_row_id
                WHERE {where}
                ORDER BY raw_score, c.stable_id
                LIMIT ?
                """,
                params,
            ).fetchall()

    def _dense_chunk_rows(
        self,
        query: str,
        embedder: TextEmbedder,
        limit: int,
        paper_sha256s: set[str] | None,
    ) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        query_vector = _normalize_vector(embedder.embed_query(query))
        model_identity = _embedding_storage_identity(embedder)
        where = "c.namespace = ? AND p.active = 1 AND e.model = ?"
        params: list[object] = [self.namespace, model_identity]
        if paper_sha256s is not None:
            if not paper_sha256s:
                return []
            placeholders = ",".join("?" for _ in paper_sha256s)
            where += f" AND p.sha256 IN ({placeholders})"
            params.extend(sorted(paper_sha256s))
        heap: list[tuple[float, str]] = []
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                SELECT c.stable_id, e.dimension, e.vector
                FROM chunk_embeddings e
                JOIN chunks c ON c.row_id = e.chunk_row_id
                JOIN papers p ON p.row_id = c.paper_row_id
                WHERE {where}
                """,
                params,
            )
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                for row in rows:
                    vector = array("f")
                    vector.frombytes(row["vector"])
                    if len(vector) != int(row["dimension"]) or len(vector) != len(query_vector):
                        continue
                    score = sum(
                        left * float(right)
                        for left, right in zip(query_vector, vector, strict=True)
                    )
                    item = (score, row["stable_id"])
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
        return [(stable_id, score) for score, stable_id in sorted(heap, reverse=True)]

    def _chunk_rows_by_stable_ids(self, stable_ids: Sequence[str]) -> list[sqlite3.Row]:
        unique = list(dict.fromkeys(stable_ids))
        if not unique:
            return []
        placeholders = ",".join("?" for _ in unique)
        with self._connect() as connection:
            return connection.execute(
                f"""
                SELECT c.*, p.sha256, p.file_name, p.title
                FROM chunks c JOIN papers p ON p.row_id = c.paper_row_id
                WHERE c.namespace = ? AND p.active = 1
                  AND c.stable_id IN ({placeholders})
                """,
                [self.namespace, *unique],
            ).fetchall()

    def _mark_memories_stale_for_paper(
        self, connection: sqlite3.Connection, paper_row_id: int
    ) -> None:
        connection.execute(
            """
            UPDATE memories SET status = 'stale', updated_at = ?
            WHERE row_id IN (
                SELECT DISTINCT me.memory_row_id
                FROM memory_evidence me
                JOIN chunks c ON c.row_id = me.chunk_row_id
                WHERE c.paper_row_id = ?
            ) AND status = 'active'
            """,
            (_now(), paper_row_id),
        )

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        object_id: str,
        details: dict[str, object],
    ) -> None:
        connection.execute(
            "INSERT INTO memory_events"
            "(namespace, event_type, object_id, details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                self.namespace,
                event_type,
                object_id[:200],
                json.dumps(details, ensure_ascii=False, sort_keys=True),
                _now(),
            ),
        )


def stable_chunk_id(paper_sha256: str, chunk: PaperChunk) -> str:
    """Return a run-order-independent ID for a source chunk."""

    if not re.fullmatch(r"[a-f0-9]{64}", paper_sha256):
        raise ValueError("paper SHA-256 must contain exactly 64 lowercase hex characters")
    text_hash = _sha256_text(chunk.text)
    page_label = (
        f"p{chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"p{chunk.page_start}-{chunk.page_end}"
    )
    return (
        f"C-{paper_sha256[:16]}-{page_label}-c{_chunk_ordinal(chunk)}-"
        f"{text_hash[:12]}"
    )


def _normalize_queries(queries: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for query in queries:
        clean = _WHITESPACE.sub(" ", str(query)).strip()[:500]
        if clean:
            normalized.append(clean)
    unique = tuple(dict.fromkeys(normalized))
    if not unique:
        raise ValueError("at least one non-empty retrieval query is required")
    if len(unique) > 16:
        raise ValueError("at most 16 retrieval queries are allowed")
    return unique


def _analysis_candidates(analysis: PaperAnalysis) -> list[_MemoryCandidate]:
    candidates: list[_MemoryCandidate] = []
    for profile in analysis.profiles:
        candidates.append(
            _MemoryCandidate(
                "paper_summary",
                profile.one_sentence_summary,
                tuple(profile.summary_evidence_ids),
                "medium",
            )
        )
        candidates.append(_point_candidate("research_problem", profile.research_problem))
        candidates.extend(_point_candidate("contribution", item) for item in profile.contributions)
        for component in profile.model_components:
            operations = "；操作：" + "、".join(component.operations) if component.operations else ""
            candidates.append(
                _MemoryCandidate(
                    "model_component",
                    f"{component.name}：{component.role}{operations}",
                    tuple(component.evidence_ids),
                    "medium",
                )
            )
        candidates.extend(
            _point_candidate("training_objective", item)
            for item in profile.training_objectives
        )
        for experiment in profile.experiments:
            candidates.append(
                _MemoryCandidate(
                    "experiment",
                    (
                        f"问题：{experiment.question}；设置：{experiment.setup}；"
                        f"数据：{experiment.dataset}；结果：{experiment.result}；"
                        f"解释：{experiment.interpretation}"
                    ),
                    tuple(experiment.evidence_ids),
                    "medium",
                )
            )
        candidates.extend(_point_candidate("limitation", item) for item in profile.limitations)
        for card in profile.learning_cards:
            candidates.append(
                _MemoryCandidate(
                    "learning_card",
                    f"{card.question}：{card.answer}",
                    tuple(card.evidence_ids),
                    "medium",
                )
            )
    candidates.extend(
        _point_candidate("cross_paper_comparison", item)
        for item in analysis.cross_paper_comparison
    )
    for step in analysis.reproduction.steps:
        candidates.append(
            _MemoryCandidate(
                "reproduction_step",
                f"{step.title}：{step.action}；验证：{step.verification}",
                tuple(step.evidence_ids),
                "medium",
            )
        )
    return candidates


def _point_candidate(kind: str, point: EvidencePoint) -> _MemoryCandidate:
    return _MemoryCandidate(kind, point.text, tuple(point.evidence_ids), point.confidence)


def _stored_chunk_hit(
    row: sqlite3.Row,
    *,
    score: float = 0.0,
    lexical_score: float = 0.0,
    semantic_score: float = 0.0,
    channels: tuple[str, ...] = (),
) -> StoredChunkHit:
    return StoredChunkHit(
        stable_id=row["stable_id"],
        paper_sha256=row["sha256"],
        file_name=row["file_name"],
        title=row["title"],
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        ordinal=int(row["ordinal"]),
        section=row["section"],
        text=row["text"],
        token_count_estimate=int(row["token_count_estimate"]),
        score=score,
        lexical_score=lexical_score,
        semantic_score=semantic_score,
        channels=channels,
    )


def _chunk_ordinal(chunk: PaperChunk) -> int:
    match = _CHUNK_INDEX.search(chunk.id)
    if match is None:
        raise ValueError(f"chunk ID has no ordinal: {chunk.id}")
    return int(match.group("ordinal"))


def _chunk_fingerprint(paper_sha256: str, chunks: Sequence[PaperChunk]) -> str:
    digest = hashlib.sha256()
    digest.update(paper_sha256.encode("ascii"))
    for chunk in chunks:
        digest.update(stable_chunk_id(paper_sha256, chunk).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _memory_id(kind: str, content: str, stable_ids: Sequence[str]) -> str:
    payload = json.dumps(
        [kind, _normalize_memory_text(content), sorted(set(stable_ids))],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"M-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _clean_memory_text(value: str) -> str:
    clean = _RUN_CITATION.sub("", str(value))
    clean = clean.replace("\x00", " ")
    clean = _WHITESPACE.sub(" ", clean).strip()
    return clean[:8_000]


def _normalize_memory_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _looks_like_named_identifier(value: str) -> bool:
    letters = [character for character in value if character.isalpha()]
    has_letter = bool(letters)
    has_digit = any(character.isdigit() for character in value)
    has_lower = any(character.islower() for character in letters)
    has_internal_upper = any(character.isupper() for character in value[1:])
    is_acronym = len(letters) >= 4 and all(character.isupper() for character in letters)
    return bool(
        (has_letter and has_digit)
        or (has_lower and has_internal_upper)
        or is_acronym
    )


def _search_text(value: str) -> str:
    return " ".join(_tokenize(value)[:20_000])


def _fts_query(value: str) -> str:
    tokens = list(dict.fromkeys(_tokenize(value)))[:64]
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def _tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    # A one-character number is meaningful (Table 1, layer 2, k=5). Retain it
    # in both indexed source text and queries; still discard single letters.
    tokens = [item for item in _LATIN_OR_NUMBER.findall(lowered) if len(item) > 1 or item.isdigit()]
    for sequence in _CHINESE.findall(lowered):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _normalize_vector(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise ValueError("embedding vector has zero norm")
    return [value / norm for value in vector]


def _embedding_storage_identity(embedder: TextEmbedder) -> str:
    value = getattr(embedder, "cache_identity", None)
    identity = value() if callable(value) else value
    if identity is None:
        identity = embedder.name
    normalized = str(identity).strip()
    if not normalized or len(normalized) > 512:
        raise ValueError("embedding storage identity must contain 1-512 characters")
    return normalized


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _validate_namespace(value: str) -> str:
    clean = value.strip()
    if not _NAMESPACE.fullmatch(clean):
        raise ValueError(
            "memory namespace must start with a letter or number and use at most "
            "64 letters, numbers, dots, underscores, or hyphens"
        )
    return clean


def _now() -> str:
    return datetime.now(UTC).isoformat()
