from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .models import stable_hash

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class PaperSearchAdapter(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]: ...


def load_callable(spec: str) -> Callable[..., Any]:
    """Load ``module.path:function`` without evaluating arbitrary expressions."""
    if ":" not in spec:
        raise ValueError("Callable adapter must use module.path:function syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute or any(part.startswith("_") for part in attribute.split(".")):
        raise ValueError(f"Unsafe callable adapter path: {spec!r}")
    value: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"Adapter is not callable: {spec}")
    return value


class CallablePaperSearchAdapter:
    """Bridge to PaperPilot's existing retriever without coupling the durable runtime
    to one internal class name. Configure with ``module:function`` or inject a callable.
    """

    def __init__(self, target: str | Callable[..., Any]):
        self.target = load_callable(target) if isinstance(target, str) else target

    def search(
        self,
        query: str,
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        signature = inspect.signature(self.target)
        kwargs: dict[str, Any] = {}
        if "top_k" in signature.parameters:
            kwargs["top_k"] = top_k
        elif "limit" in signature.parameters:
            kwargs["limit"] = top_k
        if "paper_sha256s" in signature.parameters:
            kwargs["paper_sha256s"] = paper_sha256s
        raw = self.target(query, **kwargs)
        if isinstance(raw, Mapping):
            for key in ("results", "items", "documents", "chunks", "evidence"):
                if key in raw and isinstance(raw[key], Sequence):
                    raw = raw[key]
                    break
            else:
                raw = [raw]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError("Search adapter must return a sequence or a mapping containing results")
        return [self._normalize(item, rank) for rank, item in enumerate(raw[:top_k], start=1)]

    def search_many(
        self,
        queries: Sequence[str],
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        merged: list[Mapping[str, Any]] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for query in queries:
            for item in self.search(query, top_k=top_k, paper_sha256s=paper_sha256s):
                key = (item.get("paper_sha256"), item.get("stable_id"), item.get("content"))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        return merged[:top_k]

    @staticmethod
    def _normalize(item: Any, rank: int) -> Mapping[str, Any]:
        if isinstance(item, str):
            return {"content": item, "rank": rank}
        if not isinstance(item, Mapping):
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()
            elif hasattr(item, "__dict__"):
                item = vars(item)
            else:
                return {"content": str(item), "rank": rank}
        value = dict(item)
        content = next(
            (value.get(key) for key in ("content", "text", "chunk", "passage", "body", "snippet") if value.get(key)),
            "",
        )
        source_uri = next(
            (value.get(key) for key in ("source_uri", "path", "file_path", "url", "paper_id") if value.get(key)),
            None,
        )
        return {
            **value,
            "content": str(content),
            "source_uri": str(source_uri) if source_uri is not None else None,
            "rank": value.get("rank", rank),
        }


class LocalCorpusSearchAdapter:
    """Dependency-free fallback used for smoke tests and small local corpora.

    It is intentionally not presented as a replacement for PaperPilot's BM25/Dense/RRF
    retriever. The production bridge should point to the existing retriever via
    ``PAPERPILOT_SEARCH_ADAPTER=module:function``.
    """

    def __init__(self, roots: Sequence[str | Path], *, max_files: int = 2_000, max_chars_per_file: int = 200_000):
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        self.max_files = max_files
        self.max_chars_per_file = max_chars_per_file

    def search(
        self,
        query: str,
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        del paper_sha256s  # Plain files have no PaperPilot document hash metadata.
        query_tokens = {token.lower() for token in _TOKEN_RE.findall(query)}
        candidates: list[tuple[float, dict[str, Any]]] = []
        seen = 0
        for root in self.roots:
            if not root.exists():
                continue
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if seen >= self.max_files:
                    break
                if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".rst", ".json", ".jsonl", ".pdf"}:
                    continue
                if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
                    continue
                seen += 1
                try:
                    pages = self._read(path)
                except Exception:
                    continue
                for page, content in pages:
                    normalized = re.sub(r"\s+", " ", content).strip()
                    if not normalized:
                        continue
                    tokens = {token.lower() for token in _TOKEN_RE.findall(normalized)}
                    overlap = len(tokens & query_tokens) / max(1, len(query_tokens))
                    phrase = 0.35 if query.lower() in normalized.lower() else 0.0
                    score = overlap + phrase
                    if score <= 0:
                        continue
                    candidates.append(
                        (
                            score,
                            {
                                "content": normalized[:8_000],
                                "source_uri": (
                                    f"local-corpus://{stable_hash('corpus-file', str(path))[:24]}"
                                ),
                                "file_name": path.name,
                                "page": page,
                                "score": score,
                            },
                        )
                    )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in candidates[:top_k]]

    def search_many(
        self,
        queries: Sequence[str],
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        merged: list[Mapping[str, Any]] = []
        seen: set[tuple[Any, Any]] = set()
        for query in queries:
            for item in self.search(query, top_k=top_k, paper_sha256s=paper_sha256s):
                key = (item.get("source_uri"), item.get("page"))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def _read(self, path: Path) -> list[tuple[int | None, str]]:
        if path.suffix.lower() == ".pdf":
            if PdfReader is None:
                return []
            reader = PdfReader(str(path))
            return [(index + 1, (page.extract_text() or "")[: self.max_chars_per_file]) for index, page in enumerate(reader.pages)]
        text = path.read_text(encoding="utf-8", errors="replace")[: self.max_chars_per_file]
        if path.suffix.lower() == ".jsonl":
            return [(None, line) for line in text.splitlines() if line.strip()]
        return [(None, text)]


class PaperPilotKnowledgeSearchAdapter:
    """Adapter for PaperPilot's durable FTS5/Dense/RRF corpus and memory store."""

    def __init__(
        self,
        store: Any,
        *,
        embedder: Any = None,
        reranker: Any = None,
        max_chunks_per_paper: int = 3,
        paper_roots: Sequence[str | Path] | None = None,
        rerank_stage: str = "early",
        scoped_candidate_k: int | None = None,
        scoped_diversity: float = 0.22,
    ) -> None:
        if rerank_stage not in {"early", "final"}:
            raise ValueError("rerank_stage must be early or final")
        self.rerank_stage = rerank_stage
        if scoped_candidate_k is not None and not 8 <= scoped_candidate_k <= 100:
            raise ValueError("scoped_candidate_k must be between 8 and 100")
        if not 0 <= scoped_diversity <= 1:
            raise ValueError("scoped_diversity must be between zero and one")
        self.scoped_candidate_k = scoped_candidate_k
        self.scoped_diversity = scoped_diversity
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.max_chunks_per_paper = max_chunks_per_paper
        self.paper_roots = tuple(
            Path(value).expanduser().resolve() for value in (paper_roots or ())
        )
        self._pdf_name_index: dict[str, list[Path]] | None = None
        self._resolved_pdf_cache: dict[str, str | None] = {}

    @classmethod
    def from_database(
        cls,
        path: str | Path,
        *,
        namespace: str = "default",
        embedder: Any = None,
        reranker: Any = None,
        paper_roots: Sequence[str | Path] | None = None,
    ) -> "PaperPilotKnowledgeSearchAdapter":
        from paperpilot_retrieval.store import PaperKnowledgeStore

        return cls(
            PaperKnowledgeStore(Path(path), namespace=namespace),
            embedder=embedder,
            reranker=reranker,
            paper_roots=paper_roots,
        )

    def search(
        self,
        query: str,
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        return self.search_many(
            [query],
            top_k=top_k,
            paper_sha256s=paper_sha256s,
        )

    def search_many(
        self,
        queries: Sequence[str],
        top_k: int = 8,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        clean_queries = list(dict.fromkeys(str(item).strip() for item in queries if str(item).strip()))[:6]
        return self._search_queries(clean_queries, top_k=top_k, paper_sha256s=paper_sha256s)

    def evidence_candidates(self, query: str, *, paper_sha256s: set[str]) -> Sequence[Mapping[str, Any]]:
        """Frozen select-v2 pool: single source, budget 64, first 24 pre-MMR hits.

        Keep this separate from search(top_k): changing final context size must not
        accidentally change candidate construction. A fresh adapter avoids mutable
        per-request budget overrides on a shared retriever.
        """
        if (len(paper_sha256s) != 1 or self.embedder is None or self.reranker is None
                or self.rerank_stage != 'final'):
            raise ValueError('Evidence selection requires one paper and dense/final reranking enabled')
        scoped = PaperPilotKnowledgeSearchAdapter(self.store, embedder=self.embedder,
            reranker=self.reranker, paper_roots=self.paper_roots, rerank_stage='final',
            max_chunks_per_paper=self.max_chunks_per_paper, scoped_candidate_k=64)
        return scoped._search_queries([query.strip()] if query.strip() else [],
            top_k=8, paper_sha256s=paper_sha256s, candidates_only=True)

    def _search_queries(self, clean_queries, *, top_k, paper_sha256s, candidates_only=False):
        if not clean_queries:
            return []
        candidate_k = max(top_k, min(100, top_k * 3))
        scoped_reranking = bool(self.reranker is not None and self.rerank_stage == 'final'
                                and paper_sha256s is not None and len(paper_sha256s) == 1)
        if scoped_reranking and self.scoped_candidate_k is not None:
            candidate_k = max(top_k, self.scoped_candidate_k)
        result = self.store.retrieve_queries_with_memory(
            clean_queries,
            top_k=candidate_k,
            embedder=self.embedder,
            # Diversity is applied after candidate recall. A hard cap here
            # irreversibly drops answers when the corpus/scope has one paper.
            max_chunks_per_paper=(None if scoped_reranking and self.scoped_candidate_k is not None
                                  else min(20, candidate_k)),
            reranker=self.reranker if self.rerank_stage == "early" else None,
            paper_sha256s=paper_sha256s,
        )
        items: list[dict[str, Any]] = []
        for rank, hit in enumerate(result.hits, start=1):
            items.append(
                self._hit_to_item(
                    hit,
                    rank=rank,
                    memory_ids=result.recalled_memory_ids,
                )
            )

        # A named paper is not just another diversity candidate.  Retrieve a
        # small query-specific set inside that paper, then reserve bounded slots
        # for it.  Global retrieval still fills the rest, so explicit routing is
        # a soft source anchor unless the caller supplied a hard SHA scope.
        explicit_scope = self.resolve_explicit_title_scope(
            clean_queries,
            paper_sha256s=paper_sha256s,
        )
        targeted = self._retrieve_explicit_title_hits(
            clean_queries,
            top_k=top_k,
            paper_sha256s=paper_sha256s,
        )
        by_stable_id = {
            str(item.get("stable_id")): item
            for item in items
            if item.get("stable_id")
        }
        for target in targeted:
            stable_id = str(target.get("stable_id") or "")
            existing = by_stable_id.get(stable_id)
            if existing is None:
                items.append(target)
                by_stable_id[stable_id] = target
                continue
            existing["score"] = max(
                float(existing.get("score", 0.0)),
                float(target.get("score", 0.0)),
            )
            existing["channels"] = list(
                dict.fromkeys(
                    list(existing.get("channels") or [])
                    + list(target.get("channels") or [])
                )
            )
            existing["explicit_title_scope"] = True
            existing["resolved_title_scope_size"] = len(explicit_scope)

        priority_quota = min(top_k, max(2, min(4, max(1, top_k // 2)))) if explicit_scope else 0
        if self.reranker is not None and self.rerank_stage == "final" and items:
            # Rank the merged pool once, after source routing and query fusion.
            # Earlier scores are otherwise replaced by rank fusion downstream.
            scores = self.reranker.score(clean_queries[0], [str(i["content"]) for i in items])
            if len(scores) != len(items) or not all(math.isfinite(float(s)) for s in scores):
                raise ValueError("invalid final reranker scores")
            for item, score in zip(items, scores, strict=True):
                item["score"] = float(score)
                item["channels"] = list(dict.fromkeys([*item.get("channels", []), "final_cross_encoder"]))
            items.sort(key=lambda i: -i["score"])
        selected = items[:24] if candidates_only else self._diversify(
            items,
            top_k=top_k,
            diversity=self.scoped_diversity if scoped_reranking else 0.22,
            priority_papers=explicit_scope,
            priority_quota=priority_quota,
        )
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

    def resolve_explicit_title_scope(
        self,
        queries: Sequence[str] | str,
        *,
        paper_sha256s: set[str] | None = None,
    ) -> set[str]:
        resolver = getattr(self.store, "resolve_explicit_title_scope", None)
        if not callable(resolver):
            return set()
        values = [queries] if isinstance(queries, str) else list(queries)
        scope: set[str] = set()
        for query in values:
            text = str(query).strip()
            if text:
                scope.update(resolver(text, paper_sha256s=paper_sha256s))
        return scope if len(scope) <= 8 else set()

    def resolve_pdf_paths(self, paper_sha256s: set[str]) -> list[str]:
        """Resolve indexed SHA values only inside server-configured PDF roots."""

        metadata_loader = getattr(self.store, "paper_metadata", None)
        if not callable(metadata_loader) or not self.paper_roots:
            return []
        if self._pdf_name_index is None:
            index: dict[str, list[Path]] = {}
            for root in self.paper_roots:
                if root.is_file() and root.suffix.casefold() == ".pdf":
                    index.setdefault(root.name.casefold(), []).append(root)
                    continue
                if not root.is_dir():
                    continue
                for path in root.rglob("*.pdf"):
                    if path.is_file():
                        index.setdefault(path.name.casefold(), []).append(path.resolve())
            self._pdf_name_index = index
        result: list[str] = []
        for record in metadata_loader(set(paper_sha256s)):
            paper_sha = str(record.get("sha256") or "")
            if not paper_sha:
                continue
            if paper_sha not in self._resolved_pdf_cache:
                resolved: str | None = None
                candidates = self._pdf_name_index.get(
                    str(record.get("file_name") or "").casefold(), []
                )
                for candidate in sorted(candidates, key=lambda value: str(value).casefold()):
                    digest = hashlib.sha256()
                    with candidate.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() == paper_sha:
                        resolved = str(candidate)
                        break
                self._resolved_pdf_cache[paper_sha] = resolved
            path = self._resolved_pdf_cache[paper_sha]
            if path and path not in result:
                result.append(path)
        return result

    def resolve_paper_titles(self, paper_sha256s: set[str]) -> list[str]:
        """Return canonical indexed titles for named-paper scholarly fallback."""

        metadata_loader = getattr(self.store, "paper_metadata", None)
        if not callable(metadata_loader):
            return []
        try:
            records = metadata_loader(set(paper_sha256s))
        except (OSError, TypeError, ValueError):
            return []
        return list(
            dict.fromkeys(
                str(record.get("title") or "").strip()
                for record in records
                if str(record.get("title") or "").strip()
            )
        )[:8]

    def _retrieve_explicit_title_hits(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        paper_sha256s: set[str] | None,
    ) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        per_query_k = max(2, min(6, top_k))
        for query_index, query in enumerate(queries, start=1):
            query_scope = self.resolve_explicit_title_scope(
                query,
                paper_sha256s=paper_sha256s,
            )
            if not query_scope:
                continue
            hits = self.store.search_chunks(
                query,
                top_k=per_query_k,
                embedder=self.embedder,
                paper_sha256s=query_scope,
                max_chunks_per_paper=per_query_k,
                reranker=self.reranker if self.rerank_stage == "early" else None,
            )
            for rank, hit in enumerate(hits, start=1):
                item = combined.setdefault(
                    hit.stable_id,
                    {
                        "hit": hit,
                        "score": 0.0,
                        "channels": [],
                        "scope": set(),
                    },
                )
                item["score"] = float(item["score"]) + 1.0 / (60 + rank)
                item["scope"].update(query_scope)
                for channel in (*hit.channels, "explicit_title_scope", f"title_query_{query_index}"):
                    if channel not in item["channels"]:
                        item["channels"].append(channel)
        ranked = sorted(
            combined.values(),
            key=lambda item: (-float(item["score"]), item["hit"].stable_id),
        )
        results: list[dict[str, Any]] = []
        for rank, entry in enumerate(ranked, start=1):
            item = self._hit_to_item(entry["hit"], rank=rank, memory_ids=())
            item["score"] = float(entry["score"])
            item["channels"] = list(entry["channels"])
            item["explicit_title_scope"] = True
            item["resolved_title_scope_size"] = len(entry["scope"])
            results.append(item)
        return results

    @staticmethod
    def _hit_to_item(
        hit: Any,
        *,
        rank: int,
        memory_ids: Sequence[str],
    ) -> dict[str, Any]:
        from .source_positions import chunk_position
        channels = tuple(hit.channels)
        if "cross_encoder" in channels:
            confidence = 0.82
        elif "dense" in channels and "bm25" in channels:
            confidence = 0.74
        elif "memory_evidence" in channels:
            confidence = 0.68
        else:
            confidence = 0.58
        return {
            "content": f"{hit.title}\n{hit.section}\n{hit.text}".strip(),
            "source_uri": f"paper://{hit.paper_sha256}/{hit.stable_id}",
            **chunk_position(hit),
            "score": hit.score,
            "confidence": confidence,
            "rank": rank,
            "paper_sha256": hit.paper_sha256,
            "stable_id": hit.stable_id,
            "file_name": hit.file_name,
            "title": hit.title,
            "page_end": hit.page_end,
            "channels": list(channels),
            "memory_ids": list(memory_ids),
        }

    def _diversify(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        top_k: int,
        diversity: float = 0.22,
        priority_papers: set[str] | None = None,
        priority_quota: int = 0,
        normalize_relevance: bool = True,
    ) -> list[Mapping[str, Any]]:
        """Paper-aware MMR over the already-ranked hybrid retrieval candidates."""

        remaining = [dict(item) for item in items]
        # RRF scores are typically hundredths; overlap penalties are 0..1.
        # Compare them on the same scale instead of letting MMR erase relevance.
        scores = [float(item.get("score", 0.0)) for item in remaining]
        low, high = (min(scores), max(scores)) if scores else (0.0, 0.0)
        selected: list[dict[str, Any]] = []
        per_paper: dict[str, int] = {}

        def tokens(item: Mapping[str, Any]) -> set[str]:
            return {token.lower() for token in _TOKEN_RE.findall(str(item.get("content") or ""))}

        token_cache = {id(item): tokens(item) for item in remaining}
        priority_papers = set(priority_papers or ())

        def select_one(*, priority_only: bool = False, allow_overflow: bool = False) -> bool:
            best_index = -1
            best_value = float("-inf")
            for index, item in enumerate(remaining):
                paper = str(item.get("paper_sha256") or item.get("source_uri") or "unknown")
                if priority_only and paper not in priority_papers:
                    continue
                paper_cap = (
                    max(self.max_chunks_per_paper, priority_quota)
                    if paper in priority_papers
                    else self.max_chunks_per_paper
                )
                if not allow_overflow and per_paper.get(paper, 0) >= paper_cap:
                    continue
                relevance = float(item.get("score", 0.0))
                if normalize_relevance:
                    relevance = (relevance - low) / (high - low) if high > low else 1.0
                if selected:
                    current = token_cache[id(item)]
                    maximum_similarity = max(
                        len(current & token_cache[id(other)]) / max(1, len(current | token_cache[id(other)]))
                        for other in selected
                    )
                else:
                    maximum_similarity = 0.0
                value = relevance - diversity * maximum_similarity - 0.04 * per_paper.get(paper, 0)
                if value > best_value:
                    best_value = value
                    best_index = index
            if best_index < 0:
                return False
            item = remaining.pop(best_index)
            paper = str(item.get("paper_sha256") or item.get("source_uri") or "unknown")
            per_paper[paper] = per_paper.get(paper, 0) + 1
            item["diversity_score"] = best_value
            selected.append(item)
            return True

        while remaining and len(selected) < min(top_k, priority_quota):
            if not select_one(priority_only=True):
                break
        while remaining and len(selected) < top_k:
            if not select_one():
                break
        # A diversity preference must not leave an eight-slot context with only
        # three passages when no additional papers can supply the remaining slots.
        while remaining and len(selected) < top_k:
            if not select_one(allow_overflow=True):
                break
        return selected


def make_search_adapter(workspace: str | Path) -> PaperSearchAdapter:
    configured = os.environ.get("PAPERPILOT_SEARCH_ADAPTER")
    if configured:
        return CallablePaperSearchAdapter(configured)
    root = Path(workspace).expanduser().resolve()
    configured_db = os.environ.get("PAPERPILOT_MEMORY_DB")
    database_candidates = [
        Path(configured_db).expanduser().resolve() if configured_db else None,
        root / "paper-knowledge.sqlite3",
        root / "outputs" / "paper-knowledge.sqlite3",
    ]
    for database in database_candidates:
        if database is not None and database.is_file():
            return PaperPilotKnowledgeSearchAdapter.from_database(
                database,
                namespace=os.environ.get("PAPERPILOT_MEMORY_NAMESPACE", "default"),
            )
    roots = [root / "papers", root / "data" / "papers", root / "corpus"]
    return LocalCorpusSearchAdapter(roots)


class VisionEvidenceAdapter(Protocol):
    def analyze(
        self,
        image_path: str,
        prompt: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class CallableVisionEvidenceAdapter:
    """Pluggable multimodal bridge loaded from ``module:function``.

    The callable can accept ``image_path``, ``prompt`` and ``metadata`` either as
    keyword parameters or as a single payload mapping. It must return a mapping with
    at least ``description`` or ``content``. This keeps credentials/model selection in
    PaperPilot's existing model layer instead of hard-coding a vendor client here.
    """

    def __init__(self, target: str | Callable[..., Any]):
        self.target = load_callable(target) if isinstance(target, str) else target

    def analyze(
        self,
        image_path: str,
        prompt: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        signature = inspect.signature(self.target)
        parameters = signature.parameters
        if len(parameters) == 1 and not any(
            name in parameters for name in ("image_path", "prompt", "metadata")
        ):
            raw = self.target(
                {"image_path": image_path, "prompt": prompt, "metadata": dict(metadata or {})}
            )
        else:
            kwargs: dict[str, Any] = {}
            if "image_path" in parameters:
                kwargs["image_path"] = image_path
            if "prompt" in parameters:
                kwargs["prompt"] = prompt
            if "metadata" in parameters:
                kwargs["metadata"] = dict(metadata or {})
            raw = self.target(**kwargs)
        if isinstance(raw, str):
            raw = {"description": raw}
        if not isinstance(raw, Mapping):
            raise TypeError("Vision adapter must return a mapping or string")
        value = dict(raw)
        if not (value.get("description") or value.get("content")):
            raise ValueError("Vision adapter returned no description/content")
        return value


def make_vision_adapter(
    *, api_key_file: str | Path | None = None
) -> VisionEvidenceAdapter | None:
    parser = os.getenv('PAPERPILOT_STRUCTURED_PARSER', 'legacy').strip().lower()
    if parser == 'docling':
        from .docling_adapter import DoclingEvidenceAdapter
        return DoclingEvidenceAdapter.from_env()
    if parser != 'legacy':
        raise ValueError('PAPERPILOT_STRUCTURED_PARSER must be legacy or docling')
    configured = os.environ.get("PAPERPILOT_VISION_ADAPTER")
    if configured:
        return CallableVisionEvidenceAdapter(configured)

    enabled = os.environ.get("PAPERPILOT_QWEN_VISION_ENABLED", "auto").strip().casefold()
    if enabled in {"0", "false", "no", "off", "disabled"}:
        return None
    if not (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("PAPERPILOT_QWEN_API_KEY_FILE")
        or api_key_file
    ):
        return None
    from .errors import ConfigurationError

    from .qwen_vision import QwenVisionEvidenceAdapter

    try:
        return QwenVisionEvidenceAdapter.from_env(api_key_file=api_key_file)
    except ConfigurationError:
        if enabled in {"1", "true", "yes", "on", "enabled"}:
            raise
        return None
