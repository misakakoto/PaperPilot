"""Opt-in immutable pilot adapter. The existing library remains the default."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from threading import RLock
from types import SimpleNamespace

from paperpilot_retrieval import PaperKnowledgeStore

from .adapters import PaperPilotKnowledgeSearchAdapter
from .paper_structure import PaperTextNavigator, VERSION, digest


def pilot_identity(root):
    base = digest((root / "index.json").read_bytes())
    path = root / "object-links.json"
    if not path.exists():
        return base
    links = json.loads(path.read_bytes())
    if links["source_index_identity"] != base:
        raise ValueError("Object links belong to another source index")
    return digest((base + digest(path.read_bytes())).encode())


class ReadOnlyPilotStore(PaperKnowledgeStore):
    def __init__(self, path, graph_by_chunk):
        self.path, self.namespace = path.resolve(), "structured-pilot-v1"
        self.lexical_query_policy, self._lock = "content_v1", RLock()
        self.graph_by_chunk = graph_by_chunk

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def get_chunks(self, stable_ids):
        return [
            SimpleNamespace(**asdict(hit), source_spans=self.graph_by_chunk[hit.stable_id])
            if hit.stable_id in self.graph_by_chunk
            else hit
            for hit in super().get_chunks(stable_ids)
        ]


class StructuredPilotAdapter(PaperPilotKnowledgeSearchAdapter):
    def __init__(self, root, legacy):
        self.pilot_root = root
        manifest = json.loads((root / "index.json").read_bytes())
        if (
            manifest["version"] != VERSION
            or digest((root / "corpus.sqlite3").read_bytes()) != manifest["database_sha256"]
        ):
            raise ValueError("Pilot index version/content changed")
        self.index_identity = pilot_identity(root)
        self.navigator = PaperTextNavigator(root)
        self.graphs = {
            p["sha256"]: self.navigator._graph(p["sha256"])[1] for p in manifest["papers"]
        }
        self.chunk_sources = {
            c["stable_id"]: c["parts"] for g in self.graphs.values() for c in g["chunks"]
        }
        self.paragraph_chunks = {
            (sha, p["paragraph_id"]): [
                c["stable_id"]
                for c in graph["chunks"]
                if any(part["paragraph_id"] == p["paragraph_id"] for part in c["parts"])
            ]
            for sha, graph in self.graphs.items()
            for p in graph["paragraphs"]
        }
        self.retrieval_capabilities = {
            "dense": manifest["embedding_model"] is not None,
            "dense_status": manifest.get("dense_status"),
            "lexical": True,
        }
        super().__init__(
            ReadOnlyPilotStore(root / "corpus.sqlite3", self.chunk_sources),
            embedder=getattr(legacy, "embedder", None) if manifest["embedding_model"] else None,
            reranker=getattr(legacy, "reranker", None),
            paper_roots=getattr(legacy, "paper_roots", ()),
            rerank_stage=getattr(legacy, "rerank_stage", "final"),
        )
        self.store.navigator_adapter = self

    def _hit_to_item(self, hit, *, rank, memory_ids):
        item = super()._hit_to_item(hit, rank=rank, memory_ids=memory_ids)
        item["provenance"].update(
            index_identity=self.index_identity,
            source_segments=self.chunk_sources[hit.stable_id],
            retrieval_profile="structured_pilot",
        )
        return item

    def find_text(self, paper_sha256, query):
        return self.navigator.find_text(paper_sha256, query)

    def read_matches(self, paper_sha256, matches):
        ids = [r["paragraph_id"] for r in matches]
        read = self.navigator.read(paper_sha256, ids, max_bytes=6000)
        chunk_ids = list(
            dict.fromkeys(
                cid
                for p in read["paragraphs"]
                for cid in self.paragraph_chunks[(paper_sha256, p["paragraph_id"])]
            )
        )
        rows, used = [], 0
        for h in self.store.get_chunks(chunk_ids):
            item = self._hit_to_item(h, rank=len(rows) + 1, memory_ids=[])
            size = len(item["content"].encode())
            if used + size > 6000:
                continue
            rows.append(item | {"evidence_id": "e_" + h.stable_id})
            used += size
        return rows


def select_retriever(runtime, metadata):
    profile = metadata.get("retrieval_profile", "legacy")
    if profile not in {"legacy", "structured_pilot", "token_evidence_pilot", "paper_evidence_v2", "paper_active_v1"}:
        raise ValueError("Unknown retrieval_profile")
    if profile == "legacy":
        return runtime.search_adapter, "legacy"
    if profile == 'paper_active_v1':
        from .active_paper_sources import ActivePaperSources
        adapter = ActivePaperSources(runtime.search_adapter)
        adapter.index_identity = adapter.identity
        adapter.retrieval_capabilities = {'lexical': True, 'dense': runtime.search_adapter.embedder is not None,
            'active_reading': True, 'backend': 'unchanged_legacy_hybrid'}
        sources = set(adapter.papers)
    elif profile == 'paper_evidence_v2':
        from .paper_evidence import PaperEvidenceAdapter
        adapter = PaperEvidenceAdapter(runtime.workspace / 'paper-evidence-v2', runtime.search_adapter)
        sources = set(adapter.library.papers)
    elif profile == 'token_evidence_pilot':
        from .token_evidence import TokenEvidenceAdapter
        adapter = TokenEvidenceAdapter(runtime.workspace / 'token-evidence-pilot', runtime.search_adapter)
        sources = {p['sha256'] for p in adapter.manifest['papers']}
    else:
        adapter = StructuredPilotAdapter(runtime.workspace / "structured-pilot", runtime.search_adapter)
        sources = set(adapter.graphs)
    if set(metadata.get("paper_sha256s") or []) - sources:
        raise ValueError("Selected sources are outside the frozen pilot")
    if metadata.get("retrieval_identity") not in (None, adapter.index_identity):
        raise ValueError("Frozen pilot index changed; create a new task")
    return adapter, adapter.index_identity
