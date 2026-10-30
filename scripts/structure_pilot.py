"""Freeze/build a 12-paper offline pilot, never replace the configured library."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot_retrieval import PaperKnowledgeStore, FastEmbedder

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.paper_structure import (
    VERSION,
    build_structure,
    digest,
    graph_identity,
    save_json,
)
from evals.interview.cases import CHANGE, FLOW
from evals.mcp_comparison.cases import cases


def freeze(settings, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "sources.json"
    if path.exists():
        return json.loads(path.read_bytes())
    with sqlite3.connect(settings.memory_db.as_uri() + "?mode=ro", uri=True) as c:
        c.row_factory = sqlite3.Row
        all_rows = [
            dict(r)
            for r in c.execute(
                "SELECT sha256,title,file_name FROM papers "
                "WHERE namespace=? AND active=1 ORDER BY sha256",
                (settings.memory_namespace,),
            )
        ]
    selected = [next(r for r in all_rows if r["sha256"] == sha) for sha in (CHANGE, FLOW)]
    selected += [r for r in all_rows if r["sha256"] not in {CHANGE, FLOW}][:10]
    files = [p for root in settings.paper_roots for p in root.rglob("*.pdf")]
    for row in selected:
        matches = [
            p
            for p in files
            if p.name == row["file_name"] and digest(p.read_bytes()) == row["sha256"]
        ]
        if not matches:
            raise ValueError("Registered original missing: " + row["file_name"])
        row["path"] = str(sorted(matches)[0])
        row["role"] = (
            "development" if row["sha256"] in {CHANGE, FLOW} else "heldout_from_this_repair"
        )
    result = {
        "version": VERSION,
        "baseline": "e75a9ae",
        "namespace": settings.memory_namespace,
        "source_db": str(settings.memory_db),
        "papers": selected,
        "selection": "ChangeMamba/FlowMamba, then first ten active distinct SHAs; no parse-quality selection",
    }
    save_json(path, result)
    # Preserve original pages for annotation independently of the new chunks.
    from paperpilot_agentic.layout import fitz

    for row in selected:
        with fitz.open(row["path"]) as doc:
            save_json(
                output / "original-pages" / (row["sha256"] + ".json"),
                [{"page": i + 1, "text": p.get_text(sort=True)} for i, p in enumerate(doc)],
            )
    save_json(output / "original-eleven-questions.json", cases(Path.cwd()))
    return result


def build(settings, output):
    frozen = freeze(settings, output)
    root = settings.workspace / "structured-pilot"
    if (root / "index.json").exists():
        raise ValueError(
            "Frozen index already exists; retain it or explicitly create a new version"
        )
    root.mkdir(exist_ok=True)
    docs, chunks, graphs = [], [], []
    for i, row in enumerate(frozen["papers"], 1):
        if digest(Path(row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("Frozen PDF changed")
        doc, parts, graph = build_structure(
            row["path"], root / "papers" / row["sha256"], title=row["title"], paper_id=f"P{i}"
        )
        docs.append(doc)
        chunks.extend(parts)
        graphs.append(graph)
        print(
            json.dumps(
                {
                    "paper": i,
                    "title": row["title"],
                    "paragraphs": len(graph["paragraphs"]),
                    "chunks": len(parts),
                    "sections": len(graph["sections"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    store = PaperKnowledgeStore(root / "corpus.sqlite3", namespace="structured-pilot-v1")
    embedder = None
    if os.getenv("PAPERPILOT_ENABLE_DENSE", "").lower() in ("true", "1", "yes"):
        embedder = FastEmbedder(
            cache_dir=Path(
                os.getenv(
                    "PAPERPILOT_EMBEDDING_CACHE", str(settings.workspace / "model-cache/embedding")
                )
            ),
            model_name=os.getenv(
                "PAPERPILOT_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            allow_download=False,
        )
    dense_status = "DISABLED_BY_CONFIGURATION"
    if embedder is not None:
        try:
            embedder.cache_identity
            dense_status = "ENABLED"
        except RuntimeError as exc:
            dense_status = "UNAVAILABLE_PROVENANCE: " + str(exc)
            embedder = None
    sync = store.sync_documents(docs, chunks, embedder=embedder)
    manifest = []
    with store._connect() as c:
        for row, doc, graph in zip(frozen["papers"], docs, graphs):
            stored = c.execute(
                "SELECT c.stable_id,c.ordinal FROM chunks c JOIN papers p ON p.row_id=c.paper_row_id "
                "WHERE c.namespace=? AND p.sha256=? ORDER BY c.ordinal",
                (store.namespace, doc.sha256),
            ).fetchall()
            assert len(stored) == len(graph["chunks"])
            for chunk, hit in zip(graph["chunks"], stored):
                chunk["stable_id"] = hit["stable_id"]
            graph["identity"] = graph_identity(graph)
            path = root / "papers" / doc.sha256 / "structure.json"
            save_json(path, graph)
            manifest.append(
                {
                    **row,
                    "structure_sha256": digest(path.read_bytes()),
                    "structure_identity": graph["identity"],
                }
            )
    save_json(
        root / "index.json",
        {
            "version": VERSION,
            "papers": manifest,
            "database_sha256": digest((root / "corpus.sqlite3").read_bytes()),
            "embedding_model": getattr(embedder, "name", None),
            "dense_status": dense_status,
            "namespace": store.namespace,
            "baseline_source_sha256": digest((output / "sources.json").read_bytes()),
        },
    )
    save_json(
        output / "build.json",
        {
            "version": VERSION,
            "documents": len(docs),
            "chunks": len(chunks),
            "objects": sum(len(g["objects"]) for g in graphs),
            "embedding_model": getattr(embedder, "name", None),
            "dense_status": dense_status,
            "index_identity": digest((root / "index.json").read_bytes()),
            "new_provider_requests": 0,
            "source_library_unchanged": True,
            "sync_result": str(sync),
        },
    )
    link_objects(settings, output)


def link_objects(settings, output):
    from paperpilot_agentic.reviewed_sources import ReviewedSources
    from paperpilot_agentic.structured_extraction import SourceRegion

    root = settings.workspace / "structured-pilot"
    manifest = json.loads((root / "index.json").read_bytes())
    registry = ReviewedSources.configured()
    reviewed = []
    for path in registry.root.glob("*.json"):
        payload = json.loads(path.read_bytes())
        region = SourceRegion.model_validate(payload["artifact"]["source"])
        valid = registry.resolve(region)
        if valid:
            reviewed.append(valid)
    links = []
    for paper in manifest["papers"]:
        graph = json.loads((root / "papers" / paper["sha256"] / "structure.json").read_bytes())
        paragraphs = graph["paragraphs"]
        indices = {p["paragraph_id"]: i for i, p in enumerate(paragraphs)}
        for obj in graph["objects"]:
            i = indices[obj["paragraph_id"]]
            assets = [
                a
                for a in reviewed
                if a["artifact"]["source"]["paper_sha256"] == paper["sha256"]
                and a["artifact"]["source"]["object_id"] == obj["object_id"]
                and a["artifact"]["source"]["page"] == obj["page"]
            ]
            links.append(
                {
                    "paper_sha256": paper["sha256"],
                    "object_id": obj["object_id"],
                    "kind": obj["kind"],
                    "section_id": obj["section_id"],
                    "definition_context_ids": [
                        p["paragraph_id"]
                        for p in paragraphs[max(0, i - 1) : i + 2]
                        if p["section_id"] == obj["section_id"]
                    ],
                    "context_status": "NEARBY_ORIGINALS_NOT_SEMANTICALLY_VERIFIED_DEFINITIONS",
                    "original_image_locator": {
                        "page": obj["page"],
                        "bbox": obj["bbox"],
                        "coordinate_space": obj["coordinate_space"],
                        "region_scope": "caption_only"
                        if obj["kind"] in {"table", "figure"}
                        else "detected_equation_region",
                        "rendered": False,
                    },
                    "existing_parser_assets": [
                        {
                            "asset_id": a["asset_id"],
                            "source": a["artifact"]["source"],
                            "parser": a["artifact"]["parser"],
                            "parser_version": a["artifact"]["parser_version"],
                            "reviewer_kind": a["review"]["reviewer_kind"],
                            "status": "EXACT_REGISTERED_IDENTITY_LINK; no new factual approval",
                        }
                        for a in assets
                    ],
                }
            )
    value = {
        "version": "paper-object-links-v1",
        "source_index_identity": digest((root / "index.json").read_bytes()),
        "reviewed_registry_identity": registry.identity(),
        "objects": links,
    }
    path = root / "object-links.json"
    if path.exists() and json.loads(path.read_bytes()) != value:
        raise ValueError(
            "Object link registry changed; freeze a new version instead of overwriting"
        )
    if not path.exists():
        save_json(path, value)
    save_json(
        output / "object-links-summary.json",
        {
            "objects": len(links),
            "registered_parser_links": sum(len(x["existing_parser_assets"]) for x in links),
            "all_locators_unverified_as_new_facts": True,
            "original_images_sent": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "build", "link"])
    parser.add_argument("--output", type=Path, default=Path("outputs/structure-pilot-20261020"))
    args = parser.parse_args()
    settings = RuntimeSettings.from_env()
    {"freeze": freeze, "build": build, "link": link_objects}[args.command](settings, args.output)
