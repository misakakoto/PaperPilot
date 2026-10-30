from types import SimpleNamespace

import pytest

from paperpilot_agentic.layout import fitz
from paperpilot_agentic.models import RunStatus
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.paper_structure import (
    VERSION,
    build_structure,
    digest,
    graph_identity,
    save_json,
)
from paperpilot_retrieval import PaperKnowledgeStore


def test_request_profile_is_frozen_and_host_tools_respect_scope(tmp_path):
    workspace = tmp_path / "workspace"
    root = workspace / "structured-pilot"
    root.mkdir(parents=True)
    pdf_path = tmp_path / "registered.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 50), "1. Methods", fontsize=16)
        page.insert_text((30, 90), "Alpha[1] preserves the experimental conditions.", fontsize=10)
        pdf.save(pdf_path)
    sha = digest(pdf_path.read_bytes())
    doc, chunks, graph = build_structure(
        pdf_path, root / "papers" / sha, title="Paper", paper_id="P1"
    )
    store = PaperKnowledgeStore(root / "corpus.sqlite3", namespace="structured-pilot-v1")
    store.sync_documents([doc], chunks)
    with store._connect() as connection:
        hits = connection.execute("select stable_id from chunks order by ordinal").fetchall()
    for row, hit in zip(graph["chunks"], hits):
        row["stable_id"] = hit["stable_id"]
    graph["identity"] = graph_identity(graph)
    graph_path = root / "papers" / sha / "structure.json"
    save_json(graph_path, graph)
    save_json(
        root / "index.json",
        {
            "version": VERSION,
            "database_sha256": digest((root / "corpus.sqlite3").read_bytes()),
            "embedding_model": None,
            "papers": [{"sha256": sha, "structure_sha256": digest(graph_path.read_bytes())}],
        },
    )
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=tmp_path / "runs.sqlite3")
    default = runtime.enqueue(question="A default question")["run"]
    assert "retrieval_identity" not in default["metadata"]
    run = runtime.enqueue(
        question="Explain Alpha",
        metadata={
            "engine": "answer_first_text",
            "retrieval_profile": "structured_pilot",
            "paper_sha256s": [sha],
        },
    )["run"]
    assert run["metadata"]["retrieval_identity"] == digest((root / "index.json").read_bytes())
    context = SimpleNamespace(run_id=run["run_id"], workspace=workspace, db=runtime.db)
    result = runtime.registry.get("paper.find_text").handler(
        context, {"paper_sha256": sha, "query": "Alpha[1]"}
    )
    assert result["status"] == "LOCATED"
    with pytest.raises(ValueError, match="scope"):
        runtime.registry.get("paper.find_text").handler(
            context, {"paper_sha256": "b" * 64, "query": "Alpha"}
        )
    runtime.db.update_run_status(run["run_id"], RunStatus.COMPLETED)
    child = runtime.enqueue_follow_up(parent_run_id=run["run_id"], question="Continue explaining")[
        "run"
    ]
    assert child["metadata"]["retrieval_identity"] == run["metadata"]["retrieval_identity"]
    (root / "index.json").write_bytes((root / "index.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        runtime.registry.get("paper.find_text").handler(
            context, {"paper_sha256": sha, "query": "Alpha"}
        )
