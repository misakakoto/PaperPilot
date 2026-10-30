from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import (
    LocalCorpusSearchAdapter,
    PaperPilotKnowledgeSearchAdapter,
    VisionEvidenceAdapter,
)
from .orchestrator import ResearchAgentRuntime
from .scholarly import ScholarlySearchAdapter


def build_paperpilot_runtime(
    *,
    db_path: str | Path,
    workspace: str | Path,
    knowledge_store: Any = None,
    embedder: Any = None,
    reranker: Any = None,
    llm: Any = None,
    paper_roots: Any = None,
    vision_adapter: VisionEvidenceAdapter | None = None,
    scholarly_adapter: ScholarlySearchAdapter | None = None,
    experiment_runner: Any = None,
) -> ResearchAgentRuntime:
    """Wire the durable runtime to PaperPilot's real retrieval and model layers."""
    search_adapter = (
        PaperPilotKnowledgeSearchAdapter(
            knowledge_store,
            embedder=embedder,
            reranker=reranker,
            paper_roots=paper_roots,
        )
        if knowledge_store is not None
        else LocalCorpusSearchAdapter(paper_roots or [Path(workspace) / "papers"])
    )
    return ResearchAgentRuntime(
        db_path=db_path,
        workspace=workspace,
        search_adapter=search_adapter,
        scholarly_adapter=scholarly_adapter,
        vision_adapter=vision_adapter,
        llm=llm,
        experiment_runner=experiment_runner,
    )
