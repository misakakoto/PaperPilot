from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from paperpilot_retrieval import FastEmbedder, PaperKnowledgeStore
from paperpilot_retrieval.ingest import ingest_pdfs

from .bootstrap import build_runtime
from .config import RuntimeSettings
from .layout import parse_pdf_layout
from .models import Budget
from .orchestrator import ResearchAgentRuntime


def _json_file(path: str | None, default: Any) -> Any:
    if not path:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _emit(value: Any) -> None:
    # ASCII-escaped JSON remains machine-readable under legacy Windows GBK
    # consoles and avoids UnicodeEncodeError on scholarly symbols such as ©.
    print(json.dumps(value, ensure_ascii=True, indent=2, default=str))


def _runtime(args: argparse.Namespace) -> ResearchAgentRuntime:
    settings = RuntimeSettings.from_env()
    settings = dataclasses.replace(
        settings,
        runtime_db=Path(args.db).expanduser().resolve(),
        workspace=Path(args.workspace).expanduser().resolve(),
    )
    return build_runtime(settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperpilot-agentic",
        description="Durable evidence-gap runtime for PaperPilot",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("PAPERPILOT_AGENTIC_DB", ".paperpilot_agentic/runtime.sqlite3"),
        help="SQLite runtime database",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("PAPERPILOT_WORKSPACE", "."),
        help="PaperPilot workspace and artifact root",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Create and advance a durable research run")
    start.add_argument("--question", required=True)
    start.add_argument("--claims-json", help="JSON file containing a list of claim objects")
    start.add_argument("--edges-json", help="JSON file containing claim-graph edges")
    start.add_argument("--budget-json", help="JSON file containing runtime budget")
    start.add_argument("--no-auto-advance", action="store_true")

    status = sub.add_parser("status", help="Inspect a run")
    status.add_argument("run_id")
    status.add_argument("--no-trace", action="store_true")

    resume = sub.add_parser("resume", help="Resume after approval or interruption")
    resume.add_argument("run_id")

    approve = sub.add_parser("approve", help="Approve a waiting tool call")
    approve.add_argument("approval_id")
    approve.add_argument("--actor", default="human")
    approve.add_argument("--comment")
    approve.add_argument("--resume", action="store_true", help="Resume the owning run immediately")

    reject = sub.add_parser("reject", help="Reject a waiting tool call")
    reject.add_argument("approval_id")
    reject.add_argument("--actor", default="human")
    reject.add_argument("--comment")
    reject.add_argument("--resume", action="store_true")

    cancel = sub.add_parser("cancel", help="Cancel a run")
    cancel.add_argument("run_id")
    cancel.add_argument("--reason", default="Cancelled by user")

    sub.add_parser("tools", help="List registered tools and safety metadata")

    parse = sub.add_parser("parse-pdf", help="Create a layout-aware PDF evidence manifest")
    parse.add_argument("pdf_path")
    parse.add_argument("output_dir")
    parse.add_argument("--no-render", action="store_true")

    index = sub.add_parser("index-pdfs", help="Build or update the local paper index")
    index.add_argument("roots", nargs="+", help="PDF files or directories")
    index.add_argument("--memory-db", help="Destination SQLite knowledge database")
    index.add_argument("--namespace", default=os.environ.get("PAPERPILOT_MEMORY_NAMESPACE", "default"))
    index.add_argument("--dense", action="store_true", help="Also compute semantic embeddings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "parse-pdf":
            manifest = parse_pdf_layout(args.pdf_path, args.output_dir, render_regions=not args.no_render)
            _emit(manifest.to_dict())
            return 0
        if args.command == "index-pdfs":
            settings = RuntimeSettings.from_env()
            database = Path(
                args.memory_db
                or settings.memory_db
                or (settings.workspace / "paper-knowledge.sqlite3")
            ).expanduser().resolve()
            embedder = (
                FastEmbedder(
                    cache_dir=settings.workspace / "model-cache" / "embedding",
                    allow_download=True,
                )
                if args.dense
                else None
            )
            result = ingest_pdfs(
                PaperKnowledgeStore(database, namespace=args.namespace),
                args.roots,
                embedder=embedder,
            )
            _emit(dataclasses.asdict(result))
            return 0
        runtime = _runtime(args)
        if args.command == "start":
            value = runtime.start(
                question=args.question,
                claims=_json_file(args.claims_json, None),
                edges=_json_file(args.edges_json, None),
                budget=Budget.from_dict(_json_file(args.budget_json, {})),
                auto_advance=not args.no_auto_advance,
            )
        elif args.command == "status":
            value = runtime.status(args.run_id, include_trace=not args.no_trace)
        elif args.command == "resume":
            value = runtime.resume(args.run_id)
        elif args.command == "approve":
            approval = runtime.approve(args.approval_id, actor=args.actor, comment=args.comment)
            value = runtime.resume(approval["run_id"]) if args.resume else approval
        elif args.command == "reject":
            approval = runtime.reject(args.approval_id, actor=args.actor, comment=args.comment)
            value = runtime.resume(approval["run_id"]) if args.resume else approval
        elif args.command == "cancel":
            value = runtime.cancel(args.run_id, reason=args.reason)
        elif args.command == "tools":
            value = runtime.list_tools()
        else:  # pragma: no cover
            raise AssertionError(args.command)
        _emit(value)
        return 0
    except Exception as exc:
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
