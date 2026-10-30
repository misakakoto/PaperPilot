"""Prospectively frozen, non-oracle production-workbench QASPER dev diagnostic.

freeze -> run -> score; original questions/answers and all failures stay on disk.
This is a small given-paper, text-only diagnostic, not a full benchmark claim.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import time

from evals.qasper.agent_case import paper_records
from evals.qasper.score_oracle import load_evaluator


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity():
    h = hashlib.sha256()
    for folder in (ROOT / "src/paperpilot_agentic", ROOT / "src/paperpilot_retrieval"):
        for p in sorted(folder.rglob("*.py")):
            h.update(p.relative_to(ROOT).as_posix().encode())
            h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def freeze(dataset, manifest, *, count=8, seed="20260926", model="qwen-plus"):
    data = json.loads(dataset.read_text(encoding="utf-8-sig"))
    candidates = [
        {"paper_id": str(pid), "question_id": qa["question_id"], "question": qa["question"]}
        for pid, paper in data.items()
        for qa in paper["qas"]
    ]
    if len({r["question_id"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate questions")
    candidates.sort(
        key=lambda r: hashlib.sha256((seed + ":" + r["question_id"]).encode()).hexdigest()
    )
    if not 1 <= count <= len(candidates):
        raise ValueError("Invalid sample size")
    value = {
        "kind": "production-workbench-qasper-dev-v1",
        "dataset_sha256": digest(dataset),
        "code_sha256": code_identity(),
        "model": model,
        "provider": "qwen",
        "count": count,
        "seed": seed,
        "cases": candidates[:count],
        "selection": "SHA256(seed:question_id), no answer/evidence/outcome filtering",
        "question_policy": "original, unchanged; correct paper supplied; no gold evidence supplied",
        "action": "auto",
        "visual_policy": "off; QASPER adapter has no page images",
        "score_policy": "all frozen cases; failures/missing outputs zero; raw and citation-stripped answer F1",
        "evidence_policy": "all full public paragraphs contained in cited original-source records; no gold-based selection",
        "limits": {
            "max_llm_calls_per_case": 20,
            "max_tokens_per_case": 100000,
            "max_seconds_per_case": 3600,
        },
        "full_benchmark": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    save(manifest, value)
    return value


def run(dataset, manifest, workspace):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.web_api import create_fastapi_router
    from paperpilot_retrieval import PaperKnowledgeStore

    protocol = json.loads(manifest.read_text(encoding="utf-8"))
    if digest(dataset) != protocol["dataset_sha256"] or code_identity() != protocol["code_sha256"]:
        raise ValueError("Frozen data or code changed")
    workspace.mkdir(parents=True, exist_ok=False)
    save(workspace / "protocol.json", protocol)
    data = json.loads(dataset.read_text(encoding="utf-8-sig"))
    base = RuntimeSettings.from_env()
    os.environ.setdefault(
        "PAPERPILOT_EMBEDDING_CACHE", str(base.workspace / "model-cache/embedding")
    )
    os.environ.setdefault("PAPERPILOT_RERANKER_CACHE", str(base.workspace / "model-cache/reranker"))
    settings = replace(
        base,
        provider="qwen",
        api_key=resolve_provider_api_key("qwen"),
        model=protocol["model"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        thinking_enabled=False,
    )
    if not settings.api_key:
        raise ValueError("Qwen key missing; no requests made")
    for index, case in enumerate(protocol["cases"]):
        if code_identity() != protocol["code_sha256"]:
            raise ValueError("Code changed during frozen run")
        folder = workspace / f"case-{index:02}"
        folder.mkdir()
        started = time.monotonic()
        rid = None
        try:
            public, doc, chunks = paper_records(data[case["paper_id"]], source_id=case["paper_id"])
            save(folder / "public-paper.json", public)
            store = PaperKnowledgeStore(folder / "knowledge.sqlite3", namespace="qasper-workbench")
            current = replace(
                settings,
                workspace=folder,
                runtime_db=folder / "runtime.sqlite3",
                memory_db=folder / "knowledge.sqlite3",
                memory_namespace="qasper-workbench",
                paper_roots=(folder,),
            )
            runtime = build_runtime(current)
            store.sync_documents([doc], chunks, embedder=runtime.search_adapter.embedder)
            # Reuse the real HTTP workbench boundary, not an oracle/direct-LLM shortcut.
            app = FastAPI()
            app.include_router(
                create_fastapi_router(runtime_factory=lambda: runtime, prefix="/api/agentic")
            )
            with TestClient(app) as client:
                response = client.post(
                    "/api/agentic/turns",
                    json={
                        "request_id": f"qasper-{index:04}",
                        "action": protocol["action"],
                        "question": case["question"],
                        "paper_ids": [doc.sha256],
                        "visual_policy": "off",
                    },
                )
                response.raise_for_status()
                rid = response.json()["run_id"]
                while True:
                    result = runtime.status(rid, include_trace=False)
                    view = client.get(f"/api/agentic/runs/{rid}/view").json()
                    if (
                        result["run"]["status"]
                        in {"COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXHAUSTED"}
                        and not view["worker_active"]
                    ):
                        break
                    if time.monotonic() - started > 3660:
                        raise TimeoutError("Observation timed out; do not silently restart")
                    time.sleep(0.5)
            save(folder / "result.json", result)
            save(folder / "view.json", view)
            save(
                folder / "case.json",
                {
                    **case,
                    "run_id": rid,
                    "elapsed_seconds": time.monotonic() - started,
                    "gold_sent": False,
                    "dense": runtime.search_adapter.embedder is not None,
                    "rerank": runtime.search_adapter.reranker is not None,
                    "embedding_identity": getattr(
                        runtime.search_adapter.embedder, "cache_identity", None
                    ),
                    "reranker_sha256": getattr(
                        runtime.search_adapter.reranker, "model_sha256", None
                    ),
                },
            )
            print(
                json.dumps(
                    {
                        "case": index,
                        "status": result["run"]["status"],
                        "calls": result["run"]["used_llm_calls"],
                        "tokens": result["run"]["used_tokens"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            save(
                folder / "failure.json",
                {
                    "type": type(exc).__name__,
                    "run_id": rid,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(json.dumps({"case": index, "failure": type(exc).__name__}), flush=True)


def predicted_paragraphs(paper, result):
    final = result.get("final_answer") or {}
    cited = set(final.get("citation_ids") or [])
    def normalize(s):
        return " ".join(s.split())
    contents = [
        normalize(e["content"]) for e in result.get("evidence", []) if e["evidence_id"] in cited
    ]
    paragraphs = [p for section in paper["full_text"] for p in section["paragraphs"]]
    paragraphs += [paper["abstract"]]
    return list(
        dict.fromkeys(
            p for p in paragraphs if p.strip() and any(normalize(p) in text for text in contents)
        )
    )


def score(dataset, manifest, workspace, evaluator_path):
    protocol = json.loads(manifest.read_text(encoding="utf-8"))
    if digest(dataset) != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed")
    data = json.loads(dataset.read_text(encoding="utf-8-sig"))
    evaluator = load_evaluator(evaluator_path)
    gold = evaluator.get_answers_and_evidence(data, True)
    rows = []
    for index, case in enumerate(protocol["cases"]):
        folder = workspace / f"case-{index:02}"
        file = folder / "result.json"
        result = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
        run = result.get("run", {})
        final = result.get("final_answer") or {}
        original = final.get("answer") or ""
        valid = run.get("status") == "COMPLETED" and bool(original.strip())
        clean = re.sub(r"\[evidence_[A-Za-z0-9_-]+\]", "", original).strip()
        evidence = predicted_paragraphs(data[case["paper_id"]], result) if valid else []
        refs = gold[case["question_id"]]
        rows.append(
            {
                **case,
                "run_status": run.get("status", "MISSING"),
                "valid": valid,
                "original_answer": original,
                "predicted_evidence": evidence,
                "raw_answer_f1": max(evaluator.token_f1_score(original, r["answer"]) for r in refs)
                if valid
                else 0,
                "citation_free_answer_f1": max(
                    evaluator.token_f1_score(clean, r["answer"]) for r in refs
                )
                if valid
                else 0,
                "text_evidence_f1": max(
                    evaluator.paragraph_f1_score(evidence, r["evidence"]) for r in refs
                )
                if valid
                else 0,
                "calls": run.get("used_llm_calls", 0),
                "tokens": run.get("used_tokens", 0),
                "request_duration_ms": sum(
                    c.get("duration_ms") or 0 for c in result.get("llm_calls", [])
                ),
                "cost_cny": None,
                "failure_detail": (run.get("metadata", {}).get("reader_report") or {}).get(
                    "failure_detail"
                )
                or run.get("error"),
                "result_sha256": digest(file) if file.exists() else None,
            }
        )
    means = {
        k: sum(r[k] for r in rows) / len(rows)
        for k in ("raw_answer_f1", "citation_free_answer_f1", "text_evidence_f1")
    }
    return {
        "kind": protocol["kind"],
        "scope": "frozen small dev given-paper text-only production-workbench diagnostic",
        "full_benchmark": False,
        "factual_accuracy_measured": False,
        "frozen_count": len(rows),
        "failures": sum(not r["valid"] for r in rows),
        "metrics": means,
        "rows": rows,
        "total_calls": sum(r["calls"] for r in rows),
        "total_tokens": sum(r["tokens"] for r in rows),
        "pricing_note": "Use actual provider invoices for monetary cost; missing configured prices are not free calls.",
        "evaluator_sha256": digest(evaluator_path),
        "manifest_sha256": digest(manifest),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["freeze", "run", "score"])
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--workspace", type=Path)
    p.add_argument("--evaluator", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", default="20260926")
    p.add_argument("--model", default="qwen-plus")
    a = p.parse_args()
    if a.phase == "freeze":
        freeze(a.dataset, a.manifest, count=a.count, seed=a.seed, model=a.model)
    elif a.phase == "run":
        run(a.dataset, a.manifest, a.workspace)
    else:
        save(a.output, score(a.dataset, a.manifest, a.workspace, a.evaluator))


if __name__ == "__main__":
    main()
