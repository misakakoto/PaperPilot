"""Prepare or run a synthesis ablation on a frozen private Agent snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.direct_answer import answer_from_evidence, prepare_evidence  # noqa: E402
from paperpilot_agentic.providers import OpenAICompatibleLLM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".paperpilot/comparison.json")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    question = str((snapshot.get("run") or {}).get("question") or "")
    if not question.strip():
        raise ValueError("Snapshot must include run.question")
    rows = prepare_evidence(snapshot.get("evidence") or [])
    report = {
        "kind": "frozen_evidence_synthesis_ablation",
        "source_run_id": (snapshot.get("run") or {}).get("run_id"),
        "question": question,
        "evidence_count": len(rows),
        "original_answer": (snapshot.get("final_answer") or {}).get("answer"),
        "original_outcome": (snapshot.get("run") or {}).get("research_outcome"),
        "interpretation": (
            "This isolates the reader on existing Agent evidence. It is not an "
            "end-to-end retrieval benchmark or proof of Agent superiority."
        ),
    }
    if args.execute_paid:
        settings = RuntimeSettings.from_env(require_api_key=True)
        report["direct_reader"] = answer_from_evidence(
            question, rows, OpenAICompatibleLLM(settings, max_retries=0)
        )
    else:
        report["execution"] = "DRY_RUN_ZERO_PAID_REQUESTS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "paid_execution": args.execute_paid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
