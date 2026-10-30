"""Validate internal consistency of a saved PaperPilot QASPER score report."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute reported aggregates without treating them as a current result."""

    if report.get("benchmark_name") != "qasper-v0.3":
        raise ValueError("Report is not QASPER v0.3")
    if report.get("dataset_split") != "test":
        raise ValueError("Expected the frozen QASPER test split")
    cases = report.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("Report has no per-question cases")
    if len(cases) != int(report.get("case_count") or -1):
        raise ValueError("case_count does not match the case rows")

    identifiers = [str(item.get("question_id") or "") for item in cases]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Question IDs are empty or duplicated")

    def mean(field: str, rows: Sequence[Mapping[str, Any]]) -> float:
        return sum(float(item.get(field) or 0.0) for item in rows) / len(rows)

    recomputed = {
        "answer_f1": mean("answer_f1", cases),
        "answer_exact_match": mean("exact_match", cases),
        "evidence_f1": mean("evidence_f1", cases),
    }
    mismatches: dict[str, dict[str, float]] = {}
    for field, value in recomputed.items():
        recorded = float(report.get(field) or 0.0)
        if abs(recorded - value) > 1e-12:
            mismatches[field] = {"recorded": recorded, "recomputed": value}

    status_counts: defaultdict[str, int] = defaultdict(int)
    for item in cases:
        status_counts[str(item.get("status") or "missing")] += 1
    expected_scored = int(report.get("successful_predictions") or 0)
    expected_failed = int(report.get("failed_predictions") or 0)
    if status_counts["scored"] != expected_scored or status_counts["failed"] != expected_failed:
        raise ValueError("Prediction status counts do not match per-question cases")
    if mismatches:
        raise ValueError(f"Aggregate metric mismatch: {mismatches}")

    return {
        "validation": "historical_report_internal_consistency_v1",
        "passed": True,
        "benchmark": report["benchmark_name"],
        "split": report["dataset_split"],
        "case_count": len(cases),
        "scored": status_counts["scored"],
        "failed": status_counts["failed"],
        "recomputed": recomputed,
        "dataset_fingerprint": report.get("dataset_fingerprint"),
        "prediction_model": report.get("prediction_model"),
        "prompt_version": report.get("prompt_version"),
        "official_evaluator_parity_recorded": bool(
            report.get("official_evaluator_parity_verified")
        ),
        "scope": (
            "This verifies a saved report's arithmetic and identity fields. "
            "It does not rerun QASPER or score the current AnswerFirst code."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_report(json.loads(args.report.read_text(encoding="utf-8-sig")))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
