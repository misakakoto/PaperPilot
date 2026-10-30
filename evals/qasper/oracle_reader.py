"""QASPER dev oracle-evidence Reader smoke with an explicit paid-request cap."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.providers import OpenAICompatibleLLM  # noqa: E402


def answer_type(qa: Mapping[str, Any]) -> str:
    annotations = list(qa.get("answers") or [])
    if not annotations:
        return "unknown"
    answer = dict(annotations[0].get("answer") or {})
    if answer.get("unanswerable"):
        return "unanswerable"
    if answer.get("extractive_spans"):
        return "extractive"
    if answer.get("free_form_answer"):
        return "abstractive"
    if answer.get("yes_no") is not None:
        return "boolean"
    return "unknown"


def oracle_evidence(qa: Mapping[str, Any]) -> list[str]:
    """Return de-duplicated annotated evidence without exposing gold answers."""

    result: list[str] = []
    for annotation in qa.get("answers") or []:
        for text in (annotation.get("answer") or {}).get("evidence") or []:
            normalized = str(text).strip()
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def build_cases(dataset: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    """Select a deterministic, approximately type-balanced dev subset."""

    buckets: defaultdict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for paper_id, paper in dataset.items():
        for qa in paper.get("qas") or []:
            row = {
                "paper_id": str(paper_id),
                "question_id": str(qa.get("question_id") or ""),
                "question": str(qa.get("question") or ""),
                "answer_type": answer_type(qa),
                "evidence": oracle_evidence(qa),
            }
            if row["question_id"] and row["question"]:
                buckets[row["answer_type"]].append(row)
    order = ["extractive", "abstractive", "boolean", "unanswerable", "unknown"]
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets.values()):
        for name in order:
            if buckets[name] and len(selected) < limit:
                selected.append(buckets[name].popleft())
    return selected


def messages(case: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = {"question": case["question"], "evidence": case["evidence"]}
    return [
        {
            "role": "system",
            "content": (
                "Answer the scientific-paper question in concise English using only the supplied "
                "oracle evidence. Return only the answer text. For a yes/no question return Yes or "
                "No. If the evidence cannot answer the question, return exactly Unanswerable. "
                "Do not cite IDs, explain the task, or use outside knowledge."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset.name != "qasper-dev-v0.3.json":
        raise ValueError("Development smoke only accepts qasper-dev-v0.3.json")
    if not 1 <= args.limit <= 100:
        raise ValueError("--limit must be between 1 and 100")
    if args.execute_paid and args.max_requests != args.limit:
        raise ValueError("Paid execution requires --max-requests to equal --limit")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output path")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    cases = build_cases(dataset, limit=args.limit)
    report: dict[str, Any] = {
        "kind": "qasper_v0.3_oracle_evidence_reader_dev_smoke",
        "dataset_split": "dev",
        "dataset_sha256": fingerprint(args.dataset),
        "limit": args.limit,
        "selection": "deterministic_type_round_robin_v1",
        "uses_gold_answers_in_prompt": False,
        "uses_gold_evidence_in_prompt": True,
        "retrieval_evaluated": False,
        "official_score_comparable": False,
        "limitations": [
            "Empty oracle evidence can reveal unanswerability; not an abstention benchmark.",
            "Evidence may contain only a figure/table caption, not its image.",
        ],
        "provider_attempts": 0,
        "cases": [{**case, "messages": messages(case), "prediction": None} for case in cases],
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    if not args.execute_paid:
        print(json.dumps({"output": str(args.output), "paid_requests": 0}))
        return 0
    from dotenv import load_dotenv

    if args.env_file:
        load_dotenv(args.env_file, override=False)
    settings = RuntimeSettings.from_env(require_api_key=True)
    model = OpenAICompatibleLLM(settings, max_retries=0)
    report["model"] = settings.model
    for case in report["cases"]:
        started = time.perf_counter()

        def before_attempt() -> None:
            if report["provider_attempts"] >= args.max_requests:
                raise RuntimeError("Paid request cap reached")
            report["provider_attempts"] += 1
            save()

        try:
            response = model.complete(
                case["messages"], json_mode=False, thinking_enabled=False,
                max_tokens=300, before_attempt=before_attempt,
            )
        except Exception as exc:
            # Never persist provider exception text: it may include request data or credentials.
            case["error"] = type(exc).__name__
            case["elapsed_seconds"] = time.perf_counter() - started
            report["stopped_reason"] = "provider_error"
            save()
            return 1
        case["prediction"] = str(response.text or "").strip()
        case["usage"] = dataclasses.asdict(response)
        case["elapsed_seconds"] = time.perf_counter() - started
        save()
        if not case["prediction"] or response.finish_reason != "stop":
            report["stopped_reason"] = "empty_or_incomplete_output"
            save()
            return 1
    print(json.dumps({"output": str(args.output), "paid_requests": report["provider_attempts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
