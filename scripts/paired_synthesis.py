"""Freeze matched evidence for direct Markdown versus structured synthesis.

Dry-run captures the production synthesizer prompt without calling any provider.
Paid mode executes one request per arm, persists each response, and stops on error.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paperpilot_agentic.direct_answer import build_messages  # noqa: E402
from paperpilot_agentic.intelligence import LLMAnswerSynthesizer  # noqa: E402


class PromptCaptured(Exception):
    pass


class CaptureModel:
    def complete(self, messages, *, json_mode=False, max_tokens=None, thinking_enabled=None):
        self.messages = messages
        raise PromptCaptured()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare_pair(snapshot):
    question = str(snapshot["run"]["question"])
    capture = CaptureModel()
    try:
        LLMAnswerSynthesizer(capture).synthesize(
            question=question,
            claims=snapshot["claim_graph"]["claims"], evidence=snapshot["evidence"],
        )
    except PromptCaptured:
        pass
    else:
        raise ValueError("No synthesis prompt produced; usable evidence is required")
    structured = capture.messages
    # Use precisely the selected excerpts and source metadata sent by production.
    # No manual excerpt selection, translation, hints, or answer insertion.
    evidence = json.loads(structured[-1]["content"])["evidence"]
    direct = build_messages(question, evidence)
    assert json.loads(direct[-1]["content"])["evidence"] == evidence
    return {
        "question": question,
        "source_run_id": snapshot["run"]["run_id"],
        "evidence_sha256": fingerprint(evidence),
        "evidence_count": len(evidence),
        "shared_settings": {"thinking_enabled": False, "max_tokens": 8000},
        "arms": {
            "direct": {"messages": direct, "json_mode": False},
            "structured": {"messages": structured, "json_mode": True},
        },
        "scope": "Synthesis only. Structured arm retains Claim instructions; retrieval is frozen.",
        "review_dimensions": ["mechanism correctness", "numeric correctness",
                              "citation support", "question coverage", "readability", "limitations"],
        "semantic_review": "NOT_RUN",
        "results": {},
        "provider_attempts": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".paperpilot/paired-synthesis.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output path to preserve previous comparisons")
    report = prepare_pair(json.loads(args.snapshot.read_text(encoding="utf-8-sig")))

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    if args.execute_paid:
        from dotenv import load_dotenv
        from paperpilot_agentic.config import RuntimeSettings
        from paperpilot_agentic.providers import OpenAICompatibleLLM
        if args.env_file:
            load_dotenv(args.env_file, override=False)
        settings = RuntimeSettings.from_env(require_api_key=True)
        report["model"] = settings.model
        report["temperature"] = settings.temperature
        model = OpenAICompatibleLLM(settings, max_retries=0)
        for name, arm in report["arms"].items():
            started = time.perf_counter()
            def before_attempt():
                if report["provider_attempts"] >= 2:
                    raise RuntimeError("Two-request cap reached")
                report["provider_attempts"] += 1
                save()
            try:
                response = model.complete(**arm, **report["shared_settings"], before_attempt=before_attempt)
                report["results"][name] = {
                    **dataclasses.asdict(response), "elapsed_seconds": time.perf_counter() - started,
                }
                save()
                if not response.text.strip() or response.finish_reason != "stop":
                    report["stopped_reason"] = "empty_or_incomplete_output"
                    save()
                    return 1
            except Exception as error:
                report["stopped_reason"] = type(error).__name__
                save()
                return 1
    print(json.dumps({"report": str(args.output), "provider_attempts": report["provider_attempts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
