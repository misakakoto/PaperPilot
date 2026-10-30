"""Re-audit one persisted visual result and optionally synthesize it once.

The command writes private snapshots only below ``.paperpilot`` (gitignored).
It is an operational validation tool, not a replacement for a clean live run.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

# ``scripts/paperpilot_agentic.py`` intentionally shares the package name for
# backwards compatibility, so prefer the source tree when this file is run
# directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.evidence_replay import redacted_replay_report, replay_paid_evidence
from paperpilot_agentic.paid_synthesis import (
    SingleUpstreamAttemptLLM,
    synthesize_verified_replay_once,
)
from paperpilot_agentic.providers import OpenAICompatibleLLM


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_snapshot(base_url: str, run_id: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/agentic/runs/{run_id}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - local URL by default
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Run snapshot response is not a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence-id", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--prompt", default="Read the exact visible table.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--output-dir", default=".paperpilot/demo/paid-replay")
    parser.add_argument("--paid-synthesis", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    snapshot = _fetch_snapshot(args.base_url, args.run_id)
    snapshot_path = output_dir / "source_snapshot.json"
    _write_json(snapshot_path, snapshot)

    manifest = {
        "base_snapshot": str(snapshot_path),
        "imports": [
            {
                "kind": "snapshot_qwen_evidence",
                "snapshot": str(snapshot_path),
                "evidence_id": args.evidence_id,
                "target_claim_id": args.claim_id,
                "caption": args.caption,
                "prompt": args.prompt,
            }
        ],
    }
    replay = replay_paid_evidence(manifest)
    _write_json(output_dir / "replay_private.json", replay)
    _write_json(output_dir / "replay_redacted.json", redacted_replay_report(replay))

    summary: dict[str, Any] = {
        "replay_research_outcome": replay["replay_research_outcome"],
        "resolved_claims": sum(
            item.get("status") in {"SUPPORTED", "CONTRADICTED"}
            for item in replay["claims"]
        ),
        "claim_count": len(replay["claims"]),
        "network_calls": 0,
        "paid_requests": 0,
    }
    if args.paid_synthesis:
        settings = RuntimeSettings.from_env(require_api_key=True)
        settings = replace(settings, timeout_seconds=max(30.0, args.timeout_seconds))
        if settings.provider != "qwen":
            raise ValueError("Paid replay validation is currently restricted to Qwen")
        guarded_model = SingleUpstreamAttemptLLM(
            OpenAICompatibleLLM(settings, max_retries=0),
            # This stage is schema-constrained transformation of already-audited
            # evidence. Disabling extended reasoning materially reduces timeout
            # risk without weakening the evidence/citation gates below.
            force_thinking_enabled=False,
        )
        provider_audit: dict[str, Any] = {
            "model": guarded_model.model,
            "high_level_calls": guarded_model.high_level_calls,
            "upstream_attempts": guarded_model.upstream_attempts,
        }
        try:
            paid_result = synthesize_verified_replay_once(replay, guarded_model)
        except Exception as exc:
            response = guarded_model.last_response
            provider_audit.update(
                {
                    "high_level_calls": guarded_model.high_level_calls,
                    "upstream_attempts": guarded_model.upstream_attempts,
                    "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
                    "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
                    "reasoning_tokens": int(getattr(response, "reasoning_tokens", 0) or 0),
                    "finish_reason": getattr(response, "finish_reason", None),
                    "request_id": getattr(response, "request_id", None),
                }
            )
            failure = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "provider_audit": provider_audit,
                "raw_response": getattr(response, "text", None),
                "traceback": traceback.format_exc(),
            }
            _write_json(output_dir / "paid_synthesis_failure_private.json", failure)
            summary.update(
                {
                    "paid_requests": guarded_model.upstream_attempts,
                    "synthesis_error_type": type(exc).__name__,
                    "synthesis_error": str(exc),
                    "provider_audit": provider_audit,
                }
            )
            _write_json(output_dir / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2
        response = guarded_model.last_response
        provider_audit.update(
            {
                "high_level_calls": guarded_model.high_level_calls,
                "upstream_attempts": guarded_model.upstream_attempts,
                "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
                "cached_tokens": int(getattr(response, "cached_tokens", 0) or 0),
                "reasoning_tokens": int(getattr(response, "reasoning_tokens", 0) or 0),
                "finish_reason": getattr(response, "finish_reason", None),
                "request_id": getattr(response, "request_id", None),
            }
        )
        paid_result["provider_audit"] = provider_audit
        _write_json(output_dir / "paid_synthesis_private.json", paid_result)
        summary.update(
            {
                "paid_requests": guarded_model.upstream_attempts,
                "answer_mode": paid_result["answer"]["mode"],
                "citation_count": len(paid_result["citation_ids"]),
                "provider_audit": provider_audit,
            }
        )
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
