"""Validate one conversational follow-up from stable, already-verified evidence.

This is a synthesis-only test of cross-turn evidence reuse. Parent answer text is
never treated as evidence; selected parent evidence keeps its source URI, page,
verification record and explicit lineage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.intelligence import enforce_requested_answer_contract  # noqa: E402
from paperpilot_agentic.paid_synthesis import (  # noqa: E402
    SingleUpstreamAttemptLLM,
    synthesize_verified_replay_once,
)
from paperpilot_agentic.providers import OpenAICompatibleLLM  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-replay", required=True)
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--evidence-id", action="append", required=True)
    parser.add_argument("--output-dir", default=".paperpilot/demo/follow-up-reuse")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--postprocess-existing")
    args = parser.parse_args()

    source_path = Path(args.parent_replay).expanduser().resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    source_by_id = {
        str(item.get("evidence_id") or ""): item
        for item in list(source.get("evidence") or [])
    }
    claim_id = "claim_followup_mechanism_facts"
    evidence: list[dict[str, Any]] = []
    for source_id in args.evidence_id:
        if source_id not in source_by_id:
            raise ValueError(f"Unknown parent evidence ID: {source_id}")
        item = copy.deepcopy(source_by_id[source_id])
        provenance = dict(item.get("provenance") or {})
        verification = dict(provenance.get("verification") or {})
        if not verification or verification.get("degraded"):
            raise ValueError(f"Parent evidence is not closure-grade: {source_id}")
        digest = hashlib.sha256(
            f"{args.parent_run_id}\x1f{source_id}\x1f{claim_id}".encode("utf-8")
        ).hexdigest()
        item["evidence_id"] = f"evidence_turn_reuse_{digest[:20]}"
        item["claim_id"] = claim_id
        item["claim_ids"] = [claim_id]
        item["claim_polarities"] = {claim_id: "SUPPORTS"}
        item["polarity"] = "SUPPORTS"
        provenance["cross_turn_reuse"] = True
        provenance["parent_run_id"] = args.parent_run_id
        provenance["derived_from_evidence_id"] = source_id
        item["provenance"] = provenance
        evidence.append(item)

    replay = {
        "schema_version": 1,
        "kind": "zero_network_paid_evidence_replay",
        "replay_research_outcome": "ANSWERED",
        "source_scope_violations": 0,
        "network_calls": 0,
        "paid_requests": 0,
        "question": args.question,
        "claims": [
            {
                "claim_id": claim_id,
                "statement": (
                    "FlowMamba states that MFA aligns bi-temporal features with optical-flow "
                    "displacement compensation, while ChangeMamba states that Sequential, Cross "
                    "and Parallel mechanisms arrange or concatenate bi-temporal tokens for "
                    "spatiotemporal relationship modeling."
                ),
                "status": "SUPPORTED",
                "confidence": min(float(item.get("confidence") or 0.0) for item in evidence),
                "gaps": [],
                "required_evidence_types": ["TEXT"],
                "metadata": {
                    "cross_turn_reuse": True,
                    "parent_run_id": args.parent_run_id,
                    "evidence_reuse_policy": "verified-anchor-lineage-v1",
                    "paper_sha256s": sorted(
                        {
                            str((item.get("provenance") or {}).get("paper_sha256") or "")
                            for item in evidence
                            if str((item.get("provenance") or {}).get("paper_sha256") or "")
                        }
                    ),
                },
            }
        ],
        "evidence": evidence,
        "evidence_boundary": (
            "Same-conversation reuse of previously verified, source-anchored evidence; "
            "parent answer text is not evidence."
        ),
        "source_snapshot_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    _write_json(output_dir / "followup_replay_private.json", replay)

    if args.postprocess_existing:
        existing = json.loads(
            Path(args.postprocess_existing).expanduser().resolve().read_text(encoding="utf-8")
        )
        existing["answer"] = enforce_requested_answer_contract(
            args.question,
            dict(existing.get("answer") or {}),
        )
        _write_json(output_dir / "followup_synthesis_contract_enforced.json", existing)
        print(
            json.dumps(
                {
                    "network_calls": 0,
                    "paid_requests": 0,
                    "contract_additions": existing["answer"].get(
                        "deterministic_contract_additions", []
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    settings = replace(
        RuntimeSettings.from_env(require_api_key=True),
        timeout_seconds=max(30.0, args.timeout_seconds),
    )
    if settings.provider != "qwen":
        raise ValueError("Follow-up validation is restricted to Qwen")
    model = SingleUpstreamAttemptLLM(
        OpenAICompatibleLLM(settings, max_retries=0),
        force_thinking_enabled=False,
    )
    try:
        result = synthesize_verified_replay_once(replay, model)
    except Exception as exc:
        response = model.last_response
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "upstream_attempts": model.upstream_attempts,
            "raw_response": getattr(response, "text", None),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "followup_failure_private.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2

    response = model.last_response
    result["provider_audit"] = {
        "model": model.model,
        "upstream_attempts": model.upstream_attempts,
        "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(response, "reasoning_tokens", 0) or 0),
        "finish_reason": getattr(response, "finish_reason", None),
        "request_id": getattr(response, "request_id", None),
    }
    _write_json(output_dir / "followup_synthesis_private.json", result)
    print(
        json.dumps(
            {
                "answer_mode": result["answer"]["mode"],
                "citation_count": len(result["citation_ids"]),
                "upstream_attempts": model.upstream_attempts,
                "provider_audit": result["provider_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
