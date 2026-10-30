"""Run the frozen interview demo with an explicit provider-attempt ceiling.

Dry-run is the default. Paid execution requires both ``--execute-paid`` and an
exact acknowledgement string so this helper cannot consume API quota by accident.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# Direct script execution otherwise resolves scripts/paperpilot_agentic.py
# before the actual package. Prefer this checkout's source tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperpilot_agentic.context_engine import audit_final_answer_quality

ACKNOWLEDGEMENT = "I_APPROVE_QWEN_PAID_DEMO"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXHAUSTED"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from PaperPilot: {detail[:2_000]}") from exc
    if not isinstance(value, dict):
        raise TypeError("PaperPilot returned a non-object JSON response")
    return value


def _request_optional_run(base_url: str, run_id: str) -> dict[str, Any] | None:
    """Return a historical run when it exists; a missing cache is not a suite error."""

    try:
        return _request_json(
            f"{base_url}/api/agentic/runs/{run_id}?include_trace=false",
            timeout=30,
        )
    except RuntimeError as exc:
        if "HTTP 404 from PaperPilot:" in str(exc):
            return None
        raise


def _wait_run(base_url: str, run_id: str, *, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        value = _request_json(
            f"{base_url}/api/agentic/runs/{run_id}?include_trace=false",
            timeout=30,
        )
        if str((value.get("run") or {}).get("status") or "") in TERMINAL:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"PaperPilot run did not become terminal: {run_id}")
        time.sleep(2)


def _run_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    run = dict(snapshot.get("run") or {})
    answer = dict(snapshot.get("final_answer") or {})
    claims = list((snapshot.get("claim_graph") or {}).get("claims") or [])
    if claims and all(str(item.get("claim_id") or "") for item in claims):
        quality = audit_final_answer_quality(
            answer,
            claims=claims,
            evidence=list(snapshot.get("evidence") or []),
            question=str(run.get("question") or ""),
        )
    else:
        # Synthetic/unit snapshots may intentionally omit durable identifiers.
        # Real PaperPilot snapshots always take the recomputation path above.
        quality = dict(answer.get("answer_quality") or {})
    llm_calls = list(snapshot.get("llm_calls") or [])
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "research_outcome": run.get("research_outcome"),
        "used_llm_calls": int(run.get("used_llm_calls") or 0),
        "used_tool_calls": int(run.get("used_tool_calls") or 0),
        "used_tokens": int(run.get("used_tokens") or 0),
        "completed_provider_responses": sum(
            str(item.get("status") or "") == "COMPLETED" for item in llm_calls
        ),
        "failed_provider_attempts": sum(
            max(1, int(item.get("provider_attempts") or 1))
            for item in llm_calls
            if str(item.get("status") or "") == "FAILED"
        ),
        "claim_statuses": {
            str(item.get("claim_id")): str(item.get("status")) for item in claims
        },
        "answer_mode": answer.get("mode"),
        "answer_quality_passed": quality.get("passed"),
        "answer_quality_issues": list(quality.get("issues") or []),
        "stored_answer_quality_passed": (answer.get("answer_quality") or {}).get(
            "passed"
        ),
        "citation_count": len(answer.get("citation_ids") or []),
        "answer": answer.get("answer"),
    }


def _scenario_issues(
    snapshot: dict[str, Any],
    scenario: dict[str, Any],
) -> list[str]:
    summary = _run_summary(snapshot)
    answer = str(summary.get("answer") or "")
    folded_answer = answer.casefold()
    checks = dict(scenario.get("answer_checks") or {})
    issues = [
        f"missing_required_substring:{value}"
        for value in checks.get("required_substrings") or []
        if str(value).casefold() not in folded_answer
    ]
    required_any_groups = list(checks.get("required_any_substring_groups") or [])
    for group in required_any_groups:
        if group and not any(
            str(value).casefold() in folded_answer for value in group
        ):
            issues.append(
                "missing_required_alternative:"
                + "|".join(str(value) for value in group)
            )
    required_one_of_groups = list(
        checks.get("required_one_of_substring_groups") or []
    )
    if required_one_of_groups and not any(
        group
        and all(str(value).casefold() in folded_answer for value in group)
        for group in required_one_of_groups
    ):
        rendered = "|".join(
            "+".join(str(value) for value in group)
            for group in required_one_of_groups
        )
        issues.append(f"missing_required_substring_group:{rendered}")
    issues.extend(
        f"forbidden_substring:{value}"
        for value in checks.get("forbidden_substrings") or []
        if str(value) and str(value).casefold() in folded_answer
    )
    citations = {
        str(value) for value in (snapshot.get("final_answer") or {}).get("citation_ids") or []
    }
    minimum_citations = int(checks.get("minimum_citations") or 0)
    if len(citations) < minimum_citations:
        issues.append(f"too_few_citations:{len(citations)}<{minimum_citations}")

    supported_claims = sum(
        str(item.get("status") or "") == "SUPPORTED"
        for item in (snapshot.get("claim_graph") or {}).get("claims") or []
    )
    minimum_supported = int(checks.get("minimum_supported_claims") or 0)
    if supported_claims < minimum_supported:
        issues.append(f"too_few_supported_claims:{supported_claims}<{minimum_supported}")

    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in snapshot.get("evidence") or []
        if item.get("evidence_id")
    }
    missing_citation_records = sorted(citations - set(evidence_by_id))
    issues.extend(
        f"citation_record_missing:{evidence_id}" for evidence_id in missing_citation_records
    )
    cited_source_shas: set[str] = set()
    for evidence_id in citations & set(evidence_by_id):
        evidence = evidence_by_id[evidence_id]
        provenance = dict(evidence.get("provenance") or {})
        document_sha = str(provenance.get("document_sha256") or "").casefold()
        if not document_sha:
            source_uri = str(evidence.get("source_uri") or "")
            if source_uri.startswith("paper://"):
                document_sha = source_uri.removeprefix("paper://").split("/", 1)[0].casefold()
        if document_sha:
            cited_source_shas.add(document_sha)
    required_sources = {
        str(value).casefold()
        for value in checks.get("required_citation_source_sha256s") or []
    }
    allowed_sources = {
        str(value).casefold()
        for value in checks.get("allowed_citation_source_sha256s") or []
    }
    issues.extend(
        f"required_citation_source_missing:{value}"
        for value in sorted(required_sources - cited_source_shas)
    )
    issues.extend(
        f"citation_source_out_of_scope:{value}"
        for value in sorted(cited_source_shas - allowed_sources)
        if allowed_sources
    )

    if checks.get("require_follow_up_lineage"):
        run = dict(snapshot.get("run") or {})
        metadata = dict(run.get("metadata") or {})
        if not str(metadata.get("parent_run_id") or run.get("parent_run_id") or ""):
            issues.append("follow_up_parent_run_missing")
        if not str(metadata.get("conversation_id") or run.get("conversation_id") or ""):
            issues.append("follow_up_conversation_missing")
    return issues


def _passed(snapshot: dict[str, Any], scenario: dict[str, Any]) -> bool:
    summary = _run_summary(snapshot)
    if not (
        summary["status"] == "COMPLETED"
        and summary["research_outcome"] == "ANSWERED"
        and summary["answer_quality_passed"] is True
    ):
        return False
    return not _scenario_issues(snapshot, scenario)


def _same_question(left: str, right: str) -> bool:
    return "".join(str(left).split()).casefold() == "".join(str(right).split()).casefold()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _four_call_follow_up_budget(timeout_seconds: float) -> dict[str, Any]:
    """Allocate the fixed follow-up cap to plan + two verifications + synthesis.

    Reserving a repair call inside a four-call cap leaves room for only one
    paper-level verifier after planning.  The interview follow-up explicitly
    requires fresh evidence from two papers, so its first-pass synthesis must
    be schema-safe and the deterministic quality gate remains the fallback.
    """

    return {
        "max_steps": 16,
        "max_tool_calls": 10,
        "max_llm_calls": 4,
        "max_tokens": 60000,
        "max_seconds": timeout_seconds,
        "synthesis_reserve_tokens": 10000,
        "repair_reserve_tokens": 0,
        "synthesis_reserve_calls": 1,
        "repair_reserve_calls": 0,
        "max_repair_calls": 0,
    }


def _validate_retrieval_preflight(
    result: dict[str, Any],
    *,
    scenario_path: Path,
) -> None:
    """Fail before paid execution unless the current local corpus passes retrieval."""

    if result.get("kind") != "zero_network_interview_demo_retrieval_preflight":
        raise RuntimeError("Interview preflight returned the wrong result kind")
    if Path(str(result.get("scenario_file") or "")).resolve() != scenario_path.resolve():
        raise RuntimeError("Interview preflight used a different scenario file")
    if result.get("network_calls") != 0 or result.get("paid_requests") != 0:
        raise RuntimeError("Interview preflight was not zero-network and zero-paid")
    if result.get("passed") is not True:
        raise RuntimeError("Interview retrieval preflight failed")
    dense_index = result.get("dense_index")
    if not isinstance(dense_index, dict) or dense_index.get("configured") is not True:
        raise RuntimeError("Interview retrieval preflight did not use dense retrieval")
    if dense_index.get("complete") is not True:
        raise RuntimeError("Interview dense index is incomplete for the current model identity")
    checks = list(result.get("checks") or [])
    if len(checks) != int(result.get("probe_count") or 0) or not checks:
        raise RuntimeError("Interview retrieval preflight has incomplete checks")
    if any(item.get("passed") is not True for item in checks):
        raise RuntimeError("Interview retrieval preflight contains a failed probe")


def _validate_provider_connectivity(result: dict[str, Any]) -> None:
    if result.get("probe") != "tcp_connect_v1":
        raise RuntimeError("PaperPilot returned an unknown provider connectivity probe")
    if int(result.get("billable_requests") or 0) != 0:
        raise RuntimeError("Provider connectivity preflight unexpectedly used a billable request")
    if result.get("reachable") is not True:
        raise RuntimeError(
            "PaperPilot service process cannot reach the configured model provider "
            f"({result.get('error_type') or 'unknown network error'}); no Agent run was created"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="demo/interview_scenarios.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--single-run-id", default="run_f540166d6c73463cb7e855c17727efe7")
    parser.add_argument("--max-paid-attempts", type=int, default=16)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--acknowledge", default="")
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not 1 <= args.max_paid_attempts <= 20:
        raise ValueError("--max-paid-attempts must be between 1 and 20")
    scenario_path = Path(args.scenarios).expanduser().resolve()
    scenarios_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    by_kind = {
        str(item["kind"]): item for item in scenarios_payload.get("scenarios", [])
    }
    required_kinds = {"single_paper", "multi_paper", "follow_up"}
    if set(by_kind) != required_kinds:
        raise ValueError(f"Scenario file must contain exactly {sorted(required_kinds)}")

    plan = {
        "kind": "paperpilot_interview_demo_execution_plan",
        "provider": "qwen3-vl-flash-2025-10-15",
        "max_paid_attempts": args.max_paid_attempts,
        "existing_single_run_id": args.single_run_id,
        "execution_order": [
            "run_zero_network_retrieval_preflight",
            "reuse_existing_single_only_if_question_and_answer_checks_match_otherwise_run_it",
            "run_multi_paper_question_with_max_6_llm_attempts",
            "only_if_multi_passes_run_follow_up_with_remaining_max_4_attempts",
        ],
        "stop_on_failure": True,
        "questions": {
            kind: str(by_kind[kind]["question"])
            for kind in ("single_paper", "multi_paper", "follow_up")
        },
    }
    if not args.execute_paid:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.acknowledge != ACKNOWLEDGEMENT:
        raise PermissionError(
            f"Paid execution requires --acknowledge {ACKNOWLEDGEMENT}"
        )

    base_url = args.base_url.rstrip("/")
    bootstrap = _request_json(f"{base_url}/api/bootstrap", timeout=15)
    if (bootstrap.get("provider") != "qwen" or
            bootstrap.get("model") != "qwen3-vl-flash-2025-10-15"):
        raise RuntimeError("PaperPilot is not running with qwen3-vl-flash-2025-10-15")
    provider_connectivity = _request_json(
        f"{base_url}/api/provider-connectivity",
        timeout=15,
    )
    _validate_provider_connectivity(provider_connectivity)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(
        args.output_dir or f".paperpilot/demo/interview-suite-{timestamp}"
    ).expanduser().resolve()
    preflight_path = output_dir / "retrieval_preflight.json"
    project_root = Path(__file__).resolve().parents[1]
    preflight_process = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "check_demo_retrieval.py"),
            "--scenarios",
            str(scenario_path),
            "--output",
            str(preflight_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    if preflight_process.returncode != 0:
        raise RuntimeError(
            "Zero-network interview retrieval preflight failed before paid execution: "
            f"{preflight_process.stderr[-2_000:]}"
        )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    _validate_retrieval_preflight(preflight, scenario_path=scenario_path)
    report: dict[str, Any] = {
        **plan,
        "started_at": datetime.now().astimezone().isoformat(),
        "source_root": bootstrap.get("source_root"),
        "provider_connectivity": provider_connectivity,
        "retrieval_preflight": {
            "passed": True,
            "probe_count": preflight["probe_count"],
            "paper_sha256s": [
                str(item.get("paper_sha256") or "") for item in preflight["checks"]
            ],
            "network_calls": 0,
            "paid_requests": 0,
            "dense_index": preflight["dense_index"],
        },
        "runs": {},
        "provider_attempts_used": 0,
        "passed": False,
    }

    single_scenario = by_kind["single_paper"]
    single = _request_optional_run(base_url, args.single_run_id)
    existing_question_matches = single is not None and _same_question(
        str((single.get("run") or {}).get("question") or ""),
        str(single_scenario["question"]),
    )
    existing_single_passes = bool(
        single is not None
        and existing_question_matches
        and _passed(single, single_scenario)
    )
    report["existing_single_reuse"] = {
        "run_id": args.single_run_id,
        "found": single is not None,
        "question_matches": existing_question_matches,
        "scenario_issues": (
            _scenario_issues(single, single_scenario)
            if single is not None
            else ["existing_run_not_found"]
        ),
        "reused": existing_single_passes,
    }
    if not existing_single_passes:
        if args.max_paid_attempts < 4:
            raise ValueError("A fresh single-paper scenario requires at least 4 attempts")
        single_budget = min(6, args.max_paid_attempts)
        single_queued = _request_json(
            f"{base_url}/api/agentic/runs",
            method="POST",
            payload={
                "question": single_scenario["question"],
                "budget": {
                    "max_steps": 18,
                    "max_tool_calls": 12,
                    "max_llm_calls": single_budget,
                    "max_tokens": 80000,
                    "max_seconds": args.timeout_seconds,
                    "synthesis_reserve_tokens": 12000,
                    "repair_reserve_tokens": 6000,
                    "synthesis_reserve_calls": 1,
                    "repair_reserve_calls": 1,
                    "max_repair_calls": 1,
                },
                "metadata": {
                    "validation_kind": "interview_demo_clean_single",
                    "scenario_id": single_scenario["id"],
                    "provider_attempt_cap": single_budget,
                },
            },
            timeout=30,
        )
        single_run_id = str((single_queued.get("run") or {}).get("run_id") or "")
        if not single_run_id:
            raise RuntimeError("PaperPilot did not return a single-paper run_id")
        single = _wait_run(
            base_url,
            single_run_id,
            timeout_seconds=args.timeout_seconds,
        )
    _write(output_dir / "single_snapshot_private.json", single)
    single_summary = _run_summary(single)
    single_summary["scenario_issues"] = _scenario_issues(single, single_scenario)
    report["runs"]["single_paper"] = single_summary
    if not existing_single_passes:
        report["provider_attempts_used"] += single_summary["used_llm_calls"]
    if not _passed(single, single_scenario):
        report["stop_reason"] = "single_paper_run_failed_gate"
        _write(output_dir / "suite_result_private.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    remaining = args.max_paid_attempts - int(report["provider_attempts_used"])
    if remaining < 4:
        report["stop_reason"] = "fewer_than_4_provider_attempts_left_for_multi_paper"
        _write(output_dir / "suite_result_private.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    multi_budget = min(6, remaining)
    multi_queued = _request_json(
        f"{base_url}/api/agentic/runs",
        method="POST",
        payload={
            "question": by_kind["multi_paper"]["question"],
            "budget": {
                "max_steps": 24,
                "max_tool_calls": 20,
                "max_llm_calls": multi_budget,
                "max_tokens": 100000,
                "max_seconds": args.timeout_seconds,
                "synthesis_reserve_tokens": 12000,
                "repair_reserve_tokens": 6000,
                "synthesis_reserve_calls": 1,
                "repair_reserve_calls": 1,
                "max_repair_calls": 1,
            },
            "metadata": {
                "validation_kind": "interview_demo_clean_multi",
                "scenario_id": by_kind["multi_paper"]["id"],
                "provider_attempt_cap": multi_budget,
            },
        },
        timeout=30,
    )
    multi_run_id = str((multi_queued.get("run") or {}).get("run_id") or "")
    if not multi_run_id:
        raise RuntimeError("PaperPilot did not return a multi-paper run_id")
    multi = _wait_run(base_url, multi_run_id, timeout_seconds=args.timeout_seconds)
    _write(output_dir / "multi_snapshot_private.json", multi)
    multi_summary = _run_summary(multi)
    multi_summary["scenario_issues"] = _scenario_issues(
        multi,
        by_kind["multi_paper"],
    )
    report["runs"]["multi_paper"] = multi_summary
    report["provider_attempts_used"] += multi_summary["used_llm_calls"]
    if not _passed(multi, by_kind["multi_paper"]):
        report["stop_reason"] = "multi_paper_run_failed_gate"
        _write(output_dir / "suite_result_private.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    remaining = args.max_paid_attempts - int(report["provider_attempts_used"])
    if remaining < 4:
        report["stop_reason"] = "fewer_than_4_provider_attempts_left_for_follow_up"
        _write(output_dir / "suite_result_private.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    follow_budget = 4
    follow_queued = _request_json(
        f"{base_url}/api/agentic/runs/{multi_run_id}/follow-up",
        method="POST",
        payload={
            "question": by_kind["follow_up"]["question"],
            "budget": _four_call_follow_up_budget(args.timeout_seconds),
            "metadata": {
                "validation_kind": "interview_demo_clean_follow_up",
                "scenario_id": by_kind["follow_up"]["id"],
                "provider_attempt_cap": follow_budget,
            },
        },
        timeout=30,
    )
    follow_run_id = str((follow_queued.get("run") or {}).get("run_id") or "")
    if not follow_run_id:
        raise RuntimeError("PaperPilot did not return a follow-up run_id")
    follow = _wait_run(base_url, follow_run_id, timeout_seconds=args.timeout_seconds)
    _write(output_dir / "followup_snapshot_private.json", follow)
    follow_summary = _run_summary(follow)
    report["runs"]["follow_up"] = follow_summary
    report["provider_attempts_used"] += follow_summary["used_llm_calls"]
    follow_summary["scenario_issues"] = _scenario_issues(
        follow,
        by_kind["follow_up"],
    )
    report["passed"] = _passed(follow, by_kind["follow_up"])
    report["stop_reason"] = None if report["passed"] else "follow_up_run_failed_gate"
    report["finished_at"] = datetime.now().astimezone().isoformat()
    _write(output_dir / "suite_result_private.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
