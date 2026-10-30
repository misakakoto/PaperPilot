"""Compile task views against cumulative spending and the serving window."""

import copy
import json
from dataclasses import dataclass

from .context_window import prompt_upper_bound
from .llm_runtime import PersistentLLMRuntime, LLMBudgetExhausted
from .prototype_workspace import digest
from .runtime_profile import enabled, feature


class ContextCompilationBlocked(LLMBudgetExhausted):
    request_dispatched = False


@dataclass
class ContextFrame:
    context_key: str
    messages: list
    output_tokens: int
    manifest: dict


def _project(payload, level, *, preserve_feedback=False, artifacts=None):
    p = copy.deepcopy(payload)
    changes = []
    active = p.get("active_task_id")
    if active and active in p.get("tasks", {}):
        p["tasks"] = {active: p["tasks"][active]}
        changes.append("task definitions: current task only; full plan retained")
    table = p.get("context_artifacts", {})
    if "artifact_index" in p:
        p["artifact_index"] = {i: row for i, row in p["artifact_index"].items() if i in table}
        changes.append("index: direct dependencies only; original artifacts retained")
    for key in ("feedback", "implementation_feedback"):
        if isinstance(p.get(key), dict):
            for name in ("previous_response", "rejected_candidate"):
                if name in p[key]:
                    raw = p[key][name]
                    if preserve_feedback:
                        identity = next((i for i, a in (artifacts or {}).items()
                                         if a.get("content") == raw and digest(a) == i), None)
                        if identity is None:
                            changes.append(key + "." + name + ": retained; no readable immutable artifact")
                            continue
                        p[key].pop(name)
                        p[key][name + "_ref"] = {
                            "sha256": digest(raw), "artifact_id": identity,
                            "read_action": "inspect_artifact",
                        }
                        changes.append(key + "." + name + ": exact immutable artifact reference")
                        continue
                    raw = p[key].pop(name)
                    p[key][name + "_ref"] = {
                        "sha256": digest(raw),
                        "location": "saved turn receipt",
                    }
                    changes.append(
                        key + "." + name + ": duplicate response replaced by receipt hash"
                    )
    for identity, artifact in table.items():
        content = artifact.get("content", {})
        kind = artifact.get("kind")
        if kind in {"analyze_literature", "design"}:
            for field in ("findings", "literature_basis"):
                for finding in content.get(field, []):
                    for duplicate in ("quote", "source"):
                        if duplicate in finding:
                            finding.pop(duplicate)
                            changes.append(
                                identity
                                + ": repeated quote/provenance retained in original artifact"
                            )
            content["raw_artifact_id"] = identity
        if kind == "review_literature" and content.get("passed") is True and "checks" in content:
            content.pop("checks")
            content["raw_artifact_id"] = identity
            changes.append(
                identity + ": individual checks retained in original; coverage/issues retained"
            )
    if level:
        from .prototype_selected_text import bound_view

        papers = [
            (i, a)
            for i, a in table.items()
            if a.get("kind") == "read_literature"
            and isinstance(a.get("content", {}).get("pages"), list)
        ]
        allowance = max(1024, (16000, 8000, 4000)[level - 1] // max(1, len(papers)))
        for identity, artifact in papers:
            before = sum(len(page.get("quotes", [])) for page in artifact["content"]["pages"])
            bounded = bound_view(artifact, max_bytes=allowance)
            after = sum(len(page.get("quotes", [])) for page in bounded["content"].get("pages", []))
            if not before or after:
                artifact["content"] = bounded["content"]
                changes.append(
                    identity
                    + ": bounded whole-quote view; original and coverage omissions retained"
                )
    return p, list(dict.fromkeys(changes))


class ContextCompiler:
    def compile(
        self,
        owner,
        run_id,
        state,
        snapshot,
        *,
        protected_tokens=0,
        protected_calls=0,
        window_tokens=131072,
    ):
        enabled(state)
        manifest = {
            "version": (state.get("runtime_profile") or {}).get("config", {}).get("context_compiler", "task-budget-v1"),
            "profile": state.get("runtime_profile"),
            "status": "BLOCKED",
            "remaining_tokens": snapshot.get("remaining_tokens"),
            "unknown_reserved_tokens": snapshot.get("reserved_unknown_tokens", 0),
            "protected_tokens": protected_tokens,
            "protected_calls": protected_calls,
            "projections": [],
            "missing_tokens": 0,
        }
        empty = ContextFrame("blocked", [], 0, manifest)
        if snapshot.get("unknown_results"):
            manifest["reason"] = "UNKNOWN_EXTERNAL_RESULT"
            return empty
        if (
            snapshot.get("remaining_llm_calls") is not None
            and snapshot["remaining_llm_calls"] <= protected_calls
        ):
            manifest["reason"] = "CALL_BUDGET_EXHAUSTED"
            return empty
        key, messages = owner._context(run_id, state)
        payload_index = next(i for i, m in enumerate(messages) if m["role"] == "user")
        payload = json.loads(messages[payload_index]["content"])
        if 'evidence_sources' in payload:
            manifest['evidence_sources'] = copy.deepcopy(payload['evidence_sources'])
        invalid = [
            identity
            for identity in payload.get("context_artifacts", {})
            if identity in state.get("work", {}).get("artifacts", {})
            and digest(state["work"]["artifacts"][identity]) != identity
        ]
        if invalid:
            manifest.update(reason="INPUT_VERSION_CHANGED", invalid_artifact_ids=invalid)
            return empty
        history_count = len(messages) - payload_index - 1
        messages = copy.deepcopy(messages[: payload_index + 1])
        task = payload.get("active_task_id") or state.get("role", "lead")
        output = (
            owner._decision_output_tokens(state)
            if hasattr(owner, "_decision_output_tokens")
            else 5000
        )
        if payload.get("candidate_patch"):
            output = 3000
        truncations = state.get("output_truncations", {}).get(task, 0)
        if truncations >= 2:
            manifest.update(reason="REPEATED_OUTPUT_TRUNCATION", task_id=task)
            return empty
        if truncations:
            output = 5000
        remaining = snapshot.get("remaining_tokens")
        capacity = min(
            window_tokens - 1024,
            float("inf") if remaining is None else remaining - protected_tokens,
        )
        manifest.update(
            task_id=task,
            output_tokens=output,
            artifact_versions=sorted(payload.get("context_artifacts", {})),
            original_prompt_upper_bound=prompt_upper_bound(
                PersistentLLMRuntime._with_budget(messages, snapshot)
            ),
            mandatory_fields=[
                k
                for k in (
                    "goal",
                    "constraints",
                    "validation_contract",
                    "active_task_id",
                    "required_tensor_interfaces",
                )
                if k in payload
            ],
        )
        for level in range(4):
            try:
                view, changes = _project(payload, level,
                    preserve_feedback=feature(state, "retain_unstored_feedback"),
                    artifacts=state.get("work", {}).get("artifacts", {}))
                if history_count:
                    changes.append(
                        "conversation history omitted; current observations, analysis and last feedback retained"
                    )
            except ValueError:
                continue
            prepared = copy.deepcopy(messages)
            prepared[-1]["content"] = json.dumps(view, ensure_ascii=False, sort_keys=True)
            upper = prompt_upper_bound(PersistentLLMRuntime._with_budget(prepared, snapshot))
            manifest["artifact_views"] = {
                i: {"kind": a.get("kind"), "view_sha256": digest(a), "raw_artifact_id": i}
                for i, a in view.get("context_artifacts", {}).items()
            }
            manifest.update(
                projections=changes,
                input_upper_bound=upper,
                missing_tokens=max(0, upper + output - capacity),
                request_sha256=digest(prepared),
                view_level=level,
                read_originals="inspect_artifact by ID; inspect_code by exact symbol or line",
            )
            if upper + output <= capacity:
                manifest.update(status="READY", reason=None)
                return ContextFrame(key, prepared, output, manifest)
        manifest["reason"] = "REQUIRED_CONTEXT_EXCEEDS_BUDGET"
        return ContextFrame(key, [], output, manifest)
