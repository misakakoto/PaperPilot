import copy
import pytest
from paperpilot_agentic.runtime_profile import PROFILE, attach
from paperpilot_agentic.run_recovery import bind_request, recover_response


def test_bound_response_ignores_unrelated_events_and_refuses_changed_inputs():
    state = {
        "goal": "g",
        "role": "lead",
        "turn": 0,
        "work": {"revision": 1},
        "model_protocol": "fixed",
    }
    attach(state, PROFILE)
    request = {"messages": [{"role": "user", "content": "x"}]}
    request["binding"] = bind_request("r", 1, state, request["messages"])
    events = [
        {
            "event_id": 1,
            "event_type": "llm.response.saved",
            "payload": {
                "stage": "elsewhere",
                "correlation_id": "wrong",
                "raw_response": {"text": "wrong"},
            },
        },
        {
            "event_id": 2,
            "event_type": "llm.response.saved",
            "payload": {
                "stage": "research.lead.1",
                "correlation_id": request["binding"]["correlation_id"],
                "raw_response": {"text": "right"},
            },
        },
    ]
    assert recover_response("r", state, request, events) == {"text": "right"}
    state["work"]["revision"] = 2
    assert recover_response("r", state, request, events) is None


def test_response_commit_crash_recovers_with_no_new_request(tmp_path, monkeypatch):
    from test_prototype_crash_recovery import setup
    from paperpilot_agentic import episode_runner

    rt, h, rid = setup(tmp_path, monkeypatch)
    state = h._load(rid)
    attach(state, PROFILE)
    h._save(rid, state)
    original = episode_runner.immutable

    def die(folder, name, data):
        if name.endswith("-response.json"):
            raise KeyboardInterrupt("after provider before local file")
        return original(folder, name, data)

    monkeypatch.setattr(episode_runner, "immutable", die)
    with pytest.raises(KeyboardInterrupt):
        h.advance(rid, max_turns=1)
    before = rt.db.get_run(rid)
    rt.db.emit_event(
        rid,
        "llm.response.saved",
        {"raw_response": {"text": "unrelated"}, "correlation_id": "other", "stage": "other"},
    )
    monkeypatch.setattr(episode_runner, "immutable", original)
    h.advance(rid, max_turns=1)
    assert h._load(rid)["turn"] == 1
    assert rt.db.get_run(rid)["used_llm_calls"] == before["used_llm_calls"] == 1
    assert h._load(rid)["recovery_trace"][0]["action"] == "APPLY_SAVED_RESPONSE"


def test_change_preview_is_read_only_and_commit_matches_preview(tmp_path, monkeypatch):
    from test_prototype_revisions import completed

    rt, h, rid = completed(tmp_path, monkeypatch)
    state = h._load(rid)
    attach(state, PROFILE)
    h._save(rid, state)
    before = h._load(rid)
    run = copy.deepcopy(rt.db.get_run(rid))
    calls = copy.deepcopy(rt.db.list_tool_calls(rid))
    value = {"input_cases": [[1, 48, 80]]}
    preview = h.preview_change(
        rid, kind="constraints", value=value, expected_revision=before["work"]["revision"]
    )
    assert (
        h._load(rid) == before and rt.db.get_run(rid) == run and rt.db.list_tool_calls(rid) == calls
    )
    impact = preview["change_impact"]
    assert set(impact["invalidated_task_ids"]) == {"check", "review"}
    assert before["work"]["tasks"]["code"]["output"] in impact["reusable_artifact_ids"]
    h.revise_constraints(rid, value, expected_revision=before["work"]["revision"])
    assert h._load(rid)["change_impact"] == impact
    for field in ("budget", "created_at", "used_llm_calls", "used_tokens"):
        assert rt.db.get_run(rid)[field] == run[field]


def test_diagnosis_adapter_uses_same_compiler(tmp_path):
    from test_imported_diagnosis_harness import setup

    rt, h, reader, binding = setup(tmp_path)
    result = h.start_imported_diagnosis(
        binding["bundle_id"], "runtime-v2-diagnosis", runtime_profile=PROFILE
    )
    rid = result["run"]["run_id"]
    h.advance(rid, max_turns=1)
    from paperpilot_agentic.research_diagnosis import DiagnosisHarness

    state = DiagnosisHarness(rt)._load(rid)
    assert state["context_manifest"]["status"] == "READY" and state["turn"] == 1


def test_truncation_is_retained_then_escalated_inside_same_budget(tmp_path, monkeypatch):
    from paperpilot_agentic.providers import LLMResponse
    from test_prototype_crash_recovery import setup

    rt, h, rid = setup(tmp_path, monkeypatch)
    state = h._load(rid)
    attach(state, PROFILE)
    h._save(rid, state)
    original = rt.llm_runtime.model.complete
    calls = []

    def truncated(messages, **kwargs):
        calls.append(kwargs["max_tokens"])
        if len(calls) == 1:
            return LLMResponse(
                text="{",
                model="prototype-offline-fixture",
                finish_reason="length",
                prompt_tokens=2,
                completion_tokens=2,
            )
        return original(messages, **kwargs)

    monkeypatch.setattr(rt.llm_runtime.model, "complete", truncated)
    h.advance(rid, max_turns=1)
    assert h._load(rid)["last_observation"]["code"] == "OUTPUT_TRUNCATED"
    h.advance(rid, max_turns=1)
    assert h._load(rid)["turn"] == 2 and len(calls) == 2 and calls[1] == 5000
    assert rt.db.get_run(rid)["used_llm_calls"] == 2


@pytest.mark.parametrize("kind", ["goal", "sources"])
def test_typed_changes_preserve_original_evidence_and_usage(tmp_path, monkeypatch, kind):
    from test_prototype_source_revisions import prepared

    rt, h, rid, _ = prepared(tmp_path, monkeypatch)
    state = h._load(rid)
    attach(state, PROFILE)
    h._save(rid, state)
    before = h._load(rid)
    run = rt.db.get_run(rid)
    value = "revised scientific question" if kind == "goal" else {"paper": "paper_v2"}
    preview = h.preview_change(
        rid, kind=kind, value=value, expected_revision=before["work"]["revision"]
    )
    assert h._load(rid) == before
    assert (
        before["work"]["tasks"]["read1"]["output"]
        in preview["change_impact"]["reusable_artifact_ids"]
    )
    if kind == "goal":
        h.revise_goal(rid, value, expected_revision=before["work"]["revision"])
    else:
        h.revise_sources(rid, value, expected_revision=before["work"]["revision"])
    after = h._load(rid)
    assert all(a in after["work"]["artifacts"] for a in before["work"]["artifacts"])
    assert after["change_impact"] == preview["change_impact"]
    for field in ("budget", "created_at", "used_llm_calls", "used_tokens"):
        assert rt.db.get_run(rid)[field] == run[field]


def test_unknown_result_never_redispatches(tmp_path, monkeypatch):
    from test_prototype_crash_recovery import setup

    rt, h, rid = setup(tmp_path, monkeypatch)
    state = h._load(rid)
    attach(state, PROFILE)
    h._save(rid, state)

    def lost(messages, before_attempt=None, **kwargs):
        if before_attempt:
            before_attempt()
        raise ConnectionError("fixture response lost after dispatch")

    monkeypatch.setattr(rt.llm_runtime.model, "complete", lost)
    h.advance(rid, max_turns=1)
    before = rt.db.get_run(rid)
    assert h._load(rid)["phase"] == "UNKNOWN"
    h.advance(rid, max_turns=1)
    assert rt.db.get_run(rid)["used_llm_calls"] == before["used_llm_calls"] == 1
    assert h._load(rid)["recovery_trace"][-1]["action"] == "WAIT_UNKNOWN"


@pytest.mark.parametrize("phase", ["publication", "cancel"])
def test_terminal_profiles_keep_trace_without_model_requests(tmp_path, monkeypatch, phase):
    from test_prototype_crash_recovery import setup

    rt, h, rid = setup(tmp_path, monkeypatch)
    state = h._load(rid)
    attach(state, PROFILE)
    if phase == "publication":
        h._finish(rid, state, "retained partial result", complete=False)
    h._save(rid, state)
    if phase == "cancel":
        rt.cancel(rid)
    h.advance(rid)
    assert rt.db.get_run(rid)["used_llm_calls"] == 0
    if phase == "publication":
        assert h._load(rid)["recovery_trace"][-1]["action"] == "PUBLISH_SAVED_REPORT"
