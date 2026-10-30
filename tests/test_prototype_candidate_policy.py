import copy

import pytest

from paperpilot_agentic.prototype_candidate import POLICY, AUTO_POLICY
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from test_prototype_workspace import Reader, stub_environment

SOURCE = "def build_model(config):\n    return None\n"  # Never executed by these scheduling tests.


def prepared(tmp_path, monkeypatch, *, enabled=True):
    stub_environment(monkeypatch)
    rt = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path / "runtime.db", llm=Reader())
    monkeypatch.setattr(
        rt.db, "list_evidence", lambda _: [{"evidence_id": "source", "content": "owned source"}]
    )
    h = PrototypeHarness(rt)
    rid = h.start("build", "candidate-policy-fixture", "parent", ["source"])["run"]["run_id"]
    state = h._load(rid)
    if enabled:
        state.update(candidate_input_policy=POLICY, automatic_validation_policy=AUTO_POLICY)
    rows = [
        {"id": k, "goal": k, "capability": kind, "depends_on": deps, "acceptance": "fixed"}
        for k, kind, deps in [
            ("read", "read", []),
            ("design", "design", ["read"]),
            ("code", "implement", ["design"]),
            ("check", "validate", ["code"]),
        ]
    ]
    h._apply(rid, state, {"action": "plan", "reason": "fixture", "arguments": {"tasks": rows}}, 1)
    for key, content in [("read", "source"), ("design", {"architecture": "fixture"})]:
        work(h, rid, state, key, content)
    return rt, h, rid, state


def work(h, rid, state, key, content):
    return h._apply(
        rid,
        state,
        {"action": "work", "reason": "fixture", "arguments": {"task_id": key, "content": content}},
        1,
    )


def test_source_only_default_is_explicit_and_host_validation_waits_for_approval(
    tmp_path, monkeypatch
):
    rt, h, rid, state = prepared(tmp_path, monkeypatch)
    obs = work(h, rid, state, "code", SOURCE)
    code = state["work"]["artifacts"][state["work"]["tasks"]["code"]["output"]]["content"]
    assert code == {"source": SOURCE, "config": {}}
    assert (
        state["implementation_normalizations"][0]["config_origin"] == "HOST_DECLARED_EMPTY_DEFAULT"
    )
    assert state["work"]["repairs"] == 1
    h._after_action(rid, state, "lead", obs)
    assert state["phase"] == "WAITING_APPROVAL"
    pending = copy.deepcopy(state["pending"])
    assert rt.db.get_approval(pending["approval"])["status"] == "PENDING"
    assert rt.db.get_tool_call(pending["call"])["attempt"] == 0
    assert rt.db.get_run(rid)["used_llm_calls"] == 0
    h._after_action(rid, state, "lead", obs)
    assert state["pending"] == pending and rt.db.get_tool_call(pending["call"])["attempt"] == 0


def test_legacy_source_string_still_rejected(tmp_path, monkeypatch):
    rt, h, rid, state = prepared(tmp_path, monkeypatch, enabled=False)
    with pytest.raises(ValueError, match="Expected source and config"):
        work(h, rid, state, "code", SOURCE)
    assert state["implementation_feedback"]["rejected_candidate"] == SOURCE
    assert state["work"]["tasks"]["code"]["status"] == "PENDING"


def test_normalization_does_not_accept_invalid_source_or_invent_configuration(
    tmp_path, monkeypatch
):
    _, h, rid, state = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="build_model"):
        work(h, rid, state, "code", "x=1")
    assert state["implementation_feedback"]["rejected_candidate"] == "x=1"
    assert state["work"]["tasks"]["code"]["status"] == "PENDING"
    assert (
        state["implementation_feedback"]["normalization"]["config_origin"]
        == "HOST_DECLARED_EMPTY_DEFAULT"
    )


def test_explicit_config_is_not_replaced_by_defaults(tmp_path, monkeypatch):
    _, h, rid, state = prepared(tmp_path, monkeypatch)
    work(h, rid, state, "code", {"source": SOURCE, "config": {"enabled": True}})
    code = state["work"]["artifacts"][state["work"]["tasks"]["code"]["output"]]["content"]
    assert code["config"] == {"enabled": True} and not state.get("implementation_normalizations")


def test_cancellation_and_unknown_requests_do_not_schedule_execution(tmp_path, monkeypatch):
    rt, h, rid, state = prepared(tmp_path, monkeypatch)
    obs = work(h, rid, state, "code", SOURCE)
    rt.db.update_run_metadata(rid, {"cancel_requested": True})
    with pytest.raises(InterruptedError):
        h._after_action(rid, state, "lead", obs)
    assert not state.get("pending")
    rt.db.update_run_metadata(rid, {"cancel_requested": False})
    monkeypatch.setattr(rt.llm_runtime, "budget_snapshot", lambda _: {"unknown_results": 1})
    from paperpilot_agentic.episode_runner import ResearchStopped

    with pytest.raises(ResearchStopped, match="unknown"):
        h._after_action(rid, state, "lead", obs)
    assert not state.get("pending")
