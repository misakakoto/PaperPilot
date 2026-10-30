import pytest

from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.research_training import episode_folder
from test_prototype_workspace import Reader


def blocked(tmp_path, monkeypatch):
    runtime = ResearchAgentRuntime(
        workspace=tmp_path, db_path=tmp_path / "runtime.db", llm=Reader()
    )
    monkeypatch.setattr(
        runtime.db, "list_evidence", lambda _: [{"evidence_id": "source", "content": "source"}]
    )
    harness = PrototypeHarness(runtime)
    rid = harness.start("build", "budget-stop-fixture", "parent", ["source"])["run"]["run_id"]
    complete = runtime.llm_runtime.complete

    def budget_stop(*args, **kwargs):
        raise LLMBudgetExhausted("Token budget cannot safely start research.lead.1")

    monkeypatch.setattr(runtime.llm_runtime, "complete", budget_stop)
    harness.advance(rid)
    assert harness._load(rid)["phase"] == "FAILED"
    monkeypatch.setattr(runtime.llm_runtime, "complete", complete)
    return runtime, harness, rid


def test_recover_known_undispatched_turn_preserves_request_budget_and_task(tmp_path, monkeypatch):
    runtime, harness, rid = blocked(tmp_path, monkeypatch)
    before = runtime.db.get_run(rid)
    folder = episode_folder(runtime, rid)
    original = (folder / "turn-01-request.json").read_bytes()
    harness.recover_undispatched_budget_stop(rid)
    assert harness._load(rid)["phase"] == "READY"
    assert "stop_reason" not in harness._load(rid)
    assert harness._load(rid)["preflight_recoveries"][-1]["reason"].startswith("LLMBudgetExhausted:")
    assert runtime.db.get_run(rid)["budget"] == before["budget"]
    assert runtime.db.get_run(rid)["used_llm_calls"] == 0
    harness.advance(rid, max_turns=1)
    assert harness._load(rid)["turn"] == 1
    assert runtime.db.get_run(rid)["used_llm_calls"] == 1
    assert (folder / "turn-01-request.json").read_bytes() == original
    assert (folder / "turn-01-request-revision-01.json").exists()
    assert runtime.db.get_run(rid)["created_at"] == before["created_at"]


def test_recovery_refuses_any_dispatch_evidence(tmp_path, monkeypatch):
    runtime, harness, rid = blocked(tmp_path, monkeypatch)
    runtime.db.emit_event(rid, "llm.attempt.started", {"stage": "research.lead.1"})
    with pytest.raises(ValueError, match="dispatch"):
        harness.recover_undispatched_budget_stop(rid)
    assert harness._load(rid)["phase"] == "FAILED"


def test_recovery_refuses_unknown_result_and_other_failure(tmp_path, monkeypatch):
    runtime, harness, rid = blocked(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime.llm_runtime, "budget_snapshot", lambda _: {"unknown_results": 1})
    with pytest.raises(ValueError, match="unknown"):
        harness.recover_undispatched_budget_stop(rid)
    state = harness._load(rid)
    state["stop_reason"] = "Some provider failure"
    harness._save(rid, state)
    with pytest.raises(ValueError, match="budget"):
        harness.recover_undispatched_budget_stop(rid)


def test_explicit_focused_upgrade_is_versioned_and_does_not_reset_budget(tmp_path, monkeypatch):
    from paperpilot_agentic.prototype_prompt import POLICY

    runtime, harness, rid = blocked(tmp_path, monkeypatch)
    before = runtime.db.get_run(rid)
    harness.recover_undispatched_budget_stop(rid, instruction_policy=POLICY)
    state = harness._load(rid)
    assert state["instruction_policy"] == POLICY and state["phase"] == "READY"
    record = state["preflight_recoveries"][-1]
    assert (
        record["prompt_upper_bound"] + 5000
        <= record["remaining_tokens"] - record["protected_tokens"]
    )
    after = runtime.db.get_run(rid)
    assert (
        after["budget"],
        after["used_tokens"],
        after["used_llm_calls"],
        after["created_at"],
    ) == (before["budget"], before["used_tokens"], before["used_llm_calls"], before["created_at"])


def test_focused_upgrade_refuses_if_context_still_exceeds_remaining_budget(tmp_path, monkeypatch):
    from paperpilot_agentic.prototype_prompt import POLICY

    runtime, harness, rid = blocked(tmp_path, monkeypatch)
    before = harness._load(rid)
    original = runtime.llm_runtime.budget_snapshot
    monkeypatch.setattr(
        runtime.llm_runtime, "budget_snapshot", lambda run: {**original(run), "remaining_tokens": 1}
    )
    with pytest.raises(ValueError, match="still cannot fit"):
        harness.recover_undispatched_budget_stop(rid, instruction_policy=POLICY)
    assert harness._load(rid) == before
    with runtime.db.connect() as conn:
        assert (
            conn.execute(
                "SELECT state FROM research_episode_turns WHERE run_id=? AND turn=1", (rid,)
            ).fetchone()[0]
            == "STARTED"
        )


def test_recovery_episode_write_failure_does_not_mark_turn_undispatched(tmp_path, monkeypatch):
    import sqlite3
    from paperpilot_agentic.prototype_prompt import POLICY

    runtime, harness, rid = blocked(tmp_path, monkeypatch)
    before = harness._load(rid)
    with runtime.db.connect() as conn:
        conn.execute(
            "CREATE TRIGGER fail_upgrade BEFORE UPDATE ON research_episodes BEGIN SELECT RAISE(ABORT,'recovery commit failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="recovery commit failure"):
        harness.recover_undispatched_budget_stop(rid, instruction_policy=POLICY)
    assert harness._load(rid) == before
    with runtime.db.connect() as conn:
        assert (
            conn.execute(
                "SELECT state FROM research_episode_turns WHERE run_id=? AND turn=1", (rid,)
            ).fetchone()[0]
            == "STARTED"
        )
