import pytest

from paperpilot_agentic.research_diagnosis import DiagnosisHarness
from paperpilot_agentic.research_harness import ResearchHarness
from test_imported_diagnosis_harness import setup
from test_diagnosis_content_review import revised_provider
from test_research_harness import harness as harness, start


@pytest.mark.parametrize("engine", ["research", "diagnosis"])
@pytest.mark.parametrize("after_artifact", [False, True])
def test_terminal_publication_recovers_without_new_decision_or_training(
    harness, tmp_path, monkeypatch, engine, after_artifact
):
    if engine == "research":
        h, reader, launches = harness
        rid = start(h)
        waiting = h.advance(rid)
        h.runtime.approve(waiting["pending_approvals"][0]["approval_id"])
        owner = ResearchHarness
    else:
        folder = tmp_path / "diagnosis"
        folder.mkdir()
        _, h, reader, binding = setup(folder)
        reader.complete = revised_provider(reader)
        rid = h.start_imported_diagnosis(
            binding["bundle_id"],
            "publication-recovery-diagnosis",
            comparison_mode="summary",
            report_protocol="metric-facts-v3",
        )["run"]["run_id"]
        launches = []
        owner = DiagnosisHarness
    if after_artifact:
        original = h.db.update_run_metadata

        def die(run_id, payload, *args, **kwargs):
            if "final_answer" in payload:
                raise KeyboardInterrupt("Simulated death after local artifact commit")
            return original(run_id, payload, *args, **kwargs)

        monkeypatch.setattr(h.db, "update_run_metadata", die)
    else:
        original = owner._publish

        def die(*args, **kwargs):
            raise KeyboardInterrupt("Simulated death after terminal state commit")

        monkeypatch.setattr(owner, "_publish", die)
    with pytest.raises(KeyboardInterrupt):
        h.advance(rid)
    if after_artifact:
        monkeypatch.setattr(h.db, "update_run_metadata", original)
    else:
        monkeypatch.setattr(owner, "_publish", original)
    calls, starts = len(reader.messages), len(launches)
    assert h.status(rid)["research_episode"]["phase"] == "COMPLETED"
    assert h.status(rid)["final_answer"] is None
    result = h.advance(rid)
    assert result["run"]["status"] == "COMPLETED"
    assert result["final_answer"] is not None
    assert len(reader.messages) == calls and len(launches) == starts
    reports = [
        a
        for a in result["artifacts"]
        if a["kind"] in {"research-episode-report", "research-diagnosis-report"}
    ]
    assert len(reports) == 1
    h.advance(rid)
    assert len(reader.messages) == calls and len(launches) == starts

