import json
from copy import deepcopy

import pytest

from paperpilot_agentic.reference_methods import encoded, sha
from paperpilot_agentic.research_case import register_case, load_case
from test_imported_diagnosis_harness import setup
from test_diagnosis_content_review import revised_provider


def manifest(binding):
    proposal = {"hypothesis": "Synthetic pending hypothesis", "evidence_ids": ["source-1"]}
    card = {"binding_id": "candidate-1", "hypothesis": proposal, "diff": "synthetic diff"}
    snapshot = {"evidence_id": "source-1", "content": "Synthetic paper source"}
    return {
        "version": "research-case-v1",
        "historical_run_id": "prior-run",
        "historical_candidate_id": "candidate-1",
        "experiment_bundle_id": binding["bundle_id"],
        "sources": [{"snapshot": snapshot, "sha256": sha(encoded(snapshot))}],
        "proposal": proposal,
        "approved_candidate": card,
        "approval": {
            "status": "APPROVED",
            "request": {"context": card, "arguments": {"binding_id": "candidate-1"}},
        },
        "historical_decisions": [],
        "historical_source_identity": "historical-input",
        "imported_verification": binding["verification"],
        "assurance": "synthetic case; no execution",
    }


def test_case_links_source_approval_experiment_and_new_decision_without_training(tmp_path):
    runtime, harness, reader, binding = setup(tmp_path)
    original = manifest(binding)
    saved = register_case(runtime, original)
    assert saved == register_case(runtime, original)
    assert load_case(runtime, saved["case_id"], binding["bundle_id"]) == original
    reader.complete = revised_provider(reader)
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "case-linked-diagnosis",
        report_protocol="metric-facts-v3",
        research_case_id=saved["case_id"],
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "COMPLETED", episode
    assert episode["research_case"]["historical_candidate_id"] == "candidate-1"
    assert episode["source_ids"] == ["source-1"]
    assert episode["experiment_refs"][0]["identity"] == binding["bundle_id"]
    assert episode["verification"]["training_source"].startswith("USER_DECLARED")
    assert episode["new_gpu_starts"] == 0 and not runtime.db.list_approvals(rid)
    task = json.loads(next(m["content"] for m in reader.messages[0] if m["role"] == "user"))
    assert task["research_case"]["case_id"] == saved["case_id"]
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls


@pytest.mark.parametrize("change", ["source", "approval", "provenance", "proposal"])
def test_case_rejects_broken_historical_links(tmp_path, change):
    runtime, _, _, binding = setup(tmp_path)
    value = deepcopy(manifest(binding))
    if change == "source":
        value["sources"][0]["snapshot"]["content"] = "changed"
    elif change == "approval":
        value["approval"]["status"] = "PENDING"
    elif change == "provenance":
        value["imported_verification"]["training_source"] = "VERIFIED"
    else:
        value["approved_candidate"] = {**value["approved_candidate"], "hypothesis": {}}
    with pytest.raises(ValueError):
        register_case(runtime, value)


def test_case_cannot_be_rebound_to_other_predictions_or_changed_after_registration(tmp_path):
    runtime, _, _, binding = setup(tmp_path)
    saved = register_case(runtime, manifest(binding))
    with pytest.raises(ValueError, match="different experiment"):
        load_case(runtime, saved["case_id"], "another-bundle")
    path = runtime.workspace / "research-harness/research-cases" / saved["case_id"] / "case.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="identity changed"):
        load_case(runtime, saved["case_id"], binding["bundle_id"])


def test_workbench_selects_registered_case_by_id_and_keeps_requests_idempotent(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background

    runtime, harness, reader, binding = setup(tmp_path)
    saved = register_case(runtime, manifest(binding))
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        item = client.get("/agentic/imported-experiments").json()["items"][0]
        assert item["research_cases"][0]["case_id"] == saved["case_id"]
        assert "sources" not in item["research_cases"][0]
        detail = client.get("/agentic/research-cases/" + saved["case_id"])
        assert detail.status_code == 200
        assert detail.json()["sources"][0]["content"] == "Synthetic paper source"
        assert detail.json()["candidate_diff"] == "synthetic diff"
    background = Background()
    workbench = ResearchWorkbench(runtime, background)
    payload = {
        "action": "imported_diagnosis",
        "request_id": "registered-case-workbench",
        "bundle_id": binding["bundle_id"],
        "research_case_id": saved["case_id"],
    }
    result = workbench.submit(payload)
    assert workbench.submit(payload)["reused"] is True
    assert len(background.submitted) == 1 and not reader.messages
    episode = harness.status(result["run_id"])["research_episode"]
    assert episode["research_case"]["case_id"] == saved["case_id"]
    assert episode["report_protocol"] == "metric-facts-v3"
    assert episode["pending_action"] is None
    with pytest.raises(ValueError, match="Unknown submission fields"):
        workbench.submit({**payload, "source_db": "C:/untrusted/database"})
