import copy
from io import BytesIO
import json
from zipfile import ZipFile

import pytest

from paperpilot_agentic.prototype_delivery import build_prototype_bundle
from paperpilot_agentic.prototype_workspace import plan
from test_prototype_completion import finished


def run():
    return {
        "run_id": "run_fixture",
        "budget": {"max_llm_calls": 16},
        "used_llm_calls": 9,
        "used_tokens": 58387,
    }


def bundle(state):
    return ZipFile(BytesIO(build_prototype_bundle(state, run())))


def test_bundle_delivers_exact_code_receipt_and_partial_status():
    _, state, _ = finished()
    state.update(phase="PARTIAL", completed=False, stop_reason="Retained limitations")
    original = copy.deepcopy(state)
    with bundle(state) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        code_id = state["work"]["tasks"]["code"]["output"]
        check_id = state["work"]["tasks"]["check"]["output"]
        assert (
            archive.read(f"candidates/{code_id}/model.py").decode() == "DO_NOT_REPLAY_FULL_SOURCE"
        )
        receipt = json.loads(archive.read(f"artifacts/{check_id}.json"))
        assert receipt["content"]["validated_code_id"] == code_id
        assert manifest["phase"] == "PARTIAL"
        assert manifest["engineering_validation"] == "PASSED_FOR_CURRENT_CANDIDATES"
        assert manifest["scientific_validity"] == "NOT_VERIFIED"
        assert manifest["usage"]["used_tokens"] == 58387
        assert not any(".." in p or p.startswith("/") for p in archive.namelist())
    assert state == original


def test_bundle_records_context_and_plan_policy_without_inventing_legacy_metadata():
    _, state, _ = finished()
    with bundle(state) as archive:
        assert "plan_policy" not in json.loads(archive.read("manifest.json"))
    state.update(context_policy="artifact-table-v1", plan_policy="validation-coverage-v1")
    with bundle(state) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["plan_policy"] == state["plan_policy"]
        assert manifest["context_policy"] == state["context_policy"]


def test_changed_requirements_cannot_export_stale_pass_as_current():
    _, state, rows = finished()
    old_code = state["work"]["tasks"]["code"]["output"]
    rows[2]["goal"] = "new implementation requirement"
    plan(state["work"], rows)
    state.update(phase="READY", completed=False)
    with bundle(state) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["engineering_validation"] == "INCOMPLETE_OR_FAILED"
        index = json.loads(archive.read("artifact_index.json"))
        assert index[old_code]["current"] is False
        assert f"candidates/{old_code}/model.py" in archive.namelist()


def test_tampered_artifacts_and_missing_baseline_license_are_rejected():
    _, state, _ = finished()
    state.update(phase="PARTIAL", profile_id="bound-profile")
    with pytest.raises(ValueError, match="baseline"):
        build_prototype_bundle(state, run())
    state.pop("profile_id")
    code = state["work"]["tasks"]["code"]["output"]
    state["work"]["artifacts"][code]["content"]["source"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        build_prototype_bundle(state, run())


def test_reading_evidence_is_referenced_without_paper_full_text():
    _, state, _ = finished()
    state.update(phase="PARTIAL")
    read_id = state["work"]["tasks"]["read"]["output"]
    with bundle(state) as archive:
        index = json.loads(archive.read("artifact_index.json"))
        assert index[read_id]["content_included"] is False
        assert f"artifacts/{read_id}.json" not in archive.namelist()


def test_receipt_for_other_constraints_cannot_certify_current_requirements():
    _, state, _ = finished()
    state.update(
        phase="PARTIAL", validation_contract={"kind": "unbound", "input_cases": [[1, 32, 32]]}
    )
    with bundle(state) as archive:
        assert (
            json.loads(archive.read("manifest.json"))["engineering_validation"]
            == "INCOMPLETE_OR_FAILED"
        )


def test_http_download_is_read_only_and_rejects_nonprototype_runs(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.web_api import create_fastapi_router

    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path / "runtime.db")
    rid = runtime.db.create_run("fixture", Budget(), {"engine": "research_prototype_v1"})
    _, state, _ = finished()
    state.update(phase="PARTIAL")
    PrototypeHarness(runtime)
    with runtime.db.transaction() as connection:
        connection.execute(
            "INSERT INTO research_episodes VALUES(?,?,?,?)",
            (rid, "fixture-delivery", "fixture", json.dumps(state)),
        )
    ordinary = runtime.db.create_run("ordinary", Budget(), {})
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        response = client.get(f"/agentic/runs/{rid}/prototype-bundle")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        with ZipFile(BytesIO(response.content)) as archive:
            assert json.loads(archive.read("manifest.json"))["run_id"] == rid
        assert client.get(f"/agentic/runs/{ordinary}/prototype-bundle").status_code == 400
        assert client.get("/agentic/runs/run_missing/prototype-bundle").status_code == 404
    assert runtime.db.get_run(rid)["used_llm_calls"] == 0
    assert runtime.db.list_tool_calls(rid) == []


def test_unbound_code_reference_license_is_preserved_without_full_reading_text():
    from paperpilot_agentic.prototype_workspace import put

    _, state, _ = finished()
    state.update(phase="PARTIAL")
    identity = put(
        state["work"],
        "provided-reference",
        "read",
        {
            "evidence_type": "CODE",
            "content": "SOURCE_TEXT_MUST_REMAIN_OUTSIDE_BUNDLE",
            "source_uri": "pinned-code",
            "provenance": {
                "license_text": "Exact supplied MIT license text",
                "license_url": "pinned-license",
            },
        },
    )
    with bundle(state) as archive:
        assert (
            archive.read(f"source_licenses/{identity}/LICENSE").decode()
            == "Exact supplied MIT license text"
        )
        assert f"artifacts/{identity}.json" not in archive.namelist()
        assert all(
            b"SOURCE_TEXT_MUST_REMAIN_OUTSIDE_BUNDLE" not in archive.read(p)
            for p in archive.namelist()
        )
