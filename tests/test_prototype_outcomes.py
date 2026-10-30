import copy
import json
from io import BytesIO
from zipfile import ZipFile

import pytest

from paperpilot_agentic.prototype_delivery import build_prototype_bundle
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_outcomes import validated_candidates
from paperpilot_agentic.prototype_workspace import new_workspace, plan, complete, digest, task_rows
from test_prototype_delivery import run


def partial():
    work = new_workspace("Repair within unchanged budget")
    rows = [
        {
            "id": key,
            "goal": key,
            "capability": kind,
            "depends_on": deps,
            "acceptance": "fixed contract",
        }
        for key, kind, deps in [
            ("read", "read", []),
            ("design", "design", ["read"]),
            ("bad", "implement", ["design"]),
            ("bad_check", "validate", ["bad"]),
            ("fixed", "implement", ["design"]),
            ("fixed_check", "validate", ["fixed"]),
            ("review", "review", ["design", "fixed", "fixed_check"]),
        ]
    ]
    plan(work, rows)
    contract = {"kind": "unbound", "input_cases": [[1, 33, 49]], "max_parameters": 8000000}
    for key, kind in [("read", "read"), ("design", "design")]:
        complete(work, key, kind, {"source": "fixture"})
    for key, passed in [("bad", False), ("fixed", True)]:
        code = complete(
            work, key, "implement", {"source": "def build_model(config): pass", "config": {}}
        )
        complete(
            work,
            key + "_check",
            "validate",
            {
                "passed": passed,
                "validated_code_id": code,
                "validation_contract": contract,
                "execution": {"metrics": {"parameters": 6773115}},
            },
        )
    return {
        "work": work,
        "validation_contract": contract,
        "phase": "FAILED",
        "completed": False,
        "stop_reason": "LLMBudgetExhausted: protected remaining budget",
        "sources": {},
    }


def test_failed_task_preserves_exact_passed_candidate_without_claiming_completion():
    state = partial()
    before = copy.deepcopy(state)
    rows = validated_candidates(state)
    assert (
        len(rows) == 1 and rows[0]["code_artifact_id"] == state["work"]["tasks"]["fixed"]["output"]
    )
    assert rows[0]["scientific_validity"] == "NOT_VERIFIED"
    assert "NOT_TASK_COMPLETION" in rows[0]["scope"]
    assert state == before and state["completed"] is False


def test_status_and_bundle_show_same_passed_candidate_and_failed_task():
    state = partial()
    harness = PrototypeHarness.__new__(PrototypeHarness)
    view = harness.view("unused", state=state)
    with ZipFile(BytesIO(build_prototype_bundle(state, run()))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["phase"] == "FAILED" and manifest["completed"] is False
        assert manifest["engineering_validation"] == "INCOMPLETE_OR_FAILED"
        assert manifest["validated_candidates"] == view["validated_candidates"]
        code = view["validated_candidates"][0]["code_artifact_id"]
        assert f"candidates/{code}/model.py" in archive.namelist()


@pytest.mark.parametrize("change", ["contract", "code", "design"])
def test_new_requirements_or_code_exclude_old_pass_while_preserving_history(change):
    state = partial()
    old = state["work"]["tasks"]["fixed_check"]["output"]
    if change == "contract":
        state["validation_contract"] = {**state["validation_contract"], "max_parameters": 100}
    else:
        rows = task_rows(state["work"])
        next(t for t in rows if t["id"] == ("fixed" if change == "code" else "design"))["goal"] = (
            "changed requirement"
        )
        plan(state["work"], rows)
    assert validated_candidates(state) == []
    assert old in state["work"]["artifacts"]


@pytest.mark.parametrize("change", ["passed", "validated_code_id", "validation_contract"])
def test_wrong_failed_or_foreign_receipt_is_not_usable(change):
    state = partial()
    work = state["work"]
    task = work["tasks"]["fixed_check"]
    artifact = copy.deepcopy(work["artifacts"][task["output"]])
    artifact["content"][change] = False if change == "passed" else "other version"
    identity = digest(artifact)
    work["artifacts"][identity] = artifact
    task["output"] = identity
    assert validated_candidates(state) == []


def test_tampered_artifact_and_changed_dependency_cannot_be_presented_as_pass():
    state = partial()
    work = state["work"]
    work["artifacts"][work["tasks"]["fixed"]["output"]]["content"]["source"] = "tampered"
    with pytest.raises(ValueError, match="identity"):
        validated_candidates(state)
    state = partial()
    work = state["work"]
    artifact = copy.deepcopy(work["artifacts"][work["tasks"]["fixed"]["output"]])
    artifact["content"]["source"] = "new code"
    identity = digest(artifact)
    work["artifacts"][identity] = artifact
    work["tasks"]["fixed"]["output"] = identity
    with pytest.raises(ValueError, match="dependency version"):
        validated_candidates(state)


def test_legacy_missing_contract_does_not_gain_certified_candidate_summary():
    state = partial()
    state.pop("validation_contract")
    assert validated_candidates(state) == []
