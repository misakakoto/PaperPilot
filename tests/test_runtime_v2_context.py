import copy
import json
from paperpilot_agentic.context_compiler import ContextCompiler
from paperpilot_agentic.runtime_profile import PROFILE, attach


class Owner:
    def _context(self, rid, state):
        return "prototype", [
            {"role": "system", "content": "trusted instructions"},
            {"role": "user", "content": json.dumps(state["payload"])},
        ]

    def _decision_output_tokens(self, state):
        return 1500


def test_compiler_preserves_required_data_and_compacts_optional_history():
    state = {
        "payload": {
            "goal": "fixed research",
            "active_task_id": "code",
            "tasks": {
                "code": {"goal": "repair", "acceptance": "same size logits"},
                "old": {"notes": "x" * 20000},
            },
            "validation_contract": {"input_cases": [[2, 65, 97]]},
            "artifact_index": {"old": {"noise": "x" * 20000}},
            "context_artifacts": {},
            "feedback": {
                "status": "REJECTED",
                "error": "65 versus64",
                "previous_response": {"noise": "x" * 20000},
            },
            "candidate_patch": {"base_artifact_id": "base"},
        }
    }
    attach(state, PROFILE)
    before = copy.deepcopy(state)
    result = ContextCompiler().compile(
        Owner(),
        "r",
        state,
        {"remaining_tokens": 9000, "remaining_llm_calls": 2, "unknown_results": 0},
        protected_tokens=0,
        window_tokens=131072,
    )
    assert result.manifest["status"] == "READY" and result.output_tokens == 3000
    p = json.loads(result.messages[-1]["content"])
    assert p["goal"] == "fixed research" and p["feedback"]["error"] == "65 versus64"
    assert (
        p["validation_contract"] == state["payload"]["validation_contract"]
        and p["tasks"]["code"]["acceptance"] == "same size logits"
    )
    assert state == before and result.manifest["projections"]


def test_compiler_blocks_without_discarding_required_goal_or_unknown_holds():
    state = {"payload": {"goal": "required " * 10000, "constraints": ["must retain"]}}
    attach(state, PROFILE)
    c = ContextCompiler()
    r = c.compile(
        Owner(),
        "r",
        state,
        {"remaining_tokens": 5000, "remaining_llm_calls": 1, "unknown_results": 0},
    )
    assert r.manifest["status"] == "BLOCKED" and r.manifest["missing_tokens"] > 0
    r = c.compile(
        Owner(),
        "r",
        state,
        {"remaining_tokens": 5000, "remaining_llm_calls": 1, "unknown_results": 1},
    )
    assert r.manifest["reason"] == "UNKNOWN_EXTERNAL_RESULT"


def test_patch_output_escalates_only_after_recorded_truncation():
    state = {
        "payload": {
            "goal": "g",
            "candidate_patch": {"base_artifact_id": "b"},
            "active_task_id": "code",
        },
        "output_truncations": {"code": 1},
    }
    attach(state, PROFILE)
    r = ContextCompiler().compile(
        Owner(),
        "r",
        state,
        {"remaining_tokens": 20000, "remaining_llm_calls": 1, "unknown_results": 0},
    )
    assert r.output_tokens == 5000
    state["output_truncations"]["code"] = 2
    r = ContextCompiler().compile(
        Owner(),
        "r",
        state,
        {"remaining_tokens": 20000, "remaining_llm_calls": 1, "unknown_results": 0},
    )
    assert r.manifest["status"] == "BLOCKED"
