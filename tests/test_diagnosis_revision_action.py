import copy
import json

import pytest
from jsonschema import Draft202012Validator

from paperpilot_agentic.research_contracts import diagnosis_action_schema, parse_diagnosis_action
from paperpilot_agentic.providers import LLMResponse
from test_diagnosis_comparison_design import design
from test_diagnosis_content_review import revised_provider
from test_diagnosis_scoped_review import scoped_review
from test_imported_diagnosis_harness import setup


def review():
    value = scoped_review("DEFECT")
    value["verdict"] = "NEEDS_REVISION"
    value["checks"][1].update(
        claim_path="analysis/hypotheses/0/text",
        correction="Keep the explanation conditional and distinguish alternatives",
    )
    return value


def state():
    return {"content_review_protocol": "scientific-review-v3", "comparison_mode": "summary",
            "analysis": {"hypotheses": [{"text": "An unverified interpretation"}],
                         "experiment_contract": {"decision": "PROPOSE_EXPERIMENT"}},
            "observations": {"paired": {}}}


def action():
    return {"action": "review", "reason": "Request one bounded content revision",
            "arguments": {"content_review": review()}}


def test_revision_review_omits_task_decision_and_uses_shared_schema_and_reference_validation():
    s, value = state(), action()
    original = copy.deepcopy(value)
    schema = diagnosis_action_schema("lead", s)["review"]["arguments"]
    assert Draft202012Validator(schema).is_valid(value["arguments"])
    parsed = parse_diagnosis_action("lead", value, s)
    assert set(parsed["arguments"]) == {"content_review"}
    assert parsed["arguments"]["content_review"]["checks"][1]["claim_quote"] == "An unverified interpretation"
    assert value == original


@pytest.mark.parametrize("profile,role,has_analysis", [
    ("scientific-review-v2", "lead", True),
    ("scientific-review-v3", "analyst", True),
    ("scientific-review-v3", "lead", False),
])
def test_revision_action_does_not_expand_legacy_roles_or_preanalysis_permissions(profile, role, has_analysis):
    s = state()
    s["content_review_protocol"] = profile
    if not has_analysis:
        s["analysis"] = None
    with pytest.raises(ValueError, match="not available"):
        parse_diagnosis_action(role, action(), s)


def test_revision_action_does_not_accept_limits_only_or_unowned_claims():
    value = action()
    value["arguments"]["content_review"] = scoped_review()
    with pytest.raises(ValueError):
        parse_diagnosis_action("lead", value, state())
    value = action()
    value["arguments"]["content_review"]["checks"][1]["claim_path"] = "analysis/unknown"
    with pytest.raises(ValueError, match="existing analysis claim"):
        parse_diagnosis_action("lead", value, state())


def test_review_verdict_is_still_not_a_valid_finish_decision():
    value = action()
    value["action"] = "finish"
    value["arguments"].update(decision="NEEDS_REVISION", rationale="Wrong namespace")
    with pytest.raises(ValueError, match="registered finish decision"):
        parse_diagnosis_action("lead", value, state())


@pytest.mark.parametrize("repeat_defect", [False, True])
def test_harness_routes_revision_without_final_decision_and_retains_facts_on_exhaustion(tmp_path, repeat_defect):
    _, harness, reader, binding = setup(tmp_path)
    normal = revised_provider(reader)
    seen = []

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report":
            value["arguments"]["next_experiment"]["comparison_design"] = design()
        elif value["action"] == "finish":
            seen.append(value)
            if len(seen) == 1 or repeat_defect:
                value = action()
            else:
                value["arguments"]["content_review"] = scoped_review()
        return LLMResponse(text=json.dumps(value), model=reader.model, finish_reason="stop",
                           prompt_tokens=10, completion_tokens=10)

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"], "revision-action-001", comparison_mode="summary",
        report_protocol="metric-facts-v3", content_review_protocol="scientific-review-v3",
    )["run"]["run_id"]
    first = harness.advance(rid, max_turns=3)
    episode = first["research_episode"]
    assert episode["role"] == "analyst"
    assert episode["analysis_revisions"] == 1
    assert not episode.get("final_decision")
    assert episode["decisions"][-1]["action"] == "review"
    assert episode["decisions"][-1]["status"] == "REVISE"
    assert episode["decisions"][-1]["reason_assurance"] == "UNVERIFIED_ACTION_TRACE_NOT_EVIDENCE"
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == ("PARTIAL" if repeat_defect else "COMPLETED")
    assert episode["analysis_revisions"] == 1
    assert episode["new_gpu_starts"] == 0
    assert "宿主生成的观测" in result["final_answer"]["answer"]
    assert len(reader.messages) == 5
    if repeat_defect:
        assert not episode.get("final_decision")
        assert "模型复核的待验证假设" not in result["final_answer"]["answer"]
        assert "负责人采纳的实验建议" not in result["final_answer"]["answer"]
    else:
        assert episode["final_decision"]["execution_authorized"] is False
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
