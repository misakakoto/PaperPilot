import copy
import json

from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_diagnosis import DiagnosisHarness
from test_diagnosis_comparison_design import design
from test_diagnosis_content_review import revised_provider
from test_diagnosis_scoped_review import scoped_review
from test_imported_diagnosis_harness import setup


def test_actual_review_has_fresh_context_but_keeps_sources_and_repair_history(tmp_path):
    runtime, harness, reader, binding = setup(tmp_path)
    normal = revised_provider(reader)
    marker = "UNVERIFIED_EARLY_CONCLUSION"

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "delegate_analysis":
            value["reason"] = marker
            value["arguments"] = {"feedback": marker}
        elif value["action"] == "report":
            value["arguments"]["next_experiment"]["comparison_design"] = design()
        elif value["action"] == "finish":
            value["arguments"]["content_review"] = scoped_review()
        return LLMResponse(text=json.dumps(value), model=reader.model,
                           finish_reason="stop", prompt_tokens=10, completion_tokens=10)

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"], "isolated-review-001", comparison_mode="summary",
        report_protocol="metric-facts-v3", content_review_protocol="scientific-review-v3",
    )["run"]["run_id"]
    harness.advance(rid, max_turns=2)
    owner = DiagnosisHarness(runtime)
    state = owner._load(rid)
    assert marker in json.dumps(state["contexts"]["lead"])
    assert state["analysis_feedback"] == marker
    before = copy.deepcopy(state)
    key, messages = owner._context(rid, state)
    assert marker not in json.dumps(messages)
    assert state == before  # Preserve the original action trace and feedback.
    task = json.loads(messages[1]["content"])
    assert task["independent_analysis"]["experiment_contract"]["comparison_design"] == design()
    assert task["observations"] and task["research_case"] is None
    assert task["host_observation_catalog"] and task["scientific_review_context"]
    policy = task["scientific_review_context"]["assessment_policy"]
    assert policy == {
        "version": "claim-before-disposition-v1",
        "order": ["check_report_claims", "request_bounded_revision_or_finish"],
        "stop_exempts_report_from_review": False,
        "unexecuted_measurement_is_not_itself_defect": True,
        "intervention_response_is_not_historical_cause_identification": True,
        "assurance": "MODEL_REVIEW_GUIDANCE_NOT_HOST_SCIENTIFIC_VERDICT",
    }
    timing = task["scientific_review_context"]["basis"]["scope/proposal_timing"]
    assert timing["path"] == "analysis/experiment_contract"
    assert timing["status"] == "UNEXECUTED_RESEARCH_PLAN"
    assert timing["execution_authority"] == "NONE_IN_THIS_DIAGNOSIS"
    assert (
        task["scientific_review_context"]["review_reference_catalog"]["scope/proposal_timing"]
        == "HOST_SCOPE"
    )
    assert (
        task["scientific_review_context"]["review_reference_catalog"][
            "analysis/experiment_contract/comparison_design/mode"
        ] == "MODEL_CLAIM"
    )
    assert task["analysis_feedback"] is None and task["content_review"] is None
    assert task["last_observation"] is None

    # Retries for this same report retain their own parser feedback. A revised
    # report uses a fresh context, rather than repeating the previous verdict.
    state["contexts"][key] = [{"role": "user", "content": json.dumps({
        "tool_result": {"status": "REJECTED", "error": "JSON_REPAIR_FOR_THIS_REPORT"}
    })}]
    assert "JSON_REPAIR_FOR_THIS_REPORT" in json.dumps(owner._context(rid, state)[1])
    state["analysis_revisions"] = 1
    next_key, next_messages = owner._context(rid, state)
    assert next_key != key
    assert "JSON_REPAIR_FOR_THIS_REPORT" not in json.dumps(next_messages)

    result = harness.advance(rid)
    assert result["research_episode"]["phase"] == "COMPLETED"
    review_messages = reader.messages[-1]
    assert marker not in json.dumps(review_messages)
    assert marker in json.dumps(owner._load(rid)["contexts"]["lead"])
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
    assert result["research_episode"]["new_gpu_starts"] == 0
