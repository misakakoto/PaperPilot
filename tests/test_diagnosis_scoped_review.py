import copy
import pytest
from paperpilot_agentic.research_review import apply_content_review


def scoped_review(finding="LIMITATION"):
    return {
        "verdict": "ACCEPT_TENTATIVE",
        "rationale": "A future measurement is not a claim that it has already been observed",
        "checks": [
            {
                "aspect": a,
                "finding": finding if a == "causal_identification" else "NO_DEFECT",
                "statement": "Activation measurement is proposed, not asserted as evidence",
                "basis_refs": ["analysis/experiment_contract"],
                "claim_path": "",
                "claim_quote": "",
                "correction": "",
            }
            for a in (
                "source_scope",
                "causal_identification",
                "discriminating_measurement",
                "controlled_comparison",
            )
        ],
    }


def test_limitation_does_not_consume_revision_or_remove_tentative_proposal():
    state = {"content_review_protocol": "scientific-review-v3", "analysis_revisions": 0}
    review = scoped_review()
    original = copy.deepcopy(review)
    assert apply_content_review(state, review, "PROPOSE_EXPERIMENT") == "FINISH"
    assert state["analysis_revisions"] == 0
    assert state["content_review"]["limitations"]
    assert state["content_review"]["verdict"] == "ACCEPT_TENTATIVE"
    assert review == original


def test_revision_requires_a_concrete_quoted_defect_not_missing_measurement():
    state = {"content_review_protocol": "scientific-review-v3", "analysis_revisions": 0}
    review = scoped_review()
    review["verdict"] = "NEEDS_REVISION"
    with pytest.raises(ValueError, match="quoted defect"):
        apply_content_review(state, review, "STOP")
    assert "content_review" not in state


@pytest.mark.parametrize("defect", [False, True])
def test_scoped_harness_accepts_limits_and_bounds_actual_defect_revision(tmp_path, defect):
    import json
    from paperpilot_agentic.providers import LLMResponse
    from test_imported_diagnosis_harness import setup
    from test_diagnosis_content_review import revised_provider
    from test_diagnosis_comparison_design import design

    _, harness, reader, binding = setup(tmp_path)
    normal = revised_provider(reader)

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report":
            value["arguments"]["next_experiment"]["comparison_design"] = design()
        if value["action"] == "finish":
            review = scoped_review("DEFECT" if defect else "LIMITATION")
            if defect:
                item = review["checks"][1]
                item.update(
                    claim_path="analysis/hypotheses/0/text",
                    claim_quote="F1下降可能需要激活观察解释",
                    correction="Keep the causal explanation conditional and name an alternative",
                )
            value["arguments"]["content_review"] = review
        return LLMResponse(
            text=json.dumps(value),
            model=reader.model,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
        )

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding["bundle_id"],
        "scoped-review-task-001",
        comparison_mode="summary",
        report_protocol="metric-facts-v3",
        content_review_protocol="scientific-review-v3",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["content_review_protocol"] == "scientific-review-v3"
    assert episode["phase"] == ("PARTIAL" if defect else "COMPLETED"), episode["stop_reason"]
    assert episode["analysis_revisions"] == (1 if defect else 0)
    assert "宿主生成的观测" in result["final_answer"]["answer"]
    assert episode["new_gpu_starts"] == 0
    assert (
        episode["analysis"]["experiment_contract"]["comparison_design"]["target"]
        == "paired_difference"
    )
    assert (
        "DECLARED_STRUCTURE_CHECKED"
        in episode["analysis"]["projection_record"]["comparison_design_validation"]
    )
    # Inspect the actual provider inputs, not only the persisted analysis. Input
    # compaction must keep the design that the lead is asked to review.
    tasks = [
        json.loads(next(m["content"] for m in messages if m["role"] == "user"))
        for messages in reader.messages
    ]
    reviews = [task for task in tasks if task.get("independent_analysis")]
    assert reviews
    for task in reviews:
        sent_plan = task["independent_analysis"]["experiment_contract"]
        assert sent_plan["comparison_design"] == design()
        assert (
            task["scientific_review_context"]["basis"]["analysis/experiment_contract"]
            == sent_plan
        )
        assert (
            task["scientific_review_context"]["review_reference_catalog"][
                "analysis/experiment_contract/comparison_design/outcomes/1/observation"
            ]
            == "MODEL_CLAIM"
        )
    if defect:
        assert any(task.get("analysis_feedback") for task in tasks)
        assert reviews[-1]["analysis_revisions_remaining"] == 0
    if not defect:
        assert "审核保留的限制" in result["final_answer"]["answer"]
        assert "F1下降可能" in result["final_answer"]["answer"]
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls


@pytest.mark.parametrize(
    "path,quote",
    [
        ("analysis/hypotheses/0/text", "Not the actual claim"),
        ("analysis/nonexistent", "Conditional mechanism"),
    ],
)
def test_scoped_parser_refuses_invented_criticism(path, quote):
    from paperpilot_agentic.research_contracts import parse_diagnosis_action

    state = {
        "content_review_protocol": "scientific-review-v3",
        "analysis": {"hypotheses": [{"text": "Conditional mechanism"}]},
        "comparison_mode": "summary",
        "observations": {"paired": {}},
    }
    review = scoped_review("DEFECT")
    review["checks"][1].update(
        claim_path=path, claim_quote=quote, correction="Correct the actual assertion"
    )
    with pytest.raises(ValueError, match="existing analysis claim"):
        parse_diagnosis_action(
            "lead",
            {
                "action": "finish",
                "reason": "Review",
                "arguments": {
                    "decision": "STOP",
                    "rationale": "No execution",
                    "content_review": review,
                },
            },
            state,
        )


def test_host_projects_defect_quote_from_registered_claim_without_model_transcription():
    from paperpilot_agentic.research_contracts import parse_diagnosis_action

    state = {
        "content_review_protocol": "scientific-review-v3",
        "analysis": {"hypotheses": [{"text": "A conditional explanation, not proven"}]},
        "comparison_mode": "summary",
        "observations": {"paired": {}},
    }
    review = scoped_review("DEFECT")
    review["checks"][1].update(
        claim_path="analysis/hypotheses/0/text", correction="Name the alternative"
    )
    value = {
        "action": "finish",
        "reason": "Review",
        "arguments": {"decision": "STOP", "rationale": "No execution", "content_review": review},
    }
    original = copy.deepcopy(value)
    parsed = parse_diagnosis_action("lead", value, state)
    assert (
        parsed["arguments"]["content_review"]["checks"][1]["claim_quote"]
        == "A conditional explanation, not proven"
    )
    assert value == original


def test_scoped_review_identity_is_frozen_and_cannot_be_changed_on_duplicate_submit(tmp_path):
    from test_imported_diagnosis_harness import setup

    _, harness, reader, binding = setup(tmp_path)
    first = harness.start_imported_diagnosis(
        binding["bundle_id"], "scoped-identity-001", report_protocol="metric-facts-v3"
    )
    with pytest.raises(ValueError, match="different inputs"):
        harness.start_imported_diagnosis(
            binding["bundle_id"],
            "scoped-identity-001",
            report_protocol="metric-facts-v3",
            content_review_protocol="scientific-review-v3",
        )
    assert not reader.messages
    assert (
        harness.status(first["run"]["run_id"])["research_episode"]["content_review_protocol"]
        == "scientific-review-v2"
    )


def test_scoped_review_requires_structured_report_protocol(tmp_path):
    from test_imported_diagnosis_harness import setup

    _, harness, reader, binding = setup(tmp_path)
    with pytest.raises(ValueError, match="requires metric-facts-v3"):
        harness.start_imported_diagnosis(
            binding["bundle_id"],
            "scoped-invalid-001",
            content_review_protocol="scientific-review-v3",
        )
    assert not reader.messages
