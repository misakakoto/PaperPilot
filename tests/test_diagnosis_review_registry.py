import pytest
from paperpilot_agentic.research_contracts import parse_diagnosis_action
from paperpilot_agentic.research_review import scientific_review_context
from test_diagnosis_scoped_review import scoped_review


def state():
    return {
        "content_review_protocol": "scientific-review-v3",
        "comparison_mode": "summary",
        "observations": {"paired": {}},
        "analysis": {
            "hypotheses": [
                {
                    "text": "A conditional response mechanism",
                    "alternatives": ["Unobserved optimization"],
                }
            ],
            "experiment_contract": {
                "controls": ["Keep weights frozen"],
                "measurements": ["Observe layer response"],
                "missing_observations": ["Activation statistics"],
            },
        },
    }


@pytest.mark.parametrize(
    "ref",
    [
        "analysis/experiment_contract/controls/0",
        "analysis/experiment_contract/measurements/0",
        "analysis/experiment_contract/missing_observations/0",
        "analysis/hypotheses/0/text",
        "analysis/hypotheses/0",
        "analysis/experiment_contract/controls",
    ],
)
def test_actual_owned_plan_fields_are_reviewable_without_becoming_host_facts(ref):
    review = scoped_review()
    review["checks"][1]["basis_refs"] = [ref]
    value = {
        "action": "finish",
        "reason": "Review a conditional proposal",
        "arguments": {"decision": "STOP", "rationale": "No execution", "content_review": review},
    }
    parsed = parse_diagnosis_action("lead", value, state())
    assert parsed["arguments"]["content_review"]["checks"][1]["basis_refs"] == [ref]
    kind = scientific_review_context(state())["review_reference_catalog"][ref]
    assert kind in {"MODEL_CLAIM", "MODEL_CONTEXT"}


@pytest.mark.parametrize(
    "ref",
    [
        "analysis/experiment_contract/controls/99",
        "analysis/hypotheses/2/text",
        "analysis/experiment_contract/not_present",
        "E:/arbitrary-file",
        "pattern/coverage/0/f1",
    ],
)
def test_registry_rejects_missing_unread_and_filesystem_references(ref):
    review = scoped_review()
    review["checks"][1]["basis_refs"] = [ref]
    with pytest.raises(ValueError, match="existing review basis ID"):
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
            state(),
        )
