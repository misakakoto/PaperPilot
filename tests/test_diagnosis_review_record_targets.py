import copy
import json

import pytest

from paperpilot_agentic.research_contracts import parse_diagnosis_action
from paperpilot_agentic.research_review import scientific_review_context
from test_diagnosis_comparison_design import design, proposal
from test_diagnosis_scoped_review import scoped_review


def fixture():
    plan = proposal()
    plan["comparison_design"] = design()
    state = {
        "content_review_protocol": "scientific-review-v3", "comparison_mode": "summary",
        "observations": {"paired": {}},
        "analysis": {"hypotheses": [{"text": "A conditional explanation"}],
                     "experiment_contract": plan},
    }
    review = scoped_review("DEFECT")
    review["checks"][1].update(
        claim_path="analysis/experiment_contract/comparison_design/outcomes/1",
        correction="Review the entire declared outcome and its residual alternatives",
    )
    return state, {"action": "finish", "reason": "Review a specific outcome record",
                   "arguments": {"decision": "STOP", "rationale": "No execution",
                                 "content_review": review}}


def test_registered_record_target_resolves_exact_host_quote_without_mutating_analysis():
    state, action = fixture()
    original = copy.deepcopy(state)
    parsed = parse_diagnosis_action("lead", action, state)
    item = parsed["arguments"]["content_review"]["checks"][1]
    expected = state["analysis"]["experiment_contract"]["comparison_design"]["outcomes"][1]
    assert json.loads(item["claim_quote"]) == expected
    assert state == original
    registry = scientific_review_context(state)["review_reference_catalog"]
    assert registry[item["claim_path"]] == "MODEL_CLAIM"
    assert registry[item["claim_path"] + "/observation"] == "MODEL_CLAIM"
    assert registry["analysis/experiment_contract/comparison_design"] == "MODEL_CONTEXT"


@pytest.mark.parametrize("path", [
    "analysis/experiment_contract",
    "analysis/experiment_contract/comparison_design",
    "analysis/experiment_contract/comparison_design/outcomes",
    "analysis/experiment_contract/comparison_design/outcomes/99",
    "analysis/unknown/0",
])
def test_wide_containers_and_nonexistent_records_are_not_criticizable_claims(path):
    state, action = fixture()
    action["arguments"]["content_review"]["checks"][1]["claim_path"] = path
    with pytest.raises(ValueError, match="existing analysis claim"):
        parse_diagnosis_action("lead", action, state)


def test_record_target_does_not_allow_an_invented_quote():
    state, action = fixture()
    action["arguments"]["content_review"]["checks"][1]["claim_quote"] = "A different outcome"
    with pytest.raises(ValueError, match="existing analysis claim"):
        parse_diagnosis_action("lead", action, state)
