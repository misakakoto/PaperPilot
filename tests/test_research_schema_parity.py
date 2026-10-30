from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from paperpilot_agentic.research_contracts import (
    diagnosis_action_schema,
    parse_diagnosis_action,
    research_action_schema,
    parse_research_action,
)


def check(role, action, arguments, valid, *, reviewed=False):
    schema = research_action_schema(role, reviewed_result=reviewed)[action]["arguments"]
    assert Draft202012Validator(schema).is_valid(arguments) is valid
    value = {"action": action, "reason": "A public action rationale", "arguments": arguments}
    if valid:
        parse_research_action(role, value, reviewed_result=reviewed)
    else:
        with pytest.raises(ValueError):
            parse_research_action(role, value, reviewed_result=reviewed)


@pytest.mark.parametrize(
    "arguments,valid",
    [
        ({"source_ref": "S1"}, True),
        ({"source_id": "source-1"}, True),
        ({}, False),
        ({"source_ref": None}, False),
        ({"source_ref": "S1", "source_id": "source-1"}, False),
        ({"source_ref": None, "source_id": "source-1"}, False),
        ({"source_ref": " "}, False),
    ],
)
def test_source_choice_schema_and_parser_agree(arguments, valid):
    check("lead", "read_source", arguments, valid)


@pytest.mark.parametrize(
    "arguments,valid",
    [
        ({"revision_feedback": "Explain the shape constraint"}, True),
        ({}, False),
        ({"revision_feedback": None}, False),
        ({"proposal": None}, False),
        ({"proposal": None, "revision_feedback": "Fix it"}, False),
    ],
)
def test_patch_revision_choice_schema_and_parser_agree(arguments, valid):
    check("lead", "delegate_patch", arguments, valid)


def test_required_content_review_cannot_be_schema_valid_null():
    review = {
        "verdict": "ACCEPT_TENTATIVE",
        "issues": [],
        "rationale": "No causal conclusion claimed",
    }
    check(
        "lead",
        "stop",
        {"summary": "Stop with evidence gaps", "content_review": review},
        True,
        reviewed=True,
    )
    for arguments in [{"summary": "Stop"}, {"summary": "Stop", "content_review": None}]:
        check("lead", "stop", arguments, False, reviewed=True)
    check("lead", "stop", {"summary": "Stop before experiments"}, True, reviewed=False)


def test_revision_requires_an_issue_in_generated_schema_as_well_as_parser():
    args = {
        "decision": "STOP",
        "rationale": "Stop",
        "content_review": {
            "verdict": "NEEDS_REVISION",
            "issues": [],
            "rationale": "Unmeasured cause",
        },
    }
    schema = diagnosis_action_schema("lead")["finish"]["arguments"]
    for issues, valid in [
        ([], False),
        ([" "], False),
        (["The observation does not distinguish causes"], True),
    ]:
        value = deepcopy(args)
        value["content_review"]["issues"] = issues
        assert Draft202012Validator(schema).is_valid(value) is valid
        envelope = {"action": "finish", "reason": "Review", "arguments": value}
        if valid:
            parse_diagnosis_action("lead", envelope)
        else:
            with pytest.raises(ValueError):
                parse_diagnosis_action("lead", envelope)


@pytest.mark.parametrize(
    "decision,empty,valid",
    [
        ("PROPOSE_EXPERIMENT", True, False),
        ("PROPOSE_EXPERIMENT", False, True),
        ("STOP_INSUFFICIENT_EVIDENCE", True, True),
    ],
)
def test_proposal_hypothesis_requirement_is_present_in_schema(decision, empty, valid):
    plan = {
        "decision": decision,
        "hypothesis": "A tentative explanation",
        "support_refs": ["registered-1"],
        "counter_refs": ["registered-2"],
        "missing_observations": ["Activation responses"],
        "falsification": "No predicted response difference",
        "intervention": "Measure one controlled intervention",
        "controls": ["Same inputs and weights"],
        "measurements": ["Activation response"],
        "expected_patterns": ["Different outcomes under alternatives"],
        "alternatives_remaining": ["Optimization has not been isolated"],
    }
    args = {
        "observation_ids": ["registered-1"],
        "hypotheses": []
        if empty
        else [
            {
                "text": "Pending explanation",
                "observation_ids": ["registered-1"],
                "alternatives": ["Another cause"],
            }
        ],
        "limitations": ["No mechanistic identification"],
        "next_experiment": plan,
    }
    schema = diagnosis_action_schema("analyst")["report"]["arguments"]
    assert Draft202012Validator(schema).is_valid(args) is valid
    envelope = {"action": "report", "reason": "Explain observations", "arguments": args}
    if valid:
        parse_diagnosis_action("analyst", envelope)
    else:
        with pytest.raises(ValueError):
            parse_diagnosis_action("analyst", envelope)
