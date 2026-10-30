import copy
import pytest
from paperpilot_agentic.research_contracts import parse_diagnosis_action


def proposal():
    return {
        "decision": "PROPOSE_EXPERIMENT",
        "hypothesis": "A baseline response may differ under the operator intervention",
        "support_refs": ["paired/0/delta/f1"],
        "counter_refs": ["paired/0/delta/recall"],
        "missing_observations": ["Layer response under each operator"],
        "falsification": "Describe a discriminating outcome",
        "intervention": "Compare the registered operators",
        "controls": ["Freeze the declared weights and inputs"],
        "measurements": ["Record layer response"],
        "expected_patterns": ["Different responses remain possible"],
        "alternatives_remaining": ["Optimization dynamics remain unmeasured"],
    }


def design():
    return {
        "mode": "SHARED_WEIGHTS_OPERATOR",
        "target": "paired_difference",
        "measured_variants": ["baseline", "candidate"],
        "weight_control": "SHARED_CHECKPOINT",
        "outcomes": [
            {
                "observation": "A prespecified response difference is observed",
                "effect": "SUPPORTS",
                "alternatives_remaining": [
                    "This does not establish the cause of the historical training effect"
                ],
            },
            {
                "observation": "Measurement noise prevents interpretation",
                "effect": "INCONCLUSIVE",
                "alternatives_remaining": ["The mechanism remains untested"],
            },
        ],
    }


def report(plan):
    return {
        "action": "report",
        "reason": "Propose a bounded comparison",
        "arguments": {
            "observation_ids": ["pattern/paired/0/f1"],
            "hypotheses": [
                {
                    "text": "A conditional mechanism",
                    "observation_ids": ["pattern/paired/0/f1"],
                    "alternatives": ["Unmeasured dynamics"],
                }
            ],
            "limitations": ["No activation observations exist yet"],
            "next_experiment": plan,
        },
    }


def state(scoped=True):
    return {
        "content_review_protocol": "scientific-review-v3" if scoped else "scientific-review-v2",
        "comparison_mode": "summary",
        "analysis": None,
        "observations": {"paired": {}},
    }


def test_scoped_proposal_cannot_hide_comparison_target_in_free_text():
    with pytest.raises(ValueError, match="comparison design"):
        parse_diagnosis_action("analyst", report(proposal()), state())


def test_historical_flat_proposal_contract_remains_readable():
    original = report(proposal())
    assert parse_diagnosis_action("analyst", copy.deepcopy(original), state(False)) == original


@pytest.mark.parametrize(
    "change,match",
    [
        (
            {
                "target": "baseline",
                "mode": "OBSERVATIONAL_DIAGNOSTIC",
                "measured_variants": ["candidate"],
            },
            "declared target",
        ),
        (
            {"mode": "OBSERVATIONAL_DIAGNOSTIC", "measured_variants": ["candidate"]},
            "declared target",
        ),
        ({"weight_control": "PER_VARIANT_CHECKPOINT"}, "Weight control"),
        ({"mode": "SEPARATELY_TRAINED_MODELS"}, "Weight control"),
        ({"measured_variants": ["baseline", "baseline"]}, "declared target"),
    ],
)
def test_scope_or_weight_mismatch_is_rejected(change, match):
    plan = proposal()
    plan["comparison_design"] = design() | change
    with pytest.raises(ValueError, match=match):
        parse_diagnosis_action("analyst", report(plan), state())


def test_valid_design_is_preserved_and_schema_matches_parser():
    from jsonschema import Draft202012Validator
    from paperpilot_agentic.research_contracts import diagnosis_action_schema

    plan = proposal()
    plan["comparison_design"] = design()
    value = report(plan)
    original = copy.deepcopy(value)
    schema = diagnosis_action_schema("analyst", state())["report"]["arguments"]
    assert Draft202012Validator(schema).is_valid(value["arguments"])
    assert parse_diagnosis_action("analyst", value, state()) == original
    assert value == original
    from paperpilot_agentic.research_review import scientific_review_context

    scoped_state = state()
    scoped_state["analysis"] = {
        "hypotheses": value["arguments"]["hypotheses"],
        "experiment_contract": plan,
    }
    registry = scientific_review_context(scoped_state)["review_reference_catalog"]
    assert (
        registry["analysis/experiment_contract/comparison_design/outcomes/0/observation"]
        == "MODEL_CLAIM"
    )
    assert registry["analysis/experiment_contract/comparison_design/target"] == "MODEL_CLAIM"
    missing = report(proposal())
    assert not Draft202012Validator(schema).is_valid(missing["arguments"])
    missing["arguments"]["next_experiment"]["comparison_design"] = None
    assert not Draft202012Validator(schema).is_valid(missing["arguments"])


def test_evidence_limited_stop_does_not_require_fabricated_design():
    plan = proposal()
    plan["decision"] = "STOP_INSUFFICIENT_EVIDENCE"
    value = report(plan)
    value["arguments"]["hypotheses"] = []
    parsed = parse_diagnosis_action("analyst", value, state())
    assert "comparison_design" not in parsed["arguments"]["next_experiment"]


def test_partial_measurement_is_permitted_for_a_matching_observational_question():
    plan = proposal()
    plan["comparison_design"] = design() | {
        "mode": "OBSERVATIONAL_DIAGNOSTIC",
        "target": "candidate",
        "measured_variants": ["candidate"],
        "weight_control": "PER_VARIANT_CHECKPOINT",
    }
    assert (
        parse_diagnosis_action("analyst", report(plan), state())["arguments"]["next_experiment"][
            "comparison_design"
        ]["target"]
        == "candidate"
    )


def test_legacy_projection_checks_design_but_does_not_upgrade_scientific_assurance():
    from paperpilot_agentic.diagnostic_facts import validate_experiment

    plan = proposal()
    plan["comparison_design"] = design()
    projected = validate_experiment(plan)
    assert projected["hypothesis"] == plan["hypothesis"]
    assert "comparison_design" not in projected
    plan["comparison_design"]["target"] = "baseline"
    plan["comparison_design"]["measured_variants"] = ["candidate"]
    with pytest.raises(ValueError, match="declared target"):
        validate_experiment(plan)


def test_harness_rejects_insufficient_design_scope_and_retains_host_observations(tmp_path):
    import json
    from paperpilot_agentic.providers import LLMResponse
    from test_diagnosis_content_review import revised_provider
    from test_imported_diagnosis_harness import setup

    _, harness, reader, binding = setup(tmp_path)
    normal = revised_provider(reader)

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value["action"] == "report":
            value["arguments"]["next_experiment"]["comparison_design"] = design() | {
                "mode": "OBSERVATIONAL_DIAGNOSTIC",
                "target": "baseline",
                "measured_variants": ["candidate"],
            }
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
        "design-scope-reject-001",
        comparison_mode="summary",
        report_protocol="metric-facts-v3",
        content_review_protocol="scientific-review-v3",
    )["run"]["run_id"]
    result = harness.advance(rid)
    episode = result["research_episode"]
    assert episode["phase"] == "PARTIAL"
    assert "declared target" in episode["stop_reason"]
    assert episode["analysis"] is None
    assert episode["retained_host_observations"]
    assert "宿主生成的观测" in result["final_answer"]["answer"]
    assert episode["new_gpu_starts"] == 0
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
