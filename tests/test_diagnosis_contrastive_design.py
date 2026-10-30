import copy
import json

import pytest
from jsonschema import Draft202012Validator

from paperpilot_agentic.research_contracts import diagnosis_action_schema, parse_diagnosis_action
from test_diagnosis_comparison_design import design, proposal, report, state


def contrastive_plan():
    plan = proposal()
    plan['comparison_design'] = design()
    plan['comparison_design']['outcomes'][1]['effect'] = 'WEAKENS'
    plan['comparison_design']['identification'] = {
        'question': 'Does the operator affect this fixed checkpoint response under the specified controls?',
        'scope': 'FIXED_CHECKPOINT_RESPONSE',
        'intervention_checkpoint': 'baseline',
        'alternatives': [
            {'hypothesis': 'A response difference exceeds the prespecified tolerance',
             'expected_outcomes': ['COMPATIBLE', 'EXCLUDED'],
             'assumptions': ['The hypothetical outcomes exhaust the prespecified tolerance test.']},
            {'hypothesis': 'The response difference stays within the prespecified tolerance',
             'expected_outcomes': ['EXCLUDED', 'COMPATIBLE'],
             'assumptions': ['The same weights, samples and tolerance are used for both operators.']},
        ],
    }
    plan['comparison_design']['outcomes'][1]['observation'] = 'The prespecified response is absent'
    return plan


def v4_state():
    return state() | {'content_review_protocol': 'scientific-review-v4'}


def test_unexecuted_bounded_plan_crosses_same_schema_and_parser_interface():
    value = report(contrastive_plan())
    original = copy.deepcopy(value)
    schema = diagnosis_action_schema('analyst', v4_state())['report']['arguments']
    assert Draft202012Validator(schema).is_valid(value['arguments'])
    assert parse_diagnosis_action('analyst', value, v4_state()) == original


def test_missing_identification_cannot_be_approved_by_adding_outcome_labels():
    plan = proposal() | {'comparison_design': design()}
    with pytest.raises(ValueError, match='identification'):
        parse_diagnosis_action('analyst', report(plan), v4_state())
    # Historical v3 is deliberately unchanged.
    assert parse_diagnosis_action('analyst', report(plan), state()) == report(plan)


@pytest.mark.parametrize('change,match', [
    ('compatible', 'distinguish'),
    ('unknown', 'distinguish'),
    ('wrong_scope', 'scope'),
    ('checkpoint', 'checkpoint'),
    ('missing_outcome', 'every outcome'),
    ('bare_exclusion', 'assumptions'),
    ('duplicate', 'unique'),
])
def test_invalid_comparison_claims_fail_at_the_real_report_parser(change, match):
    plan = contrastive_plan()
    item = plan['comparison_design']['identification']
    if change in {'compatible', 'unknown'}:
        for alternative in item['alternatives']:
            alternative['expected_outcomes'] = [
                'COMPATIBLE' if change == 'compatible' else 'UNKNOWN'] * 2
    elif change == 'wrong_scope':
        item['scope'] = 'TRAINING_INTERVENTION_EFFECT'
    elif change == 'checkpoint':
        item['intervention_checkpoint'] = 'NOT_APPLICABLE'
    elif change == 'missing_outcome':
        item['alternatives'][0]['expected_outcomes'].append('UNKNOWN')
    elif change == 'bare_exclusion':
        item['alternatives'][0]['assumptions'] = []
    else:
        item['alternatives'][1]['hypothesis'] = item['alternatives'][0]['hypothesis']
    with pytest.raises(ValueError, match=match):
        parse_diagnosis_action('analyst', report(plan), v4_state())


def test_non_discriminating_exploration_is_allowed_when_it_claims_no_exclusion():
    plan = contrastive_plan()
    for row in plan['comparison_design']['identification']['alternatives']:
        row['expected_outcomes'] = ['COMPATIBLE', 'UNKNOWN']
        row['assumptions'] = []
    for outcome in plan['comparison_design']['outcomes']:
        outcome['effect'] = 'INCONCLUSIVE'
    assert parse_diagnosis_action('analyst', report(plan), v4_state()) == report(plan)


def test_evidence_gap_stop_does_not_require_an_invented_experiment_matrix():
    value = report(proposal() | {'decision': 'STOP_INSUFFICIENT_EVIDENCE'})
    value['arguments']['hypotheses'] = []
    assert parse_diagnosis_action('analyst', value, v4_state()) == value


def test_fixed_checkpoint_observational_diagnostic_is_not_an_operator_intervention():
    plan = contrastive_plan()
    design = plan['comparison_design']
    design['mode'] = 'OBSERVATIONAL_DIAGNOSTIC'
    design['identification']['intervention_checkpoint'] = 'NOT_APPLICABLE'
    for item in design['identification']['alternatives']:
        item['expected_outcomes'] = ['COMPATIBLE', 'UNKNOWN']
        item['assumptions'] = []
    for outcome in design['outcomes']:
        outcome['effect'] = 'INCONCLUSIVE'
    assert parse_diagnosis_action('analyst', report(plan), v4_state()) == report(plan)


def test_real_harness_persists_declared_matrix_and_recovers_without_calls(tmp_path):
    from paperpilot_agentic.providers import LLMResponse
    from test_diagnosis_content_review import revised_provider
    from test_diagnosis_scoped_review import scoped_review
    from test_imported_diagnosis_harness import setup

    runtime, harness, reader, binding = setup(tmp_path)
    normal = revised_provider(reader)

    def complete(messages, **kwargs):
        response = normal(messages, **kwargs)
        value = json.loads(response.text)
        if value['action'] == 'report':
            value['arguments']['next_experiment']['comparison_design'] = contrastive_plan()['comparison_design']
        elif value['action'] == 'finish':
            value['arguments']['content_review'] = scoped_review()
        return LLMResponse(text=json.dumps(value), model=reader.model,
                           finish_reason='stop', prompt_tokens=10, completion_tokens=10)

    reader.complete = complete
    rid = harness.start_imported_diagnosis(
        binding['bundle_id'], 'contrastive-plan-001', comparison_mode='summary',
        report_protocol='metric-facts-v3', content_review_protocol='scientific-review-v4',
    )['run']['run_id']
    result = harness.advance(rid)
    assert result['research_episode']['phase'] == 'COMPLETED'
    review_task = json.loads(next(m['content'] for m in reader.messages[-1] if m['role'] == 'user'))
    audit = review_task['scientific_review_context']['basis']['scope/identification_structure']
    assert audit['assurance'] == 'DECLARATIONS_CHECKED_NOT_CAUSAL_PROOF'
    assert audit['outcomes'][0]['compatible'] == [0]
    assert audit['outcomes'][0]['excluded'] == [1]
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
    assert result['research_episode']['new_gpu_starts'] == 0
