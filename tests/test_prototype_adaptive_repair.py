import copy
import json

import pytest

from paperpilot_agentic.prototype_repair import POLICY, repair_options
from paperpilot_agentic.prototype_workspace import complete, plan, task_rows
from test_prototype_completion import finished


def failed():
    harness, state, rows = finished()
    state.update(repair_policy=POLICY, turn=7)
    state['work']['repairs'] = 1
    rows[3]['goal'] = 'actual config load failure'
    plan(state['work'], rows)
    code = state['work']['tasks']['code']['output']
    complete(state['work'], 'check', 'validate', {
        'passed': False, 'validated_code_id': code, 'failure_kind': 'IMPLEMENTATION',
        'error': 'config.base_width must be 24'})
    return harness, state


@pytest.mark.parametrize('target', ['design', 'code'])
def test_adaptive_targeted_repair_keeps_evidence_artifacts_and_budget_counters(target):
    harness, state = failed()
    before = copy.deepcopy(state)
    assert 'repair' in harness.allowed_actions(state)
    harness._apply('run', state, {'action': 'repair', 'reason': 'Use dict config access',
                                 'arguments': {'target_task_id': target}}, 8)
    work = state['work']
    assert work['tasks']['read'] == before['work']['tasks']['read']
    assert work['tasks']['design']['status'] == ('PENDING' if target == 'design' else 'DONE')
    assert all(work['tasks'][key]['status'] == 'PENDING' for key in ('code', 'check', 'review'))
    assert work['artifacts'] == before['work']['artifacts']
    assert work['repairs'] == 1 and state['turn'] == 7
    assert state['repair_history'][0]['failure_bindings'][0]['receipt_artifact_id'] == before['work']['tasks']['check']['output']
    assert not repair_options(state) and 'repair' not in harness.allowed_actions(state)


@pytest.mark.parametrize('case', ['legacy', 'budget', 'environment', 'interrupted', 'unknown', 'stale', 'contract'])
def test_unavailable_repairs_cannot_consume_unrelated_or_uncertain_feedback(case):
    harness, state = failed()
    if case == 'legacy':
        state.pop('repair_policy')
    elif case == 'budget':
        state['work']['repairs'] = 3
    elif case == 'stale':
        rows = task_rows(state['work'])
        rows[2]['goal'] = 'new candidate not executed'
        plan(state['work'], rows)
    else:
        rows = task_rows(state['work'])
        rows[3]['goal'] += ' new receipt'
        plan(state['work'], rows)
        content = {'passed': False, 'validated_code_id': state['work']['tasks']['code']['output'],
                   'failure_kind': {'environment': 'ENVIRONMENT', 'interrupted': 'EXECUTION_INTERRUPTED',
                                    'unknown': 'UNKNOWN', 'contract': 'IMPLEMENTATION'}[case]}
        if case == 'contract':
            state['validation_contract'] = {'version': 'current'}
            content['validation_contract'] = {'version': 'old'}
        complete(state['work'], 'check', 'validate', content)
    assert not repair_options(state)
    assert 'repair' not in harness.allowed_actions(state)


@pytest.mark.parametrize('arguments', [{'target_task_id': 'read'}, {'target_task_id': []},
                                     {'target_task_id': 'code', 'max_parameters': 999999}])
def test_bad_repair_requests_do_not_mutate_state(arguments):
    harness, state = failed()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        harness._apply('run', state, {'action': 'repair', 'reason': 'invalid', 'arguments': arguments}, 8)
    assert state == before


def test_repair_context_exposes_only_current_target_ids_and_preserves_legacy_protocol():
    harness, state = failed()
    _, messages = harness._context('run', state)
    payload = json.loads(messages[-1]['content'])
    assert set(payload['repair_options']) == {'design', 'code'}
    assert 'config is a JSON dictionary' in messages[0]['content']
    state.pop('repair_policy')
    _, messages = harness._context('run', state)
    assert 'repair_options' not in json.loads(messages[-1]['content'])


def test_blocking_review_exposes_the_exact_candidate_and_its_design():
    harness, state, rows = finished()
    state['repair_policy'] = POLICY
    rows[-1]['goal'] = 'identify wrong gate position'
    plan(state['work'], rows)
    complete(state['work'], 'review', 'review', {
        'implementation_consistent': False, 'issues': ['wrong gate position'], 'limitations': []})
    assert set(repair_options(state)) == {'design', 'code'}
