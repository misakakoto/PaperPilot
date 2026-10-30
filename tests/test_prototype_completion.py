import json

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import new_workspace, plan, complete


def finished():
    work = new_workspace('Bounded prototype')
    rows = [
        {'id': key, 'goal': key, 'capability': capability, 'depends_on': deps, 'acceptance': 'exact'}
        for key, capability, deps in [('read', 'read', []), ('design', 'design', ['read']),
                                     ('code', 'implement', ['design']), ('check', 'validate', ['code']),
                                     ('review', 'review', ['design', 'code', 'check'])]]
    plan(work, rows)
    complete(work, 'read', 'read', {'content': 'public source'})
    complete(work, 'design', 'design', {'hypothesis': 'unmeasured design'})
    code = complete(work, 'code', 'implement', {'source': 'DO_NOT_REPLAY_FULL_SOURCE', 'config': {}})
    complete(work, 'check', 'validate', {'passed': True, 'validated_code_id': code,
                                       'execution': {'metrics': {'parameters': 123, 'checks': [1, 2, 3]}}})
    complete(work, 'review', 'review', {'implementation_consistent': True, 'issues': [],
                                       'limitations': ['No training or accuracy evaluation']})
    harness = PrototypeHarness.__new__(PrototypeHarness)
    state = {'work': work, 'sources': {}, 'mode': 'adaptive'}
    return harness, state, rows


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_no_work_action_when_all_tasks_are_done(mode):
    harness, state, _ = finished()
    state['mode'] = mode
    assert 'work' not in harness.allowed_actions(state)
    assert 'finish' in harness.allowed_actions(state)


def test_finish_context_includes_current_review_metrics_and_exact_artifact_ids():
    harness, state, _ = finished()
    _, messages = harness._context('fixture', state)
    payload = json.loads(messages[-1]['content'])
    result = payload['completion_context']
    assert result['review']['content']['limitations'] == ['No training or accuracy evaluation']
    assert result['check']['content']['execution']['metrics']['parameters'] == 123
    assert result['code']['artifact_id'] == state['work']['tasks']['code']['output']
    assert 'DO_NOT_REPLAY_FULL_SOURCE' not in messages[-1]['content']


def test_replanning_never_presents_stale_review_as_current():
    harness, state, rows = finished()
    old_review = state['work']['tasks']['review']['output']
    rows[2]['goal'] = 'change implementation'
    plan(state['work'], rows)
    _, messages = harness._context('fixture', state)
    payload = json.loads(messages[-1]['content'])
    assert 'work' in harness.allowed_actions(state)
    assert 'review' not in payload.get('completion_context', {})
    assert old_review in state['work']['artifacts']
