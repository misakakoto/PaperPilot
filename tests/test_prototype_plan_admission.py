"""Reject unexecutable plans before they lock the agent into a work step."""
import copy

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import new_workspace, plan, complete
from test_prototype_workspace import tasks


@pytest.mark.parametrize('capability,dependencies', [
    ('design', []), ('implement', ['read']), ('validate', ['design']),
    ('review', ['check']), ('review', ['code', 'check']),
    ('validate', ['code', 'code2']),
])
def test_unexecutable_plan_rejected_without_mutating_current_work(capability, dependencies):
    harness = object.__new__(PrototypeHarness)
    work = new_workspace('build')
    plan(work, tasks())
    complete(work, 'read', 'read', {'evidence': 'original'})
    state = {'work': work}
    before = copy.deepcopy(state)
    proposed = tasks() + [
        {'id': 'code2', 'goal': 'alternative code', 'capability': 'implement',
         'depends_on': ['design'], 'acceptance': 'checked'},
        {'id': 'invalid', 'goal': 'invalid dependency contract', 'capability': capability,
         'depends_on': dependencies, 'acceptance': 'checked'},
    ]
    with pytest.raises(ValueError, match='dependency|candidate'):
        harness._apply('run', state, {'action': 'plan', 'reason': 'test',
                                     'arguments': {'tasks': proposed}}, 1)
    assert state == before
    assert 'plan' in harness.allowed_actions(state)


def test_valid_replan_can_follow_rejection_and_preserves_evidence():
    harness = object.__new__(PrototypeHarness)
    work = new_workspace('build')
    plan(work, tasks())
    evidence = complete(work, 'read', 'read', {'evidence': 'original'})
    state = {'work': work}
    revised = tasks()
    revised[1]['goal'] = 'new channels'
    harness._apply('run', state, {'action': 'plan', 'reason': 'revise design',
                                 'arguments': {'tasks': revised}}, 2)
    assert state['work']['tasks']['read']['output'] == evidence
    assert state['plan_requires_work'] is True
    assert 'work' in harness.allowed_actions(state)
