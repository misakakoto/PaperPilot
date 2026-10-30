import copy
import json
import os

import pytest

from paperpilot_agentic.prototype_workspace import new_workspace, plan, complete, context, check_code
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.models import ApprovalStatus


def tasks():
    return [{'id': key, 'goal': key, 'capability': kind, 'depends_on': deps, 'acceptance': 'checked'}
            for key, kind, deps in [('read', 'read', []), ('design', 'design', ['read']),
                                    ('code', 'implement', ['design']), ('check', 'validate', ['code'])]]


def test_replanning_invalidates_only_dependent_work():
    work = new_workspace('build prototype')
    plan(work, tasks())
    for key in ('read', 'design', 'code', 'check'):
        complete(work, key, key, {'version': 1})
    before = copy.deepcopy(work)
    revised = tasks()
    revised[1]['goal'] = 'different channel count'
    plan(work, revised)
    assert work['tasks']['read']['status'] == 'DONE'
    assert all(work['tasks'][k]['status'] == 'PENDING' for k in ('design', 'code', 'check'))
    assert work['artifacts'] == before['artifacts']
    complete(work, 'design', 'design', {'channels': 48})
    assert len(context(work, 'code')['dependencies']) == 1


def test_cycles_and_premature_completion_rejected():
    work = new_workspace('goal')
    rows = tasks()
    rows[0]['depends_on'] = ['check']
    with pytest.raises(ValueError, match='cyclic'):
        plan(work, rows)
    assert work['tasks'] == {}
    plan(work, tasks())
    with pytest.raises(ValueError, match='dependencies'):
        complete(work, 'code', 'implement', {})
    with pytest.raises(ValueError, match='build_model'):
        check_code('x = 1')


class Reader:
    model = 'prototype-offline-fixture'

    def complete(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        rows = payload['tasks']
        if not rows:
            action, args = 'plan', {'tasks': tasks()}
        else:
            key = next(k for k, t in rows.items() if t['status'] != 'DONE' and all(rows[d]['status'] == 'DONE' for d in t['depends_on']))
            content = {'read': 'source', 'design': {'channels': 32},
                       'code': {'source': 'def build_model(config):\n return None\n', 'config': {}},
                       'check': None}[key]
            action, args = 'work', {'task_id': key, 'content': content}
        return LLMResponse(text=json.dumps({'action': action, 'reason': 'offline fixture',
                                           'arguments': args}), model=self.model, finish_reason='stop',
                           prompt_tokens=20, completion_tokens=20)


def stub_environment(monkeypatch):
    if os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1':
        monkeypatch.setattr('paperpilot_agentic.prototype_harness.preflight_environment', lambda: {
            'available': True, 'image_id': 'sha256:'+'a'*64, 'platform': 'linux/amd64',
            'image_reference': 'paperpilot-prototype:cpu'})


def waiting(tmp_path, monkeypatch):
    stub_environment(monkeypatch)
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [
        {'evidence_id': 'source', 'content': 'owned test source'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'prototype-test-001', 'parent', ['source'])['run']['run_id']
    harness.advance(rid)
    return runtime, harness, rid


def test_environment_failure_retains_candidate_and_stops(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])

    def unavailable(*args):
        raise RuntimeError('Docker unavailable')

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', unavailable)
    result = harness.advance(rid)
    assert result['run']['status'] == 'COMPLETED'
    state = harness._load(rid)
    assert state['phase'] == 'PARTIAL'
    assert state['last_observation']['failure_kind'] == 'ENVIRONMENT'
    assert state['work']['tasks']['code']['output'] in state['work']['artifacts']


def test_interrupted_execution_is_not_relaunched(tmp_path, monkeypatch):
    from paperpilot_agentic.models import ToolCallStatus
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    runtime.db.update_tool_call(state['pending']['call'], ToolCallStatus.RUNNING)
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'UNKNOWN'
    count = len(runtime.db.list_tool_calls(rid))
    harness.advance(rid)
    assert len(runtime.db.list_tool_calls(rid)) == count


def test_saved_validation_receipt_recovers_without_execution(tmp_path, monkeypatch):
    from paperpilot_agentic.models import ToolCallStatus
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    runtime.db.update_tool_call(state['pending']['call'], ToolCallStatus.COMPLETED,
                               result={'ok': True, 'output': {'passed': True}})
    assert harness._training(rid, state)
    assert state['work']['tasks']['check']['status'] == 'DONE'
    revised = tasks() + [{'id': 'review', 'goal': 'inspect implementation', 'capability': 'review',
                          'depends_on': ['design', 'code', 'check'], 'acceptance': 'consistent'}]
    plan(state['work'], revised)
    harness._apply(rid, state, {'action': 'work', 'reason': 'review', 'arguments': {
        'task_id': 'review', 'content': {'implementation_consistent': True,
                                       'issues': [], 'limitations': ['offline mock only']}}}, 6)
    harness._apply(rid, state, {'action': 'finish', 'reason': 'done',
                              'arguments': {'summary': 'engineering only'}}, 6)
    assert state['completed'] is True


def test_cancellation_settles_state_without_execution(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    runtime.cancel(rid)
    assert harness._load(rid)['phase'] == 'CANCELLED'
    assert runtime.db.list_approvals(rid)[0]['status'] == 'PENDING'


def test_simple_loop_uses_fixed_plan_and_same_limits(tmp_path, monkeypatch):
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'prototype-simple-001', 'parent', ['source'], mode='simple_loop')['run']['run_id']
    state = harness._load(rid)
    assert 'plan' not in harness.allowed_actions(state)
    assert state['limits']['max_turns'] == 16
    assert {t['capability'] for t in state['work']['tasks'].values()} == {
        'read', 'design', 'implement', 'validate', 'review'}
    with pytest.raises(ValueError, match='unavailable'):
        harness._apply(rid, state, {'action': 'plan', 'reason': 'change', 'arguments': {'tasks': tasks()}}, 1)


def test_durable_run_stops_before_candidate_execution(tmp_path, monkeypatch):
    stub_environment(monkeypatch)
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.db',
                                   llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [
        {'evidence_id': 'source', 'content': 'owned test source'}])
    harness = PrototypeHarness(runtime)
    first = harness.start('build', 'prototype-test-001', 'parent', ['source'])
    rid = first['run']['run_id']
    result = runtime.resume(rid)
    state = harness._load(rid)
    assert result['run']['status'] == 'WAITING_APPROVAL', state.get('stop_reason')
    assert state['work']['tasks']['code']['status'] == 'DONE'
    assert state['work']['tasks']['check']['status'] == 'PENDING'
    assert harness.start('build', 'prototype-test-001', 'parent', ['source'])['run']['run_id'] == rid
    runtime.db.decide_approval(state['pending']['approval'], ApprovalStatus.REJECTED)
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'PARTIAL'
    calls = len(runtime.db.list_tool_calls(rid))
    harness.advance(rid)
    assert len(runtime.db.list_tool_calls(rid)) == calls
