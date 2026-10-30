import copy

import pytest

from paperpilot_agentic.prototype_workspace import plan, complete
from test_prototype_completion import finished


def failed():
    harness, state, rows = finished()
    state.update(mode='simple_loop', turn=7)
    state['work']['repairs'] = 1
    rows[3]['goal'] = 'measured resource failure'
    plan(state['work'], rows)
    code = state['work']['tasks']['code']['output']
    complete(state['work'], 'check', 'validate', {
        'passed': False, 'validated_code_id': code, 'failure_kind': 'DESIGN_CONSTRAINT'})
    return harness, state


@pytest.mark.parametrize('target', ['design', 'implement'])
def test_simple_loop_can_choose_design_or_implementation_repair(target):
    harness, state = failed()
    before = copy.deepcopy(state)
    assert 'repair' in harness.allowed_actions(state)
    harness._apply('run', state, {'action': 'repair', 'reason': 'reduce measured resource use',
                                 'arguments': {'target': target}}, 8)
    work = state['work']
    assert work['tasks']['read'] == before['work']['tasks']['read']
    assert work['tasks']['design']['status'] == ('PENDING' if target == 'design' else 'DONE')
    assert all(work['tasks'][key]['status'] == 'PENDING' for key in ('code', 'check', 'review'))
    assert work['artifacts'] == before['work']['artifacts']
    assert work['repairs'] == 1 and state['turn'] == 7
    assert work['revision'] == before['work']['revision'] + 1
    assert 'repair' not in harness.allowed_actions(state)
    assert state['plan_requires_work'] is True


def test_simple_loop_can_repair_blocking_review_after_engineering_pass():
    harness, state, rows = finished()
    state['mode'] = 'simple_loop'
    rows[-1]['goal'] = 'review identifies missing structure'
    plan(state['work'], rows)
    complete(state['work'], 'review', 'review', {
        'implementation_consistent': False, 'issues': ['Declared structure absent'], 'limitations': []})
    assert 'repair' in harness.allowed_actions(state)


@pytest.mark.parametrize('condition', ['successful', 'budget', 'adaptive', 'pending'])
def test_repair_is_unavailable_without_current_failure_or_remaining_attempts(condition):
    harness, state = failed()
    if condition == 'successful':
        harness, state, _ = finished()
        state['mode'] = 'simple_loop'
    elif condition == 'budget':
        state['work']['repairs'] = 3
    elif condition == 'adaptive':
        state['mode'] = 'adaptive'
    else:
        rows = [{k: t[k] for k in ('id', 'goal', 'capability', 'depends_on', 'acceptance')}
                for t in state['work']['tasks'].values()]
        rows[2]['goal'] = 'new implementation not checked'
        plan(state['work'], rows)
    assert 'repair' not in harness.allowed_actions(state)


def test_simple_loop_repair_cannot_rewrite_evidence_or_limits():
    harness, state = failed()
    before = copy.deepcopy(state)
    for arguments in ({'target': 'read'}, {'target': 'design', 'max_parameters': 999999999}):
        with pytest.raises(ValueError):
            harness._apply('run', state, {'action': 'repair', 'reason': 'invalid request',
                                         'arguments': arguments}, 8)
        assert state == before


@pytest.mark.parametrize('kind', ['UNKNOWN', 'RESOURCE_LIMIT', None, 'ENVIRONMENT', 'EXECUTION_INTERRUPTED'])
def test_simple_loop_cannot_treat_uncertain_execution_as_a_code_defect(kind):
    harness,state=failed()
    rows=[{k:t[k] for k in ('id','goal','capability','depends_on','acceptance')}
          for t in state['work']['tasks'].values()]
    rows[3]['goal']='execution attribution is not established'
    plan(state['work'],rows)
    complete(state['work'],'check','validate',{'passed':False,
        'validated_code_id':state['work']['tasks']['code']['output'],'failure_kind':kind})
    before=copy.deepcopy(state)
    assert 'repair' not in harness.allowed_actions(state)
    with pytest.raises(ValueError):
        harness._apply('run',state,{'action':'repair','reason':'guess a fix',
            'arguments':{'target':'implement'}},8)
    assert state==before


def test_simple_loop_cannot_repair_using_a_receipt_for_an_old_validation_contract():
    harness,state=failed()
    state['validation_contract']={'kind':'unbound','max_parameters':1000}
    rows=[{k:t[k] for k in ('id','goal','capability','depends_on','acceptance')}
          for t in state['work']['tasks'].values()]
    rows[3]['goal']='old constraints are not current validation evidence'
    plan(state['work'],rows)
    complete(state['work'],'check','validate',{'passed':False,
        'validated_code_id':state['work']['tasks']['code']['output'],'failure_kind':'IMPLEMENTATION',
        'validation_contract':{'kind':'unbound','max_parameters':2000}})
    assert 'repair' not in harness.allowed_actions(state)


@pytest.mark.parametrize('kind', ['IMPLEMENTATION', 'DESIGN_CONSTRAINT'])
def test_current_attributed_failure_still_allows_a_bounded_repair(kind):
    harness,state=failed()
    state['validation_contract']={'kind':'unbound','max_parameters':1000}
    rows=[{k:t[k] for k in ('id','goal','capability','depends_on','acceptance')}
          for t in state['work']['tasks'].values()]
    rows[3]['goal']='confirmed failure under current requirements'
    plan(state['work'],rows)
    complete(state['work'],'check','validate',{'passed':False,
        'validated_code_id':state['work']['tasks']['code']['output'],'failure_kind':kind,
        'validation_contract':copy.deepcopy(state['validation_contract'])})
    before=copy.deepcopy(state['work'])
    assert 'repair' in harness.allowed_actions(state)
    harness._apply('run',state,{'action':'repair','reason':'address the attributed failure',
        'arguments':{'target':'implement'}},8)
    assert state['work']['tasks']['design']==before['tasks']['design']
    assert state['work']['tasks']['code']['status']=='PENDING'
    assert state['work']['tasks']['check']['status']=='PENDING'
    assert state['work']['artifacts']==before['artifacts']


@pytest.mark.parametrize('mode', ['simple_loop', 'adaptive'])
def test_fixed_loop_design_repair_requires_new_approval_and_finishes_with_same_budget(tmp_path, monkeypatch, mode):
    import json
    from types import SimpleNamespace
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.providers import LLMResponse
    from test_prototype_workspace import stub_environment

    class Coder:
        model = 'maintained-simple-loop-fixture'
        designs = 0

        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if not payload['tasks']:
                action = 'plan'
                arguments = {'tasks': [
                    {'id': key, 'goal': key, 'capability': kind, 'depends_on': deps, 'acceptance': 'fixture'}
                    for key, kind, deps in [('read0', 'read', []), ('design', 'design', ['read0']),
                        ('code', 'implement', ['design']), ('check', 'validate', ['code']),
                        ('review', 'review', ['design', 'code', 'check'])]]}
            elif 'repair' in payload['available_actions']:
                action, arguments = 'repair', ({'target_task_id': 'design'} if mode == 'adaptive'
                                                else {'target': 'design'})
            elif payload['ready_contexts']:
                key, task = next(iter(payload['ready_contexts'].items()))
                kind = task['task']['capability']
                if kind == 'design':
                    self.designs += 1
                content = {'read': 'source', 'design': {'version': self.designs},
                    'implement': {'source': f'def build_model(config):\n return None # version {self.designs}\n',
                                  'config': {}}, 'validate': None,
                    'review': {'implementation_consistent': True, 'issues': [], 'limitations': ['fixture only']}}[kind]
                action, arguments = 'work', {'task_id': key, 'content': content}
            else:
                action, arguments = 'finish', {'summary': 'Maintained fixed-loop fixture completed'}
            return LLMResponse(text=json.dumps({'action': action, 'reason': 'maintained fixture',
                'arguments': arguments}), model=self.model, finish_reason='stop', prompt_tokens=20, completion_tokens=20)

    stub_environment(monkeypatch)
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.db', llm=Coder())
    reserve_classes = []
    original_complete = runtime.llm_runtime.complete

    def observe_complete(*args, **kwargs):
        reserve_classes.append(kwargs['policy'].reserve_class)
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(runtime.llm_runtime, 'complete', observe_complete)
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'public fixture'}])
    executions = []

    def execute(*args):
        executions.append(1)
        passed = len(executions) == 2
        return SimpleNamespace(status='COMPLETED', exit_code=0, timed_out=False,
            metrics={'passed': passed, 'failure_kind': None if passed else 'DESIGN_CONSTRAINT'},
            to_dict=lambda: {'status': 'COMPLETED'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', execute)
    harness = PrototypeHarness(runtime)
    rid = harness.start('bounded model', 'fixed-loop-design-repair', 'parent', ['source'], mode=mode)['run']['run_id']
    harness.advance(rid)
    initial = harness._load(rid)
    budget = copy.deepcopy(runtime.db.get_run(rid)['budget'])
    runtime.approve(initial['pending']['approval'])
    harness.advance(rid)
    revised = harness._load(rid)
    assert revised['phase'] == 'WAITING_APPROVAL' and len(executions) == 1
    assert revised['pending']['approval'] != initial['pending']['approval']
    assert revised['pending']['binding']['code'] != initial['pending']['binding']['code']
    assert revised['work']['tasks']['read0']['output'] == initial['work']['tasks']['read0']['output']
    assert revised['work']['repairs'] == 2
    runtime.approve(revised['pending']['approval'])
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'COMPLETED'
    assert len(executions) == 2
    assert runtime.db.get_run(rid)['budget'] == budget
    expected_calls = 11 if mode == 'adaptive' else 10
    assert runtime.db.get_run(rid)['used_llm_calls'] == expected_calls
    assert runtime.db.get_run(rid)['used_tokens'] == expected_calls * 40
    assert reserve_classes[:4] == ['general'] * 4
    assert 'repair' in reserve_classes and reserve_classes[-2:] == ['synthesis', 'synthesis']
