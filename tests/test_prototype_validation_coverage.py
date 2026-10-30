import copy

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import complete, new_workspace, plan
from test_prototype_workspace import tasks

POLICY = 'validation-coverage-v1'


def current():
    work = new_workspace('Every implementation has its own independent check')
    plan(work, tasks())
    complete(work, 'read', 'read', {'content': 'retained evidence'})
    return {'work': work, 'plan_policy': POLICY}


def apply(state, rows):
    return PrototypeHarness.__new__(PrototypeHarness)._apply('fixture', state,
        {'action': 'plan', 'reason': 'proposed graph', 'arguments': {'tasks': rows}}, 1)


def extra_code():
    return {'id': 'revision', 'goal': 'repair using execution feedback', 'capability': 'implement',
            'depends_on': ['design', 'check'], 'acceptance': 'tested revised model'}


def extra_check():
    return {'id': 'check_revision', 'goal': 'check revised model', 'capability': 'validate',
            'depends_on': ['revision'], 'acceptance': 'fixed verifier'}


def review(dependencies):
    return {'id': 'review', 'goal': 'review checked implementation', 'capability': 'review',
            'depends_on': dependencies, 'acceptance': 'current code matches design'}


def test_new_policy_rejects_uncovered_repair_without_mutation():
    state = current()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='validation coverage'):
        apply(state, tasks()+[extra_code()])
    assert state == before


@pytest.mark.parametrize('dependencies', [
    ['design', 'revision', 'check'], ['design', 'code', 'check_revision'],
    ['design', 'code', 'revision', 'check']])
def test_review_cannot_mix_code_with_validation_of_another_implementation(dependencies):
    state = current()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='review validation binding'):
        apply(state, tasks()+[extra_code(), extra_check(), review(dependencies)])
    assert state == before


def test_complete_repair_path_is_accepted_and_preserves_existing_evidence():
    state = current()
    evidence = state['work']['tasks']['read']['output']
    rows = tasks()+[extra_code(), extra_check(), review(['design', 'revision', 'check_revision'])]
    apply(state, rows)
    assert state['work']['tasks']['read']['output'] == evidence
    assert state['work']['tasks']['check_revision']['depends_on'] == ['revision']
    assert state['plan_requires_work'] is True


def test_legacy_graph_keeps_its_original_admission_policy():
    state = current()
    state.pop('plan_policy')
    apply(state, tasks()+[extra_code()])
    assert 'revision' in state['work']['tasks']


def test_unknown_plan_policy_is_rejected_without_mutation():
    state = current()
    state['plan_policy'] = 'unknown'
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='plan policy'):
        apply(state, tasks())
    with pytest.raises(ValueError, match='plan policy'):
        PrototypeHarness.__new__(PrototypeHarness)._context('fixture', state)
    assert state == before


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_new_policy_is_recorded_but_old_submission_is_not_migrated(tmp_path, monkeypatch, mode):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from test_prototype_workspace import Reader
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'plan-policy-fixture', 'parent', ['source'], mode=mode)['run']['run_id']
    state = harness._load(rid)
    assert state['plan_policy'] == POLICY
    assert runtime.db.get_run(rid)['metadata']['prototype_plan_policy'] == POLICY
    state.pop('plan_policy')
    harness._save(rid, state)
    before = runtime.db.get_run(rid)
    result = harness.start('build', 'plan-policy-fixture', 'parent', ['source'], mode=mode)
    assert result['run']['run_id'] == rid
    assert 'plan_policy' not in harness._load(rid)
    assert runtime.db.get_run(rid)['budget'] == before['budget']
    assert runtime.db.get_run(rid)['used_llm_calls'] == 0
