import copy

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import complete, plan, new_workspace
from test_prototype_workspace import tasks


def state():
    work = new_workspace('modify original code')
    plan(work, tasks())
    return {'work': work, 'implementation_source_ids': ['original-code'], 'sources': {'original-code': {
        'evidence_id': 'original-code', 'evidence_type': 'CODE', 'content': 'original source'}}}


@pytest.mark.parametrize('stage', ['implement', 'review'])
def test_plan_rejects_code_work_without_original_source_context_atomically(stage):
    harness = object.__new__(PrototypeHarness)
    current = state()
    before = copy.deepcopy(current)
    rows = tasks()+[{'id': 'review', 'goal': 'review', 'capability': 'review',
                     'depends_on': ['design', 'code', 'check', 'read'], 'acceptance': 'source consistent'}]
    rows[2]['depends_on'] += ['read']
    next(t for t in rows if t['capability'] == stage)['depends_on'].remove('read')
    with pytest.raises(ValueError, match='source'):
        harness._apply('run', current, {'action': 'plan', 'reason': 'change',
                                       'arguments': {'tasks': rows}}, 1)
    assert current == before


def test_generation_cannot_silently_use_design_instead_of_original_code():
    harness = object.__new__(PrototypeHarness)
    current = state()
    complete(current['work'], 'read', 'read', {'evidence_id': 'original-code', 'content': 'original source'})
    complete(current['work'], 'design', 'design', {'proposal': 'new gate'})
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='source'):
        harness._apply('run', current, {'action': 'work', 'reason': 'implement', 'arguments': {
            'task_id': 'code', 'content': {'source': 'def build_model(config): return None', 'config': {}}}}, 4)
    assert current == before


def test_resolved_wrong_read_does_not_satisfy_code_source_requirement():
    harness = object.__new__(PrototypeHarness)
    current = state()
    complete(current['work'], 'read', 'read', {'evidence_id': 'unrelated-evidence'})
    rows = tasks()
    rows[2]['depends_on'] += ['read']
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='source'):
        harness._apply('run', current, {'action': 'plan', 'reason': 'copy source',
                                       'arguments': {'tasks': rows}}, 3)
    assert current == before


def test_exact_source_version_is_bound_to_generated_artifact():
    harness = object.__new__(PrototypeHarness)
    current = state()
    rows = tasks()
    rows[2]['depends_on'] += ['read']
    harness._apply('run', current, {'action': 'plan', 'reason': 'source-bound code',
                                   'arguments': {'tasks': rows}}, 1)
    original = complete(current['work'], 'read', 'read', {'evidence_id': 'original-code', 'content': 'exact version'})
    complete(current['work'], 'design', 'design', {'proposal': 'gate'})
    harness._apply('run', current, {'action': 'work', 'reason': 'code', 'arguments': {
        'task_id': 'code', 'content': {'source': 'def build_model(config): return None', 'config': {}}}}, 4)
    code = current['work']['tasks']['code']['output']
    assert original in current['work']['artifacts'][code]['inputs']


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_new_code_tasks_register_source_policy_without_model_calls(tmp_path, monkeypatch, mode):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from test_prototype_workspace import Reader
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [
        {'evidence_id': 'original-code', 'evidence_type': 'CODE', 'content': 'source'},
        {'evidence_id': 'notes', 'evidence_type': 'TEXT', 'content': 'notes'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('copy source', 'source-context-'+mode, 'parent', ['original-code', 'notes'],
                        mode=mode)['run']['run_id']
    current = harness._load(rid)
    assert current['implementation_source_ids'] == ['original-code']
    assert runtime.db.get_run(rid)['used_llm_calls'] == 0
    if mode == 'simple_loop':
        assert 'read0' in current['work']['tasks']['code']['depends_on']
        assert 'read0' in current['work']['tasks']['review']['depends_on']
