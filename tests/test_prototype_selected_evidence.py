import copy

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import complete, new_workspace, plan
from test_prototype_workspace import tasks

POLICY = 'selected-evidence-v1'


def state():
    work = new_workspace('Use both selected code and license')
    plan(work, tasks())
    return {'work': work, 'source_policy': POLICY, 'required_evidence_ids': ['code', 'license'],
            'sources': {'code': {'evidence_id': 'code', 'content': 'original code'},
                        'license': {'evidence_id': 'license', 'content': 'original license'}}}


def apply(current, action, args):
    return PrototypeHarness.__new__(PrototypeHarness)._apply('fixture', current,
        {'action': action, 'reason': 'fixture', 'arguments': args}, 1)


def rows():
    value = tasks()+[{'id': 'read_license', 'goal': 'read license', 'capability': 'read',
                     'depends_on': [], 'acceptance': 'read exact selected source'}]
    value[1]['depends_on'] = ['read', 'read_license']
    return value


def test_plan_rejects_one_read_slot_claiming_two_sources_atomically():
    current = state()
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='selected evidence'):
        apply(current, 'plan', {'tasks': tasks()})
    assert current == before


def test_read_after_design_cannot_satisfy_its_evidence_requirement():
    current = state()
    value = rows()
    value[1]['depends_on'] = ['read']
    value[-1]['depends_on'] = ['design']
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='selected evidence'):
        apply(current, 'plan', {'tasks': value})
    assert current == before


def test_duplicate_read_does_not_count_as_license_before_design():
    current = state()
    apply(current, 'plan', {'tasks': rows()})
    for key in ('read', 'read_license'):
        apply(current, 'work', {'task_id': key, 'content': 'code'})
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='selected evidence.*license'):
        apply(current, 'work', {'task_id': 'design', 'content': {'hypothesis': 'not yet grounded'}})
    assert current == before


def test_exact_selected_reads_allow_design_and_bind_their_versions():
    current = state()
    apply(current, 'plan', {'tasks': rows()})
    for key, sid in [('read', 'code'), ('read_license', 'license')]:
        apply(current, 'work', {'task_id': key, 'content': sid})
    apply(current, 'work', {'task_id': 'design', 'content': {'hypothesis': 'explicitly untested'}})
    work = current['work']
    design = work['artifacts'][work['tasks']['design']['output']]
    assert set(design['inputs']) == {work['tasks'][k]['output'] for k in ('read', 'read_license')}


def test_legacy_task_keeps_old_source_admission():
    current = state()
    current.pop('source_policy')
    current.pop('required_evidence_ids')
    complete(current['work'], 'read', 'read', current['sources']['code'])
    apply(current, 'work', {'task_id': 'design', 'content': {'hypothesis': 'legacy'}})
    assert current['work']['tasks']['design']['status'] == 'DONE'


def test_tampered_read_artifact_is_rejected_before_design():
    current = state()
    apply(current, 'plan', {'tasks': rows()})
    for key, sid in [('read', 'code'), ('read_license', 'license')]:
        apply(current, 'work', {'task_id': key, 'content': sid})
    artifact = current['work']['artifacts'][current['work']['tasks']['read_license']['output']]
    artifact['content']['content'] = 'changed'
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='identity'):
        apply(current, 'work', {'task_id': 'design', 'content': {}})
    assert current == before


def test_transitive_reads_count_without_forcing_all_originals_into_direct_context():
    current = state()
    value = rows()
    value[-1]['depends_on'] = ['read']
    value[1]['depends_on'] = ['read_license']
    apply(current, 'plan', {'tasks': value})
    for key, sid in [('read', 'code'), ('read_license', 'license')]:
        apply(current, 'work', {'task_id': key, 'content': sid})
    apply(current, 'work', {'task_id': 'design', 'content': {'hypothesis': 'not measured'}})
    work = current['work']
    artifact = work['artifacts'][work['tasks']['design']['output']]
    assert artifact['inputs'] == [work['tasks']['read_license']['output']]


def test_omitted_retained_tasks_return_actionable_full_replacement_error():
    current = state()
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='full replacement.*read'):
        apply(current, 'plan', {'tasks': tasks()[1:]})
    assert current == before


@pytest.mark.parametrize('change', [{'source_policy': 'unknown'}, {'required_evidence_ids': []},
                                  {'required_evidence_ids': ['code']},
                                  {'required_evidence_ids': ['code', 'code']}])
def test_unknown_policy_or_changed_required_selection_is_rejected(change):
    current = state()
    current.update(change)
    before = copy.deepcopy(current)
    with pytest.raises(ValueError, match='source policy|selected evidence'):
        apply(current, 'plan', {'tasks': rows()})
    with pytest.raises(ValueError, match='source policy|selected evidence'):
        PrototypeHarness.__new__(PrototypeHarness)._context('fixture', current)
    assert current == before


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_new_task_registers_selection_without_migrating_resubmitted_legacy_task(tmp_path, monkeypatch, mode):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from test_prototype_workspace import Reader
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: list(state()['sources'].values()))
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'evidence-policy', 'parent', ['code', 'license'], mode=mode)['run']['run_id']
    current = harness._load(rid)
    assert current['source_policy'] == POLICY
    assert current['required_evidence_ids'] == ['code', 'license']
    assert runtime.db.get_run(rid)['metadata']['prototype_source_policy'] == POLICY
    current.pop('source_policy')
    current.pop('required_evidence_ids')
    harness._save(rid, current)
    before = runtime.db.get_run(rid)
    result = harness.start('build', 'evidence-policy', 'parent', ['code', 'license'], mode=mode)
    assert result['run']['run_id'] == rid
    assert 'source_policy' not in harness._load(rid)
    assert runtime.db.get_run(rid)['budget'] == before['budget']
    assert runtime.db.get_run(rid)['used_llm_calls'] == 0
