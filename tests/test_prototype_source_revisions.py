import copy

import pytest

from paperpilot_agentic.models import Budget, EvidenceRecord, EvidenceType, RunStatus
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from test_prototype_workspace import Reader, stub_environment


def prepared(tmp_path, monkeypatch, kind=EvidenceType.TEXT):
    stub_environment(monkeypatch)
    runtime = ResearchAgentRuntime(workspace=tmp_path/'workspace', db_path=tmp_path/'runtime.db', llm=Reader())
    parent = runtime.db.create_run('owned reading', Budget(), {})
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    for eid, content in [('paper', 'paper v1'), ('license', 'license v1'), ('paper_v2', 'paper v2')]:
        runtime.db.add_evidence(EvidenceRecord(evidence_id=eid, run_id=parent,
                                             evidence_type=kind if eid.startswith('paper') else EvidenceType.TEXT,
                                             content=content))
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'source-revision-fixture', parent, ['paper', 'license'], mode='simple_loop')['run']['run_id']
    state = harness._load(rid)
    for key, content in [('read0', 'paper'), ('read1', 'license'), ('design', {'hypothesis': 'unverified'}),
                         ('code', {'source': 'def build_model(config):\n return None\n', 'config': {}}), ('check', None)]:
        harness._apply(rid, state, {'action': 'work', 'reason': 'maintained fixture',
                                  'arguments': {'task_id': key, 'content': content}}, 1)
    harness._save(rid, state)
    return runtime, harness, rid, parent


def test_source_replacement_invalidates_only_consumers_and_revokes_unsent_execution(tmp_path, monkeypatch):
    runtime, harness, rid, parent = prepared(tmp_path, monkeypatch)
    before = harness._load(rid)
    run = runtime.db.get_run(rid)
    result = harness.revise_sources(rid, {'paper': 'paper_v2'}, expected_revision=before['work']['revision'])
    after = harness._load(rid)
    assert result['revision_applied'] and after['phase'] == 'READY'
    assert after['work']['task_order'] == before['work']['task_order']
    assert after['source_run_id'] == parent
    assert set(after['sources']) == {'paper_v2', 'license'}
    assert after['required_evidence_ids'] == ['license', 'paper_v2']
    assert after['work']['tasks']['read1'] == before['work']['tasks']['read1']
    for key in ('read0', 'design', 'code', 'check', 'review'):
        assert after['work']['tasks'][key]['status'] == 'PENDING'
        assert after['work']['tasks'][key]['output'] is None
    for key in ('artifacts', 'heads', 'repairs'):
        assert after['work'][key] == before['work'][key]
    for key in ('budget', 'created_at', 'used_llm_calls', 'used_tokens'):
        assert runtime.db.get_run(rid)[key] == run[key]
    assert runtime.db.get_tool_call(before['pending']['call'])['status'] == 'REJECTED'
    history = after['source_revisions'][-1]
    assert history['replacements'] == {'paper': 'paper_v2'}
    assert history['previous_sources'] == before['sources']
    # Restart the adapter, retrieve the registered replacement, then bind its exact version.
    harness = PrototypeHarness(runtime)
    state = harness._load(rid)
    harness._apply(rid, state, {'action': 'work', 'reason': 'replacement',
                              'arguments': {'task_id': 'read0', 'content': 'paper_v2'}}, 2)
    artifact = state['work']['artifacts'][state['work']['tasks']['read0']['output']]
    assert artifact['content']['content'] == 'paper v2'
    assert before['work']['tasks']['read0']['output'] in state['work']['artifacts']
    for key, content in [('design', {'hypothesis': 'replacement-based'}),
                         ('code', {'source': 'def build_model(config):\n return None\n', 'config': {}}), ('check', None)]:
        harness._apply(rid, state, {'action': 'work', 'reason': 'replacement',
                                  'arguments': {'task_id': key, 'content': content}}, 3)
    assert state['pending']['approval'] != before['pending']['approval']
    assert state['pending']['binding']['code'] != before['pending']['binding']['code']
    assert harness._training(rid, state) is False


@pytest.mark.parametrize('replacement', [{}, {'absent': 'paper_v2'}, {'paper': 'absent'},
                                        {'paper': 'license'}, {'paper': 1}, ['paper_v2']])
def test_invalid_source_replacements_leave_state_and_approval_unchanged(tmp_path, monkeypatch, replacement):
    runtime, harness, rid, _ = prepared(tmp_path, monkeypatch)
    before = copy.deepcopy(harness._load(rid))
    with pytest.raises(ValueError):
        harness.revise_sources(rid, replacement, expected_revision=before['work']['revision'])
    assert harness._load(rid) == before
    assert runtime.db.get_approval(before['pending']['approval'])['status'] == 'PENDING'


def test_foreign_source_and_changed_type_cannot_replace_selected_evidence(tmp_path, monkeypatch):
    runtime, harness, rid, parent = prepared(tmp_path, monkeypatch)
    other = runtime.db.create_run('foreign reading', Budget(), {})
    runtime.db.add_evidence(EvidenceRecord(evidence_id='foreign', run_id=other,
        evidence_type=EvidenceType.TEXT, content='foreign evidence'))
    runtime.db.add_evidence(EvidenceRecord(evidence_id='code', run_id=parent,
        evidence_type=EvidenceType.CODE, content='different type'))
    before = harness._load(rid)
    for replacement in ('foreign', 'code'):
        with pytest.raises(ValueError):
            harness.revise_sources(rid, {'paper': replacement}, expected_revision=before['work']['revision'])
        assert harness._load(rid) == before


def test_code_version_replacement_updates_direct_source_gate(tmp_path, monkeypatch):
    _, harness, rid, _ = prepared(tmp_path, monkeypatch, kind=EvidenceType.CODE)
    before = harness._load(rid)
    harness.revise_sources(rid, {'paper': 'paper_v2'}, expected_revision=before['work']['revision'])
    after = harness._load(rid)
    assert after['implementation_source_ids'] == ['paper_v2']
    assert after['work']['tasks']['read1'] == before['work']['tasks']['read1']
    assert after['work']['tasks']['code']['status'] == 'PENDING'


def test_source_revision_workbench_ownership_idempotency_and_export_privacy(tmp_path, monkeypatch):
    import json
    from io import BytesIO
    from zipfile import ZipFile
    from paperpilot_agentic.workbench import ResearchWorkbench
    from paperpilot_agentic.prototype_delivery import build_prototype_bundle
    from test_workbench import Background
    runtime, harness, rid, _ = prepared(tmp_path, monkeypatch)
    background = Background()
    wb = ResearchWorkbench(runtime, background)
    payload = {'request_id': 'source-web', 'action': 'revise_prototype_sources',
               'conversation_id': rid, 'parent_run_id': rid,
               'expected_revision': harness._load(rid)['work']['revision'],
               'source_replacements': {'paper': 'paper_v2'}}
    with pytest.raises(ValueError, match='another conversation'):
        wb.submit(payload | {'request_id': 'foreign-source-web', 'conversation_id': 'foreign'})
    assert wb.submit(payload)['start']
    assert wb.submit(payload)['reused'] and background.submitted == [rid]
    state = harness._load(rid)
    with ZipFile(BytesIO(build_prototype_bundle(state, runtime.db.get_run(rid)))) as archive:
        history = json.loads(archive.read('source_revisions.json'))
        assert 'content' not in history[0]['previous_sources']['paper']
        assert history[0]['previous_sources']['paper']['content_hash']
        assert json.loads(archive.read('manifest.json'))['engineering_validation'] == 'INCOMPLETE_OR_FAILED'
    assert state['source_revisions'][0]['previous_sources']['paper']['content'] == 'paper v1'


@pytest.mark.parametrize('condition', ['unknown', 'unapplied', 'spent', 'rollback', 'legacy', 'stale'])
def test_source_revision_retains_transaction_and_recovery_guards(tmp_path, monkeypatch, condition):
    from paperpilot_agentic.llm_attempts import AttemptLedger
    runtime, harness, rid, _ = prepared(tmp_path, monkeypatch)
    before = harness._load(rid)
    if condition == 'unknown':
        AttemptLedger(runtime.db).begin(rid, stage='source-unknown', model='fixture', messages=[], output_tokens=1,
            required_tokens=2, policy_sha256=None, max_calls=16, max_tokens=100000)
    if condition == 'spent':
        before['turn'] = before['limits']['max_turns']
        harness._save(rid, before)
        harness.revise_sources(rid, {'paper': 'paper_v2'}, expected_revision=before['work']['revision'])
        after = harness._load(rid)
        assert after['phase'] == 'PARTIAL' and after['turn'] == before['turn']
        assert 'paper_v2' in after['sources'] and 'budget exhausted' in after['stop_reason']
        return
    if condition == 'legacy':
        before.pop('source_run_id')
        harness._save(rid, before)
    with runtime.db.transaction() as conn:
        if condition == 'unapplied':
            conn.execute('INSERT INTO research_episode_turns VALUES(?,?,?,?,?,?,?,?)',
                         (rid, 1, 'lead', '{}', 'RETURNED', 0, '{"text":"saved"}', None))
        if condition == 'rollback':
            conn.execute("CREATE TRIGGER fail_source BEFORE UPDATE ON research_episodes BEGIN SELECT RAISE(ABORT,'fixture rollback'); END")
    with pytest.raises(Exception, match={'unknown': 'unknown', 'unapplied': 'saved model turn',
                                        'rollback': 'fixture rollback', 'legacy': 'ownership', 'stale': 'revision'}[condition]):
        harness.revise_sources(rid, {'paper': 'paper_v2'},
                               expected_revision=before['work']['revision']+int(condition == 'stale'))
    assert harness._load(rid) == before
    assert runtime.db.get_approval(before['pending']['approval'])['status'] == 'PENDING'
