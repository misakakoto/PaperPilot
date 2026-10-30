import copy

import pytest

from test_prototype_workspace import waiting
from test_prototype_revisions import completed


@pytest.mark.parametrize('fixture', [waiting, completed])
def test_goal_revision_preserves_sources_history_and_budget_but_invalidates_results(tmp_path, monkeypatch, fixture):
    runtime, harness, rid = fixture(tmp_path, monkeypatch)
    before = harness._load(rid)
    run = runtime.db.get_run(rid)
    result = harness.revise_goal(rid, 'Keep SNUNet and add a spatial gate',
                                expected_revision=before['work']['revision'])
    after = harness._load(rid)
    assert result['revision_applied'] and after['phase'] == 'READY'
    assert after['work']['task_order'] == before['work']['task_order']
    assert after['goal'] == after['work']['goal'] == 'Keep SNUNet and add a spatial gate'
    assert after['work']['tasks']['read'] == before['work']['tasks']['read']
    assert all(t['status'] == 'PENDING' and t['output'] is None
               for t in after['work']['tasks'].values() if t['capability'] != 'read')
    for key in ('artifacts', 'heads', 'repairs'):
        assert after['work'][key] == before['work'][key]
    for key in ('sources', 'validation_contract', 'turn', 'model_protocol'):
        assert after[key] == before[key]
    for key in ('created_at', 'budget', 'used_llm_calls', 'used_tokens', 'question'):
        assert runtime.db.get_run(rid)[key] == run[key]
    assert after['goal_revisions'][-1]['previous_goal'] == before['goal']
    assert after['goal_revisions'][-1]['previous_tasks'] == before['work']['tasks']
    assert runtime.db.get_run(rid)['metadata']['final_answer'] is None
    if 'pending' in before and fixture is waiting:
        assert runtime.db.get_tool_call(before['pending']['call'])['status'] == 'REJECTED'
    assert 'pending' not in after
    with pytest.raises(RuntimeError, match='revision'):
        harness.revise_goal(rid, 'Another goal', expected_revision=before['work']['revision'])


def test_same_goal_noop_and_invalid_goal_are_atomic(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before = copy.deepcopy(harness._load(rid))
    assert not harness.revise_goal(rid, before['goal'], expected_revision=before['work']['revision'])['revision_applied']
    for goal in ('', '  ', None, 12):
        with pytest.raises(ValueError, match='goal'):
            harness.revise_goal(rid, goal, expected_revision=before['work']['revision'])
    assert harness._load(rid) == before
    assert runtime.db.get_approval(before['pending']['approval'])['status'] == 'PENDING'


def test_workbench_goal_revision_owned_and_idempotent(tmp_path, monkeypatch):
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    background = Background()
    wb = ResearchWorkbench(runtime, background)
    payload = {'request_id': 'goal-revision', 'action': 'revise_prototype_goal',
               'conversation_id': rid, 'parent_run_id': rid,
               'expected_revision': harness._load(rid)['work']['revision'], 'question': 'Add spatial gate'}
    with pytest.raises(ValueError, match='another conversation'):
        wb.submit(payload | {'request_id': 'foreign-goal', 'conversation_id': 'foreign'})
    assert wb.submit(payload)['start']
    assert wb.submit(payload)['reused'] and background.submitted == [rid]
    assert len(harness._load(rid)['goal_revisions']) == 1


@pytest.mark.parametrize('condition', ['unknown', 'unapplied', 'commit_failure', 'spent'])
def test_goal_revision_failure_and_budget_boundaries(tmp_path, monkeypatch, condition):
    from paperpilot_agentic.llm_attempts import AttemptLedger
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before = harness._load(rid)
    if condition == 'unknown':
        AttemptLedger(runtime.db).begin(rid, stage='goal-unknown', model='fixture', messages=[], output_tokens=1,
            required_tokens=2, policy_sha256=None, max_calls=16, max_tokens=100000)
    with runtime.db.transaction() as conn:
        if condition == 'unapplied':
            conn.execute('INSERT INTO research_episode_turns VALUES(?,?,?,?,?,?,?,?)',
                         (rid, 6, 'lead', '{}', 'RETURNED', 0, '{"text":"saved"}', None))
        if condition == 'commit_failure':
            conn.execute("CREATE TRIGGER fail_goal BEFORE UPDATE ON research_episodes BEGIN SELECT RAISE(ABORT,'fixture failure'); END")
    if condition == 'spent':
        before['turn'] = before['limits']['max_turns']
        harness._save(rid, before)
        harness.revise_goal(rid, 'New goal', expected_revision=before['work']['revision'])
        after = harness._load(rid)
        assert after['goal'] == 'New goal' and after['phase'] == 'PARTIAL'
        assert after['turn'] == before['turn'] and 'budget exhausted' in after['stop_reason']
    else:
        with pytest.raises(Exception, match={'unknown': 'unknown', 'unapplied': 'saved model turn',
                                             'commit_failure': 'fixture failure'}[condition]):
            harness.revise_goal(rid, 'New goal', expected_revision=before['work']['revision'])
        assert harness._load(rid) == before
        assert runtime.db.get_approval(before['pending']['approval'])['status'] == 'PENDING'


def test_goal_revision_export_cannot_claim_historical_validation_current(tmp_path, monkeypatch):
    from io import BytesIO
    import json
    from zipfile import ZipFile
    from paperpilot_agentic.prototype_delivery import build_prototype_bundle
    runtime, harness, rid = completed(tmp_path, monkeypatch)
    before = harness._load(rid)
    harness.revise_goal(rid, 'Different structure', expected_revision=before['work']['revision'])
    with ZipFile(BytesIO(build_prototype_bundle(harness._load(rid), runtime.db.get_run(rid)))) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['engineering_validation'] == 'INCOMPLETE_OR_FAILED'
        assert manifest['goal'] == 'Different structure'
        assert json.loads(archive.read('goal_revisions.json'))[0]['previous_goal'] == 'build'


def test_goal_revision_does_not_inject_previous_goal_compilation_feedback(tmp_path, monkeypatch):
    import json
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    feedback = {'task_id': 'code', 'attempt': 1, 'error': 'old goal syntax error',
                'rejected_candidate': {'source': '# rejected old candidate'}}
    state['implementation_feedback'] = feedback
    harness._save(rid, state)
    harness.revise_goal(rid, 'Implement different structure', expected_revision=state['work']['revision'])
    resumed = PrototypeHarness(runtime)
    current = resumed._load(rid)
    _, messages = resumed._context(rid, current)
    payload = json.loads(messages[-1]['content'])
    assert payload['goal'] == 'Implement different structure'
    assert payload['implementation_feedback'] is None
    assert current['goal_revisions'][-1]['previous_implementation_feedback'] == feedback
