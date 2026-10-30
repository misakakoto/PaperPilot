import copy
import json

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import complete, new_workspace, plan


def shared_context_state(mode):
    work = new_workspace('Shared dependencies must appear once without losing versions')
    rows = [{'id': k, 'goal': k, 'capability': kind, 'depends_on': deps, 'acceptance': 'exact'}
            for k, kind, deps in [('read', 'read', []), ('design', 'design', ['read']),
                ('code', 'implement', ['design']), ('check', 'validate', ['code']),
                ('review1', 'review', ['read', 'design', 'code', 'check']),
                ('review2', 'review', ['read', 'design', 'code', 'check'])]]
    plan(work, rows)
    complete(work, 'read', 'read', {'content': 'SOURCE_UNIQUE_CONTENT'*400})
    complete(work, 'design', 'design', {'hypothesis': 'Unmeasured'})
    code = complete(work, 'code', 'implement', {'source': 'MODEL_UNIQUE_CONTENT'*300, 'config': {}})
    raw = {'passed': False, 'validated_code_id': code, 'failure_kind': 'IMPLEMENTATION',
           'execution': {'metrics': {'error': 'shape mismatch'}, 'stderr_path': 'RAW_PRIVATE_PATH'}}
    complete(work, 'check', 'validate', raw)
    return {'work': work, 'sources': {}, 'mode': mode, 'phase': 'READY',
            'validation_feedback': {'code_id': code, 'result': raw},
            'context_policy': 'artifact-table-v1'}


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_shared_dependencies_are_serialized_once_with_exact_task_references(mode):
    state = shared_context_state(mode)
    before = copy.deepcopy(state)
    harness = PrototypeHarness.__new__(PrototypeHarness)
    _, messages = harness._context('fixture', state)
    wire = messages[-1]['content']
    assert wire.count('SOURCE_UNIQUE_CONTENT') == 400
    assert wire.count('MODEL_UNIQUE_CONTENT') == 300
    assert wire.count(state['work']['goal']) == 1
    payload = json.loads(wire)
    assert payload['context_policy'] == 'artifact-table-v1'
    assert set(payload['ready_contexts']) == {'review1', 'review2'}
    for task in payload['ready_contexts'].values():
        assert payload['tasks'][task['task']['task_ref']]['capability'] == 'review'
        for identity, reference in task['dependencies'].items():
            assert reference == {'artifact_ref': identity}
            view = payload['context_artifacts'][identity]
            original = state['work']['artifacts'][identity]
            if original['kind'] == 'validate':
                assert view['raw_receipt_artifact_id'] == identity
                assert view['content']['execution']['metrics']['error'] == 'shape mismatch'
            else:
                assert view == original
    feedback_ref = payload['validation_feedback']['result']
    assert feedback_ref['field'] == 'content'
    assert payload['context_artifacts'][feedback_ref['artifact_ref']]['content']['failure_kind'] == 'IMPLEMENTATION'
    assert 'RAW_PRIVATE_PATH' not in wire
    assert state == before


def test_legacy_state_keeps_its_previous_context_protocol():
    state = shared_context_state('adaptive')
    state.pop('context_policy')
    harness = PrototypeHarness.__new__(PrototypeHarness)
    _, messages = harness._context('fixture', state)
    payload = json.loads(messages[-1]['content'])
    assert 'context_artifacts' not in payload
    assert messages[-1]['content'].count('SOURCE_UNIQUE_CONTENT') == 800


def test_new_task_records_policy_but_resubmission_does_not_migrate_legacy_task(tmp_path, monkeypatch):
    from test_prototype_crash_recovery import setup
    runtime, harness, rid = setup(tmp_path, monkeypatch)
    state = harness._load(rid)
    assert state['context_policy'] == 'artifact-table-v1'
    assert runtime.db.get_run(rid)['metadata']['prototype_context_policy'] == 'artifact-table-v1'
    state.pop('context_policy')
    harness._save(rid, state)
    before = runtime.db.get_run(rid)
    result = harness.start('build', 'crash-fixture', 'parent', ['source'])
    assert result['run']['run_id'] == rid
    assert 'context_policy' not in harness._load(rid)
    assert runtime.db.get_run(rid)['budget'] == before['budget']
    assert runtime.db.get_run(rid)['created_at'] == before['created_at']


def test_unknown_context_policy_is_rejected_without_mutating_state():
    state = shared_context_state('adaptive')
    state['context_policy'] = 'unknown'
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='context policy'):
        PrototypeHarness.__new__(PrototypeHarness)._context('fixture', state)
    assert state == before
