import copy
import json

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import complete, new_workspace, plan


def two_branches(second_size=20000):
    work = new_workspace('Two independent implementation tasks, serial execution')
    rows = [{'id': key, 'goal': key, 'capability': cap, 'depends_on': deps, 'acceptance': 'exact'}
            for key, cap, deps in [('read_a', 'read', []), ('read_b', 'read', []),
                                   ('design_a', 'design', ['read_a']), ('design_b', 'design', ['read_b']),
                                   ('code_a', 'implement', ['read_a', 'design_a']),
                                   ('code_b', 'implement', ['read_b', 'design_b'])]]
    plan(work, rows)
    for key, text in [('read_a', 'SOURCE_A'*500), ('read_b', 'SOURCE_B'*second_size)]:
        complete(work, key, 'read', {'content': text})
    for key in ('design_a', 'design_b'):
        complete(work, key, 'design', {'hypothesis': key})
    return {'work': work, 'sources': {}, 'mode': 'adaptive', 'phase': 'READY',
            'context_policy': 'artifact-table-v1', 'task_context_policy': 'serial-task-v1'}


def test_only_active_task_context_is_built_even_if_other_ready_context_is_oversized():
    state = two_branches()
    before = copy.deepcopy(state)
    _, messages = PrototypeHarness.__new__(PrototypeHarness)._context('fixture', state)
    payload = json.loads(messages[-1]['content'])
    assert payload['active_task_id'] == 'code_a'
    assert payload['ready_task_ids'] == ['code_a', 'code_b']
    assert set(payload['ready_contexts']) == {'code_a'}
    assert 'SOURCE_B' not in messages[-1]['content']
    assert 'SOURCE_A' in messages[-1]['content']
    assert set(payload['artifact_index']) == set(state['work']['artifacts'])
    assert state == before


def test_non_active_work_cannot_skip_host_schedule():
    state = two_branches()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='active task'):
        PrototypeHarness.__new__(PrototypeHarness)._apply('fixture', state,
            {'action': 'work', 'reason': 'skip order', 'arguments': {'task_id': 'code_b',
                'content': {'source': 'def build_model(config):\n return None\n', 'config': {}}}}, 1)
    assert state == before


def test_unknown_schedule_policy_stops_context_and_apply():
    state = two_branches()
    state['task_context_policy'] = 'future-unknown'
    with pytest.raises(ValueError, match='task context policy'):
        PrototypeHarness.__new__(PrototypeHarness)._context('fixture', state)
    with pytest.raises(ValueError, match='task context policy'):
        PrototypeHarness.__new__(PrototypeHarness)._apply('fixture', state,
            {'action': 'work', 'reason': 'fixture', 'arguments': {'task_id': 'code_a', 'content': {}}}, 1)


def test_legacy_multi_ready_context_semantics_unchanged():
    state = two_branches()
    state.pop('task_context_policy')
    # Legacy protocol still attempts all contexts and rejects the oversized one.
    with pytest.raises(ValueError, match='context exceeds window'):
        PrototypeHarness.__new__(PrototypeHarness)._context('fixture', state)


def test_plan_order_survives_canonical_json_and_revisions():
    from paperpilot_agentic.reference_methods import encoded
    state = two_branches(second_size=10)
    tasks = state['work']['tasks']
    order = ['code_b', 'code_a']+[k for k in tasks if not k.startswith('code_')]
    rows = [{field: tasks[k][field] for field in ('id', 'goal', 'capability', 'depends_on', 'acceptance')}
            for k in order]
    plan(state['work'], rows)
    reloaded = json.loads(encoded(state))
    _, messages = PrototypeHarness.__new__(PrototypeHarness)._context('fixture', reloaded)
    payload = json.loads(messages[-1]['content'])
    assert payload['active_task_id'] == 'code_b'
    assert payload['ready_task_ids'] == ['code_b', 'code_a']


@pytest.mark.parametrize('order', [None, ['code_a']*6, 'code_a', ['absent']])
def test_invalid_saved_schedule_is_rejected(order):
    state = two_branches(second_size=10)
    state['work']['task_order'] = order
    with pytest.raises(ValueError, match='policy order'):
        PrototypeHarness.__new__(PrototypeHarness)._context('fixture', state)


def test_new_task_policy_recorded_but_old_submission_is_not_migrated(tmp_path, monkeypatch):
    from test_prototype_crash_recovery import setup
    runtime, harness, rid = setup(tmp_path, monkeypatch)
    state = harness._load(rid)
    assert state['task_context_policy'] == 'serial-task-v1'
    assert runtime.db.get_run(rid)['metadata']['prototype_task_context_policy'] == 'serial-task-v1'
    state.pop('task_context_policy')
    harness._save(rid, state)
    before = runtime.db.get_run(rid)
    assert harness.start('build', 'crash-fixture', 'parent', ['source'])['run']['run_id'] == rid
    assert 'task_context_policy' not in harness._load(rid)
    assert runtime.db.get_run(rid)['budget'] == before['budget']
    assert runtime.db.get_run(rid)['used_llm_calls'] == before['used_llm_calls']


def test_delivery_records_schedule_policy_and_order():
    from io import BytesIO
    from zipfile import ZipFile
    from paperpilot_agentic.prototype_delivery import build_prototype_bundle
    state = two_branches(second_size=10)
    with ZipFile(BytesIO(build_prototype_bundle(state, {'run_id': 'fixture', 'used_llm_calls': 0,
                                                       'used_tokens': 0, 'budget': {}}))) as archive:
        assert json.loads(archive.read('manifest.json'))['task_context_policy'] == 'serial-task-v1'
        assert json.loads(archive.read('task_order.json')) == state['work']['task_order']
