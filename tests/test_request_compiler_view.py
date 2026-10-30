"""Projection invariants, not model interpretation accuracy."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.request_state import (
    apply_request_delta, empty_request_state, request_compiler_view)
from paperpilot_agentic.request_state_compiler import compile_request_batch
from paperpilot_agentic.providers import LLMResponse


def edit(state, sid, text, operations):
    return apply_request_delta(state, {'id': sid, 'text': text},
                               {'expected_revision': state['revision'], 'operations': operations})


def seed():
    text = 'SAR change detection. Do not train. Plan latency measurement, repeat 30 times.'
    return edit(empty_request_state(), 'r0', text, [
        {'kind': 'create_task', 'task_id': 'A', 'label': 'SAR change', 'quote': 'SAR change detection'},
        {'kind': 'activate_task', 'task_id': 'A', 'quote': 'SAR change detection'},
        {'kind': 'set_condition', 'task_id': 'A', 'key': 'no_train', 'text': 'Do not train',
         'quote': 'Do not train', 'expect': None, 'applies_to': None},
        {'kind': 'add_todo', 'task_id': 'A', 'key': 'latency', 'text': 'Plan latency measurement',
         'quote': 'Plan latency measurement', 'expect': None},
        {'kind': 'set_condition', 'task_id': 'A', 'key': 'repetitions', 'text': '30',
         'quote': 'repeat 30 times', 'expect': None, 'applies_to': 'latency'},
    ])


def many_updates(count=200):
    state = seed()
    for i in range(count):
        text = f'Memory limit now {i + 1} MB.'
        old = state['tasks']['A']['conditions'].get('memory')
        state = edit(state, f'u{i}', text, [
            {'kind': 'set_condition', 'task_id': 'A', 'key': 'memory', 'text': f'{i + 1} MB',
             'quote': text, 'replaces': old['id'] if old else None}])
    return state


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def complete(self, messages, **kwargs):
        self.messages.append(deepcopy(messages))
        return LLMResponse(text=json.dumps(next(self.outputs)), prompt_tokens=10,
                           completion_tokens=5, finish_reason='stop')


def test_projection_preserves_every_current_record_and_scoped_paused_work():
    state = seed()
    state = edit(state, 'pause', 'Pause latency', [{'kind': 'pause_todo', 'task_id': 'A',
        'key': 'latency', 'quote': 'Pause latency', 'replaces': state['tasks']['A']['todos']['latency']['id']}])
    state = edit(state, 'switch', 'New task RGB roads', [
        {'kind': 'create_task', 'task_id': 'B', 'label': 'Roads', 'quote': 'RGB roads'},
        {'kind': 'activate_task', 'task_id': 'B', 'quote': 'New task RGB roads'}])
    before = deepcopy(state)
    view = request_compiler_view(state)
    assert view['active_task_id'] == 'B' and set(view['tasks']) == {'A', 'B'}
    for tid in state['tasks']:
        for field in ('conditions', 'todos'):
            assert view['tasks'][tid][field] == state['tasks'][tid][field]
    assert view['tasks']['A']['todos']['latency']['status'] == 'paused'
    assert view['tasks']['A']['conditions']['repetitions']['applies_to'] == 'latency'
    view['tasks']['A']['conditions'].clear()
    assert state == before


def test_cancelled_values_not_in_current_projection_but_remain_reloadable():
    state = seed()
    state = edit(state, 'cancel', 'Cancel latency', [{'kind': 'cancel_todo', 'task_id': 'A',
        'key': 'latency', 'quote': 'Cancel latency', 'replaces': state['tasks']['A']['todos']['latency']['id']}])
    view = request_compiler_view(state)
    assert view['tasks']['A']['history_available'] is True
    assert set(view['tasks']['A']['conditions']) == {'no_train'}
    assert not view['tasks']['A']['todos']
    assert len(state['tasks']['A']['archive']) == 2 and 'repeat 30' in state['sources']['r0']
    assert 'sources' not in view and 'events' not in view and 'archive' not in view['tasks']['A']


def test_former_names_remain_identity_aliases_without_replaying_whole_rename_history():
    state = seed()
    for i, name in enumerate(['Buildings', 'SAR change', 'Buildings']):
        state = edit(state, f'name{i}', f'Rename to {name}', [{'kind': 'rename_task', 'task_id': 'A',
            'previous_label': state['tasks']['A']['label'], 'label': name, 'quote': name}])
    view = request_compiler_view(state)
    assert view['tasks']['A']['former_labels'] == [{'label': 'SAR change', 'source_id': 'name1'}]
    assert len(state['tasks']['A']['label_history']) == 3


def test_200_updates_do_not_reinject_archived_values_into_model_requests():
    state = many_updates()
    before = deepcopy(state)
    sources = [{'id': 'new', 'text': 'Continue task A.'}]
    proposal = {'base_revision': state['revision'], 'frames': [{'source_id': 'new', 'operations': []}]}
    model = Model([proposal, {'verdict': 'ACCEPT', 'issues': [], 'missing_requirements': []}])
    assert len(json.dumps(state).encode()) > 64000
    result = compile_request_batch(state, sources, model, max_input_bytes=12000)
    assert result['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
    assert all(c['input_bytes'] < 12000 for c in result['calls'])
    projected = json.loads(model.messages[0][1]['content'])['state']
    assert projected['tasks']['A']['conditions']['memory']['text'] == '200 MB'
    assert projected['tasks']['A']['conditions'] == state['tasks']['A']['conditions']
    assert len(result['candidate_state']['events']) == 202
    assert result['candidate_state']['sources']['u0'] == 'Memory limit now 1 MB.'
    assert state == before


def test_many_live_conditions_are_not_silently_dropped_to_fit():
    state = seed()
    for i in range(100):
        text = f'Keep requirement {i}: ' + 'original detail ' * 20
        state = edit(state, f'c{i}', text, [{'kind': 'set_condition', 'task_id': 'A',
            'key': f'c{i}', 'text': text, 'quote': text, 'expect': None}])
    model = Model([])
    result = compile_request_batch(state, [{'id': 'new', 'text': 'Continue'}], model)
    assert result['status'] == 'NEEDS_SMALLER_BATCH' and not model.messages
    assert len(request_compiler_view(state)['tasks']['A']['conditions']) == 102


def test_missing_historical_value_returns_reload_request_not_guessed_candidate():
    state = many_updates(3)
    before = deepcopy(state)
    model = Model([{'status': 'NEEDS_HISTORY', 'task_ids': ['A'], 'reason': 'Original memory limit'}])
    result = compile_request_batch(state, [{'id': 'restore', 'text': 'Restore the first memory limit.'}], model)
    assert result['status'] == 'NEEDS_HISTORY' and len(model.messages) == 1
    assert 'candidate_state' not in result and state == before


@pytest.mark.parametrize('history_payload', [
    {'status': 'NEEDS_HISTORY', 'task_ids': ['foreign'], 'reason': 'Old value'},
    {'status': 'NEEDS_HISTORY', 'task_ids': ['A', 'A'], 'reason': 'Old value'},
    {'status': 'NEEDS_HISTORY', 'task_ids': ['A'], 'reason': 'Old value', 'frames': []},
    {'status': 'NEEDS_HISTORY', 'task_ids': ['A'], 'reason': ''},
])
def test_invalid_history_requests_cannot_be_published(history_payload):
    model = Model([history_payload])
    result = compile_request_batch(seed(), [{'id': 'restore', 'text': 'Restore old limit.'}], model)
    assert result['status'] == 'FAILED' and 'history_request' not in result
    assert len(model.messages) == 1
