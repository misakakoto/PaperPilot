"""Deterministic projection coverage, not model answer-quality measurements."""
from copy import deepcopy

from paperpilot_agentic.request_state import request_work_view
from paperpilot_agentic.request_worksets import byte_size, plan_worksets
from test_request_activity_scope import scoped
from test_request_state import apply


def large_view(count=100):
    return {
        'active_task_id': 'research', 'label': 'Change detection',
        'conditions': {
            'no_train': {'text': 'No training authorization', 'applies_to': None},
            **{f'c{i}': {'text': f'configuration {i} ' + 'x' * 80, 'applies_to': f'e{i}'}
               for i in range(count)}},
        'pending_actions': {f'e{i}': {'text': f'Experiment {i}', 'status': 'pending'}
                            for i in range(count)},
        'paused_actions': {}, 'paused_conditions': {},
        'assurance': 'Derived requirements, not authorization',
    }


def test_hundred_activities_partition_without_lost_or_duplicate_records():
    view = large_view()
    before = deepcopy(view)
    result = plan_worksets(view, 'revision1', 1800)
    assert result['status'] == 'NEEDS_WORK_BATCHES'
    seen = []
    for page in result['pages']:
        assert byte_size(page) <= 1800
        assert page['conditions']['no_train'] == view['conditions']['no_train']
        assert page['workset']['revision'] == 'revision1'
        assert page['workset']['scope'] == 'partial_pending_activities'
        for key in page['pending_actions']:
            seen.append(key)
            assert page['conditions']['c' + key[1:]] == view['conditions']['c' + key[1:]]
        assert set(page['conditions']) == {'no_train'} | {'c' + key[1:] for key in page['pending_actions']}
    assert seen == list(view['pending_actions'])
    assert view == before
    assert result == plan_worksets(view, 'revision1', 1800)


def test_global_constraints_never_dropped_to_make_pages_fit():
    view = large_view()
    view['conditions']['no_train']['text'] = 'x' * 3000
    result = plan_worksets(view, 'r1', 1800)
    assert result['status'] == 'BLOCKED'
    assert result['reason'] == 'GLOBAL_REQUIREMENTS_TOO_LARGE'
    assert result['pages'] == []


def test_single_oversized_activity_does_not_return_misleading_partial_plan():
    view = large_view(5)
    view['conditions']['c4']['text'] = 'x' * 3000
    result = plan_worksets(view, 'r1', 1800)
    assert result['status'] == 'BLOCKED'
    assert result['activity_key'] == 'e4'
    assert result['pages'] == []


def test_paused_details_offloaded_only_under_pressure_and_restored_on_resume():
    state = scoped()
    text = 'x' * 4000
    state = apply(state, 'r3', text, [{'kind': 'set_condition', 'task_id': 'A',
        'key': 'batch', 'text': text, 'quote': text,
        'replaces': state['tasks']['A']['conditions']['batch']['id']}])
    state = apply(state, 'r4', 'Pause', [{'kind': 'pause_todo', 'task_id': 'A',
        'key': 'latency', 'quote': 'Pause', 'replaces': state['tasks']['A']['todos']['latency']['id']}])
    before = deepcopy(state)
    assert 'batch' in request_work_view(state)['view']['paused_conditions']
    result = request_work_view(state, max_bytes=1800)
    assert result['status'] == 'READY'
    assert result['view']['workset']['deferred_paused_conditions'] == 1
    assert result['view']['conditions']['training']['text'] == '不授权训练'
    assert state == before
    state = apply(state, 'r5', 'Resume', [{'kind': 'resume_todo', 'task_id': 'A',
        'key': 'latency', 'quote': 'Resume', 'replaces': state['tasks']['A']['todos']['latency']['id']}])
    assert request_work_view(state, max_bytes=1800)['status'] == 'OVERFLOW'
    assert request_work_view(state)['view']['conditions']['batch']['text'] == text


def test_short_workset_unchanged_and_detached():
    view = large_view(1)
    result = plan_worksets(view, 'r1', 12000)
    assert result['view'] == view
    result['view']['conditions'].clear()
    assert view['conditions']


def test_compiler_audits_full_state_not_cold_projection():
    from test_request_compiler_view import Model
    from paperpilot_agentic.request_state_compiler import compile_request_batch
    state = scoped()
    text = 'x' * 4000
    state = apply(state, 'r3', text, [{'kind': 'set_condition', 'task_id': 'A',
        'key': 'batch', 'text': text, 'quote': text,
        'replaces': state['tasks']['A']['conditions']['batch']['id']}])
    state = apply(state, 'r4', 'Pause', [{'kind': 'pause_todo', 'task_id': 'A',
        'key': 'latency', 'quote': 'Pause', 'replaces': state['tasks']['A']['todos']['latency']['id']}])
    model = Model([
        {'base_revision': state['revision'], 'frames': [{'source_id': 'r5', 'operations': []}]},
        {'verdict': 'ACCEPT', 'issues': [], 'missing_requirements': []},
    ])
    report = compile_request_batch(state, [{'id': 'r5', 'text': 'Continue'}], model,
                                   max_work_bytes=1800)
    assert report['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
    assert report['work_view']['view']['paused_conditions'] == {}
    assert report['candidate_state']['tasks']['A']['conditions']['batch']['text'] == text
    assert text in model.messages[1][-1]['content']
