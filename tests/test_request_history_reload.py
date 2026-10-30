"""Owned history pagination and budgeted continuation, with scripted model output."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget
from paperpilot_agentic.request_state_store import RequestStateStore
from paperpilot_agentic.request_state_compiler import compile_request_batch
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.providers import LLMResponse


def setup(tmp_path, count=4):
    db = RuntimeDatabase(tmp_path/'history.sqlite3')
    store = RequestStateStore(db)
    ids, state = [], None
    for i in range(count):
        text = f'Memory limit {i + 1} MB. Do not train.'
        rid = db.create_run(text, Budget(), metadata={'conversation_id': 'C', 'parent_run_id': ids[-1] if ids else None})
        ids.append(rid)
        ops = []
        if state is None:
            ops += [{'kind': 'create_task', 'task_id': 'A', 'label': 'Study', 'quote': 'Memory limit'},
                    {'kind': 'activate_task', 'task_id': 'A', 'quote': 'Memory limit'},
                    {'kind': 'set_condition', 'task_id': 'A', 'key': 'no_train', 'text': 'Do not train',
                     'expect': None, 'quote': 'Do not train'}]
        ops += [{'kind': 'set_condition', 'task_id': 'A', 'key': 'memory', 'text': f'{i + 1} MB',
                 'quote': f'Memory limit {i + 1} MB',
                 'replaces': state['tasks']['A']['conditions']['memory']['id'] if state else None}]
        state = store.commit(ids[0], source_run_id=rid,
            delta={'expected_revision': state['revision'] if state else 'initial', 'operations': ops})
    current = db.create_run('Restore the very first memory limit.', Budget(),
                           metadata={'conversation_id': 'C', 'parent_run_id': ids[-1]})
    return db, store, ids, state, current


def page(store, ids, state, requester, **kwargs):
    return store.history_page(ids[0], requester_id=requester, task_ids=['A'],
                              expected_revision=state['revision'], **kwargs)


def test_pages_are_complete_originals_with_stable_order_and_revision(tmp_path):
    db, store, ids, state, current = setup(tmp_path)
    first = page(store, ids, state, current, max_items=2)
    assert [s['id'] for s in first['sources']] == ids[:2]
    assert first['complete'] is False and first['next_offset'] == 2
    assert first['starts_at_ledger_root'] is True
    second = page(store, ids, state, current, offset=first['next_offset'], max_items=2)
    assert [s['id'] for s in second['sources']] == ids[2:]
    assert second['complete'] is True and second['next_offset'] is None
    assert second['starts_at_ledger_root'] is False
    assert first['sources'][0]['text'] == db.get_run(ids[0])['question']
    assert store.load(ids[0], requester_id=current) == state


def test_byte_limit_does_not_slice_an_original_or_skip_blocked_source(tmp_path):
    _, store, ids, state, current = setup(tmp_path)
    first = page(store, ids, state, current, max_bytes=1)
    assert first['status'] == 'SOURCE_TOO_LARGE' and first['sources'] == []
    assert first['next_offset'] == 0 and first['blocked_source']['source_id'] == ids[0]
    fitting = page(store, ids, state, current, max_bytes=first['blocked_source']['required_bytes'])
    assert len(fitting['sources']) == 1 and fitting['next_offset'] == 1


@pytest.mark.parametrize('kind', ['foreign_conversation', 'sibling', 'stale_revision', 'changed_original', 'unknown_task'])
@pytest.mark.parametrize('query', [None, '1 MB'])
def test_unowned_or_changed_history_is_rejected(tmp_path, kind, query):
    db, store, ids, state, current = setup(tmp_path)
    task_ids, revision = ['A'], state['revision']
    if kind in {'foreign_conversation', 'sibling'}:
        current = db.create_run('Other', Budget(), metadata={
            'conversation_id': 'X' if kind == 'foreign_conversation' else 'C', 'parent_run_id': ids[0]})
    elif kind == 'stale_revision':
        revision = 'stale'
    elif kind == 'changed_original':
        with db.transaction() as conn:
            conn.execute('UPDATE runs SET question=? WHERE run_id=?', ('Changed private original', ids[0]))
    else:
        task_ids = ['missing']
    with pytest.raises(ValueError):
        store.history_page(ids[0], requester_id=current, task_ids=task_ids, expected_revision=revision, query=query)


NEED = {'status': 'NEEDS_HISTORY', 'task_ids': ['A'], 'reason': 'Need first memory limit'}
ACCEPT = {'verdict': 'ACCEPT', 'issues': [], 'missing_requirements': []}


class Provider:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls.append(deepcopy(messages))
        return LLMResponse(text=json.dumps(next(self.outputs)), prompt_tokens=100,
                           completion_tokens=50, finish_reason='stop')


def proposal(state, current):
    return {'base_revision': state['revision'], 'frames': [{'source_id': current, 'operations': [
        {'kind': 'set_condition', 'task_id': 'A', 'key': 'memory', 'text': '1 MB', 'applies_to': None,
         'expect': {'text': '4 MB', 'status': 'active', 'applies_to': None},
         'quote': 'Restore the very first memory limit.'}]}]}


def compile_with(db, store, ids, state, current, outputs, *, loader=None):
    provider = Provider(outputs)
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), current, stage='request_memory')
    reloads = []
    def load(request):
        reloads.append(request)
        return page(store, ids, state, current, max_items=2) if loader is None else loader(request)
    result = compile_request_batch(state, [{'id': current, 'text': db.get_run(current)['question']}],
                                   model, history_loader=load)
    return result, provider, reloads


def test_reload_continues_to_audited_proposal_and_commits_only_current_frame(tmp_path):
    db, store, ids, state, current = setup(tmp_path)
    result, provider, reloads = compile_with(db, store, ids, state, current, [NEED, proposal(state, current), ACCEPT])
    assert result['status'] == 'MODEL_ACCEPTED_UNVERIFIED' and len(reloads) == 1
    assert len(provider.calls) == db.get_run(current)['used_llm_calls'] == 3
    assert db.get_run(current)['used_tokens'] == 450
    # No write during compile; the caller explicitly accepts/commits the proposal.
    assert store.load(ids[0], requester_id=current) == state
    committed = store.commit_batch(ids[0], source_run_ids=[current], proposal=result['proposal'])
    assert committed['tasks']['A']['conditions']['memory']['text'] == '1 MB'
    assert committed['tasks']['A']['conditions']['no_train'] == state['tasks']['A']['conditions']['no_train']
    assert len(committed['events']) == len(state['events']) + 1
    for messages in provider.calls[1:]:
        payload = json.loads(messages[-1]['content'])
        assert payload['history']['complete'] is False
        assert [s['id'] for s in payload['sources']] == [current]


def test_second_history_request_stops_without_unbounded_reload_loop(tmp_path):
    db, store, ids, state, current = setup(tmp_path)
    result, provider, reloads = compile_with(db, store, ids, state, current, [NEED, NEED])
    assert result['status'] == 'NEEDS_HISTORY' and 'candidate_state' not in result
    assert len(provider.calls) == 2 and len(reloads) == 1


@pytest.mark.parametrize('corruption', ['revision', 'text', 'task_ids'])
def test_injected_history_mismatch_stops_before_second_model_call(tmp_path, corruption):
    db, store, ids, state, current = setup(tmp_path)
    def loader(request):
        value = page(store, ids, state, current)
        if corruption == 'text':
            value['sources'][0]['text'] = 'invented old value'
        elif corruption == 'revision':
            value['revision'] = 'stale'
        else:
            value['task_ids'] = ['other']
        return value
    result, provider, _ = compile_with(db, store, ids, state, current, [NEED], loader=loader)
    assert result['status'] == 'FAILED' and len(provider.calls) == 1
    assert store.load(ids[0], requester_id=current) == state


def test_too_large_original_does_not_trigger_paid_recompile(tmp_path):
    db, store, ids, state, current = setup(tmp_path)
    result, provider, _ = compile_with(db, store, ids, state, current, [NEED],
        loader=lambda request: page(store, ids, state, current, max_bytes=1))
    assert result['status'] == 'NEEDS_HISTORY' and len(provider.calls) == 1
    assert result['history']['status'] == 'SOURCE_TOO_LARGE'


@pytest.mark.parametrize('field', ['complete', 'starts_at_ledger_root', 'source_order'])
def test_history_origin_and_coverage_are_checked_against_ledger(tmp_path, field):
    db, store, ids, state, current = setup(tmp_path)
    def loader(request):
        value = page(store, ids, state, current, max_items=2)
        if field == 'source_order':
            value['sources'].reverse()
        else:
            value[field] = not value[field]
        return value
    result, provider, _ = compile_with(db, store, ids, state, current, [NEED], loader=loader)
    assert result['status'] == 'FAILED' and len(provider.calls) == 1
