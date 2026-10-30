import json

import pytest

from test_request_history_reload import setup, page, Provider, ACCEPT
from paperpilot_agentic.request_state_compiler import compile_request_batch
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime


NEED = {'status': 'NEEDS_HISTORY', 'task_ids': ['A'], 'reason': 'Need third memory value'}


def run(tmp_path, outputs, *, max_pages=4, ignore_offset=False):
    db, store, ids, state, current = setup(tmp_path)
    # Current request is not committed; set the test's actual input, not history.
    question = 'Restore the third memory limit.'
    with db.transaction() as conn:
        conn.execute('UPDATE runs SET question=? WHERE run_id=?', (question, current))
    provider = Provider(outputs(state, current, question))
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), current, stage='request_memory')
    offsets = []
    def load(request):
        offset = request.get('offset', 0)
        offsets.append(offset)
        return page(store, ids, state, current, offset=0 if ignore_offset else offset, max_items=2)
    result = compile_request_batch(state, [{'id': current, 'text': question}], model,
                                   history_loader=load, max_history_pages=max_pages)
    return result, provider, offsets, db, store, ids, state, current


def successful(state, current, question):
    return [NEED, {**NEED, 'offset': 2},
        {'base_revision': state['revision'], 'frames': [{'source_id': current, 'operations': [{
            'kind': 'set_condition', 'task_id': 'A', 'key': 'memory', 'text': '3 MB',
            'applies_to': None, 'expect': {'text': '4 MB', 'status': 'active', 'applies_to': None},
            'quote': question}]}]}, ACCEPT]


def test_second_page_restores_later_value_without_growing_model_history(tmp_path):
    result, provider, offsets, db, store, ids, state, current = run(tmp_path, successful)
    assert result['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
    assert offsets == [0, 2]
    assert len(result['history_pages']) == 2
    assert db.get_run(current)['used_llm_calls'] == 4
    assert db.get_run(current)['used_tokens'] == 600
    for call in provider.calls[2:]:
        payload = json.loads(call[-1]['content'])
        assert [s['id'] for s in payload['history']['sources']] == ids[2:]
        # Current records retain provenance IDs, but old request bodies do not
        # accumulate alongside the next page.
        assert 'Memory limit 1 MB. Do not train.' not in json.dumps(payload)
    assert store.load(ids[0], requester_id=current) == state
    committed = store.commit_batch(ids[0], source_run_ids=[current], proposal=result['proposal'])
    assert committed['tasks']['A']['conditions']['memory']['text'] == '3 MB'
    assert committed['tasks']['A']['conditions']['no_train'] == state['tasks']['A']['conditions']['no_train']
    assert len(committed['events']) == len(state['events']) + 1


def test_page_budget_stops_before_another_loader_or_paid_call(tmp_path):
    result, provider, offsets, *_ = run(tmp_path, successful, max_pages=1)
    assert result['status'] == 'NEEDS_HISTORY'
    assert result['history_stop_reason'] == 'page_budget_exhausted'
    assert result['history_request']['offset'] == 2
    assert offsets == [0] and len(provider.calls) == 2
    assert 'candidate_state' not in result


def test_loader_ignoring_cursor_is_rejected_before_another_model_call(tmp_path):
    result, provider, offsets, *_ = run(tmp_path, successful, ignore_offset=True)
    assert result['status'] == 'FAILED' and result['error_type'] == 'ValueError'
    assert offsets == [0, 2] and len(provider.calls) == 2


@pytest.mark.parametrize('offset', [-1, True, 1, 3])
def test_invalid_or_skipped_cursor_cannot_hide_history(tmp_path, offset):
    result, provider, offsets, *_ = run(tmp_path,
        lambda *args: [NEED, {**NEED, 'offset': offset}])
    assert result['status'] == 'FAILED'
    assert offsets == [0] and len(provider.calls) == 2


@pytest.mark.parametrize('query', [None, '37 MB'])
def test_default_memory_entry_forwards_cursor_and_checkpoints_result(tmp_path, query):
    from types import SimpleNamespace
    from paperpilot_agentic.reader_request_memory import prepare_reader_request_memory
    db, store, ids, state, current = setup(tmp_path, count=40)
    question = 'Restore the memory limit at update 37.'
    with db.transaction() as conn:
        conn.execute('UPDATE runs SET question=? WHERE run_id=?', (question, current))
    db.update_run_metadata(ids[-1], {'request_state_root': ids[0]})
    lookup = [NEED, {**NEED, 'offset': 32}] if query is None else [{**NEED, 'query': query}]
    outputs = lookup + [{'base_revision': state['revision'], 'frames': [{'source_id': current, 'operations': [{
            'kind': 'set_condition', 'task_id': 'A', 'key': 'memory', 'text': '37 MB',
            'applies_to': None, 'expect': {'text': '40 MB', 'status': 'active', 'applies_to': None},
            'quote': question}]}]}, ACCEPT]
    provider = Provider(outputs)
    runtime = SimpleNamespace(db=db, llm_runtime=PersistentLLMRuntime(db, provider))
    result = prepare_reader_request_memory(runtime, db.get_run(current), db.get_run(ids[-1]))
    assert result['status'] == 'READY'
    assert store.load(ids[0], requester_id=current)['tasks']['A']['conditions']['memory']['text'] == '37 MB'
    second = json.loads(provider.calls[len(lookup)][-1]['content'])['history']
    if query is None:
        assert second['offset'] == 32 and [s['id'] for s in second['sources']] == ids[32:]
    else:
        assert second['query'] == query and second['order'] == 'lexical_relevance'
        assert second['sources'][0]['id'] == ids[36] and len(second['sources']) == 4
    assert prepare_reader_request_memory(runtime, db.get_run(current), db.get_run(ids[-1])) == result
    assert len(provider.calls) == db.get_run(current)['used_llm_calls'] == len(lookup) + 2
