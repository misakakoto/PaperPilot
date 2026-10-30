import json

import pytest

from paperpilot_agentic.request_history_search import history_source_ids
from test_request_history_reload import setup, page, compile_with, NEED, ACCEPT, proposal


def test_ranked_originals_do_not_imply_chronology_or_complete_history(tmp_path):
    _, store, ids, state, current = setup(tmp_path)
    found = page(store, ids, state, current, query='3 MB', max_items=2)
    assert found['sources'][0] == {'id': ids[2], 'text': state['sources'][ids[2]]}
    assert found['order'] == 'lexical_relevance'
    assert found['coverage'] == 'task_attributed_query_matches'
    assert not found['starts_at_ledger_root'] and not found['complete']
    assert found['next_offset'] == 2
    empty = page(store, ids, state, current, query='absent-special-marker')
    assert empty['sources'] == [] and empty['complete']
    assert empty['coverage'] != 'task_attributed_committed_events'


def test_chinese_bigrams_and_identifier_tokens():
    state = {'sources': {'a': '海岸数据集：限制八百兆', 'b': '城市数据集：限制六百兆',
                         'c': 'review-phase-73: memory 777 MB'},
             'events': [{'source_id': sid, 'operations': [{'task_id': 'A'}]} for sid in 'abc']}
    assert history_source_ids(state, ['A'], '恢复海岸数据集的限制')[0] == 'a'
    assert history_source_ids(state, ['A'], 'review-phase-73') == ['c']
    assert history_source_ids(state, ['B'], 'review-phase-73') == []


@pytest.mark.parametrize('query', ['', ' ', 4, 'x' * 501])
def test_invalid_query_is_rejected(tmp_path, query):
    _, store, ids, state, current = setup(tmp_path)
    with pytest.raises(ValueError):
        page(store, ids, state, current, query=query)


def test_query_candidate_uses_same_audit_and_atomic_submission(tmp_path):
    db, store, ids, state, current = setup(tmp_path)
    request = {**NEED, 'query': '1 MB'}
    result, provider, _ = compile_with(db, store, ids, state, current,
        [request, proposal(state, current), ACCEPT], loader=lambda request:
        page(store, ids, state, current, query=request['query'], max_items=1))
    assert result['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
    assert len(provider.calls) == 3
    assert json.loads(provider.calls[-1][-1]['content'])['history']['query'] == '1 MB'
    assert store.load(ids[0], requester_id=current) == state


def test_empty_query_results_can_fall_back_to_owned_chronological_history(tmp_path):
    db, store, ids, state, current = setup(tmp_path)
    queries = []
    def load(request):
        queries.append(request.get('query'))
        return page(store, ids, state, current, query=request.get('query'), max_items=2)
    result, provider, _ = compile_with(db, store, ids, state, current,
        [{**NEED, 'query': 'missing-marker'}, NEED, proposal(state, current), ACCEPT], loader=load)
    assert result['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
    assert queries == ['missing-marker', None]
    assert len(provider.calls) == db.get_run(current)['used_llm_calls'] == 4


@pytest.mark.parametrize('corruption', ['query', 'order', 'root', 'rank'])
def test_loader_cannot_mislabel_or_reorder_query_matches(tmp_path, corruption):
    db, store, ids, state, current = setup(tmp_path)
    def load(request):
        value = page(store, ids, state, current, query=request['query'], max_items=2)
        if corruption == 'query':
            value['query'] = 'other'
        elif corruption == 'order':
            value['order'] = 'oldest_first'
        elif corruption == 'root':
            value['starts_at_ledger_root'] = True
        else:
            value['sources'].reverse()
        return value
    result, provider, _ = compile_with(db, store, ids, state, current,
        [{**NEED, 'query': '3 MB'}], loader=load)
    assert result['status'] == 'FAILED' and len(provider.calls) == 1
