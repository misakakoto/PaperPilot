import copy
import json

import pytest

from paperpilot_agentic.prototype_workspace import complete, plan, task_rows
from test_prototype_analysis import prepared, proposal
from test_prototype_literature import work

POLICY = 'task-local-quote-handles-v1'


def rendered(tmp_path, monkeypatch):
    _, h, rid, state, identity = prepared(tmp_path, monkeypatch)
    state['quote_handle_policy'] = POLICY
    state['analysis_read_policy'] = 'selected-span-sequence-v1'
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    pair = payload['context_artifacts'][identity]['content']['pages'][0]['quotes'][0]
    return h, rid, state, identity, pair


def test_context_handles_resolve_to_exact_original_quotes_and_audit(tmp_path, monkeypatch):
    h, rid, state, identity, pair = rendered(tmp_path, monkeypatch)
    assert pair[0].startswith('h_')
    content = proposal(state, identity)
    canonical = content['findings'][0]['quote_id']
    content['findings'][0]['quote_id'] = pair[0]
    before = copy.deepcopy(state['work']['artifacts'])
    work(h, rid, state, 'analysis', content)
    result = state['work']['artifacts'][state['work']['tasks']['analysis']['output']]['content']
    assert result['findings'][0]['quote_id'] == canonical
    assert result['findings'][0]['quote'] == pair[1]
    assert all(state['work']['artifacts'][k] == v for k, v in before.items())
    assert state['analysis_quote_normalizations'][0]['resolutions'][0]['handle'] == pair[0]
    assert content['findings'][0]['quote_id'] == pair[0]


@pytest.mark.parametrize('mutation', ['goal_revision', 'source_revision', 'foreign_source', 'unknown_handle'])
def test_stale_foreign_or_unknown_handles_are_rejected_without_guessing(tmp_path, monkeypatch, mutation):
    h, rid, state, identity, pair = rendered(tmp_path, monkeypatch)
    content = proposal(state, identity)
    content['findings'][0]['quote_id'] = pair[0]
    if mutation == 'goal_revision':
        rows = task_rows(state['work'])
        next(t for t in rows if t['id'] == 'analysis')['goal'] += ' revised scope'
        plan(state['work'], rows)
    elif mutation == 'source_revision':
        rows = task_rows(state['work'])
        next(t for t in rows if t['id'] == 'paper')['goal'] += ' read updated source'
        original = copy.deepcopy(state['work']['artifacts'][identity]['content'])
        original['evidence'][0]['content'] += ' Additional method detail.'
        plan(state['work'], rows)
        complete(state['work'], 'paper', 'read_literature', original)
    elif mutation == 'foreign_source':
        content['findings'][0]['artifact_id'] = 'foreign'
    else:
        content['findings'][0]['quote_id'] = pair[0] + 'wrong'
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='handle'):
        work(h, rid, state, 'analysis', content)
    assert state == before
