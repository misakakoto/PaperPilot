import copy
import json

from paperpilot_agentic.prototype_workspace import complete, plan, task_rows
from test_prototype_analysis import prepared


def test_multiple_papers_share_a_byte_budget_without_changing_originals(tmp_path, monkeypatch):
    _, h, rid, state, _ = prepared(tmp_path, monkeypatch, text='A bounded methodological statement. ' * 1000)
    state.update(analysis_read_policy='selected-span-sequence-v1', quote_handle_policy='task-local-quote-handles-v1',
                 analysis_workset_policy='shared-byte-workset-v1')
    rows = task_rows(state['work'])
    analysis = next(t for t in rows if t['id'] == 'analysis')
    for key in ('paper_b', 'paper_c'):
        rows.append({'id': key, 'goal': 'Read another method', 'capability': 'read_literature',
                     'depends_on': ['search'], 'acceptance': 'Original pages'})
        analysis['depends_on'].append(key)
    plan(state['work'], rows)
    for key in ('paper_b', 'paper_c'):
        complete(state['work'], key, 'read_literature', {'ok': True, 'evidence': [{
            'evidence_type': 'TEXT', 'content': 'Another bounded methodological statement. ' * 1000,
            'source_uri': 'paper://' + key, 'page': 4}]})
    before = copy.deepcopy(state)
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    views = [v for v in payload['context_artifacts'].values() if v['kind'] == 'read_literature']
    assert sum(len(json.dumps(v, ensure_ascii=False).encode()) for v in views) <= 20000
    assert all(v['content']['context_view']['omitted_spans'] > 0 for v in views)
    assert len(messages[-1]['content'].encode()) < 64000
    assert state == before
