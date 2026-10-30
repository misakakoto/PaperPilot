import copy

import pytest

from paperpilot_agentic.builtin_tools import _search_handler
from paperpilot_agentic.prototype_analysis import spans
from paperpilot_agentic.prototype_literature import LOCAL_READ_POLICY
from paperpilot_agentic.prototype_workspace import complete, plan, put
from test_prototype_discovery_dates import prepared

SHA = 'a'*64


def setup(tmp_path, monkeypatch, *, done=True, tool='paper.search'):
    rt, h, rid, state, _ = prepared(tmp_path, monkeypatch)
    state['literature_local_read_policy'] = LOCAL_READ_POLICY
    plan(state['work'], [
        {'id': 'local', 'goal': 'discover', 'capability': 'search_literature', 'depends_on': [], 'acceptance': 'receipt'},
        {'id': 'details', 'goal': 'method', 'capability': 'read_literature', 'depends_on': ['local'], 'acceptance': 'TEXT'}],
        enforce_capabilities=True)
    if done:
        complete(state['work'], 'local', 'search_literature', {'ok': True, 'tool': tool, 'evidence': [
            {'provenance': {'paper_sha256': SHA}}]})
    calls = []

    class Local:
        def search_many(self, queries, *, top_k, paper_sha256s=None):
            calls.append((queries, top_k, paper_sha256s))
            return [{'content': 'Local temporal fusion preserves boundaries.', 'title': 'Paper',
                     'page': 3, 'paper_sha256': SHA}]

    monkeypatch.setattr(rt.registry.get('paper.search'), 'handler', _search_handler(Local()))
    return rt, h, rid, state, calls


def read(h, rid, state, **changes):
    return h._apply(rid, state, {'action': 'work', 'reason': 'targeted paper reading', 'arguments': {
        'task_id': 'details', 'content': {'paper_sha256': SHA, 'query': 'temporal fusion method',
                                      'top_k': 3, **changes}}}, 2)


def test_local_read_uses_scoped_registered_tool_and_literal_text(tmp_path, monkeypatch):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    read(h, rid, state)
    assert calls == [(['temporal fusion method'], 3, {SHA})]
    identity = state['work']['tasks']['details']['output']
    artifact = state['work']['artifacts'][identity]
    assert artifact['inputs'] == [state['work']['tasks']['local']['output']]
    assert artifact['content']['coverage'] == 'TARGETED_LOCAL_TEXT_RETRIEVAL_NOT_FULL_PAPER_READ'
    assert spans(identity, artifact)[0]['text'] == 'Local temporal fusion preserves boundaries.'
    assert rt.db.list_tool_calls(rid)[0]['tool_name'] == 'paper.search'


@pytest.mark.parametrize('change', [{'paper_sha256': 'b'*64}, {'paper_sha256': '../private'},
    {'query': ''}, {'top_k': True}, {'top_k': 7}, {'url': 'file:///private.pdf'}])
def test_local_read_rejects_unselected_papers_and_invalid_arguments(tmp_path, monkeypatch, change):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        read(h, rid, state, **change)
    assert not calls and not rt.db.list_tool_calls(rid) and state == before


@pytest.mark.parametrize('done,tool', [(False, 'paper.search'), (True, 'scholar.search')])
def test_local_read_requires_completed_local_discovery(tmp_path, monkeypatch, done, tool):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch, done=done, tool=tool)
    with pytest.raises(ValueError):
        read(h, rid, state)
    assert not calls and not rt.db.list_tool_calls(rid)


def test_saved_tool_receipt_recovers_without_requery(tmp_path, monkeypatch):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    read(h, rid, state)
    count = rt.db.get_run(rid)['used_tool_calls']
    read(h, rid, before)
    assert len(calls) == 1 and rt.db.get_run(rid)['used_tool_calls'] == count
    assert before['work']['tasks']['details']['output'] == state['work']['tasks']['details']['output']


def test_dependency_change_uses_new_receipt_even_without_plan_revision(tmp_path, monkeypatch):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    read(h, rid, state)
    work = state['work']
    old_id = work['tasks']['details']['output']
    local_id = work['tasks']['local']['output']
    updated = copy.deepcopy(work['artifacts'][local_id]['content'])
    updated['retrieval_note'] = 'a distinct discovery receipt version'
    new_id = put(work, 'local', 'search_literature', updated)
    work['tasks']['local'].update(status='DONE', output=new_id)
    assert work['tasks']['details']['status'] == 'PENDING'
    read(h, rid, state)
    assert len(calls) == 2 and len(rt.db.list_tool_calls(rid)) == 2
    assert work['tasks']['details']['output'] != old_id
    assert old_id in work['artifacts']


def test_historical_protocol_does_not_gain_local_read_form(tmp_path, monkeypatch):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    state.pop('literature_local_read_policy')
    with pytest.raises(ValueError, match='Read content'):
        read(h, rid, state)
    assert not calls and not rt.db.list_tool_calls(rid)


def test_scope_violation_is_not_accepted_as_selected_paper_evidence(tmp_path, monkeypatch):
    from paperpilot_agentic.models import ToolResult
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(rt.registry.get('paper.search'), 'handler', lambda context, args: ToolResult(
        ok=True, output={}, evidence=[{'evidence_type':'TEXT', 'content':'Wrong paper text.',
                                      'provenance':{'paper_sha256':'b'*64}}]))
    read(h, rid, state)
    identity = state['work']['tasks']['details']['output']
    artifact = state['work']['artifacts'][identity]
    assert artifact['content']['ok'] is False and not artifact['content']['evidence']
    assert artifact['content']['coverage'] == 'SCOPE_MISMATCH_NOT_EVIDENCE'
    raw = rt.db.list_tool_calls(rid)[0]['result']
    assert raw['evidence'][0]['content'] == 'Wrong paper text.'


def test_tampered_discovery_does_not_dispatch(tmp_path, monkeypatch):
    rt, h, rid, state, calls = setup(tmp_path, monkeypatch)
    identity = state['work']['tasks']['local']['output']
    state['work']['artifacts'][identity]['content']['evidence'][0]['provenance']['paper_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='identity changed'):
        read(h, rid, state)
    assert not calls and not rt.db.list_tool_calls(rid)
