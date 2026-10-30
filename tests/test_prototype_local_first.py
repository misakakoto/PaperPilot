import copy

import pytest

from paperpilot_agentic.prototype_literature import LOCAL_FIRST_POLICY
from paperpilot_agentic.prototype_workspace import complete, plan
from test_prototype_discovery_dates import prepared, apply


def configured(tmp_path, monkeypatch, *, receipt=None, dependency=True):
    rt, h, rid, state, academic = prepared(tmp_path, monkeypatch)
    state['literature_local_first_policy'] = LOCAL_FIRST_POLICY
    plan(state['work'], [
        {'id': 'local', 'goal': 'library evidence', 'capability': 'search_literature',
         'depends_on': [], 'acceptance': 'local receipt'},
        {'id': 'search', 'goal': 'fill gap', 'capability': 'search_literature',
         'depends_on': ['local'] if dependency else [], 'acceptance': 'external receipt'}],
         enforce_capabilities=True)
    if receipt is not None:
        complete(state['work'], 'local', 'search_literature', receipt)
    return rt, h, rid, state, academic


def test_new_tasks_default_to_local_first(tmp_path, monkeypatch):
    from test_prototype_workspace import Reader
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    rt = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'new.db', llm=Reader())
    monkeypatch.setattr(rt.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'baseline'}])
    h = PrototypeHarness(rt)
    rid = h.start('research', 'new-local', 'parent', ['source'], literature_enabled=True)['run']['run_id']
    assert h._load(rid)['literature_local_first_policy'] == LOCAL_FIRST_POLICY


@pytest.mark.parametrize('receipt,dependency,gap', [
    (None, True, 'need recent evidence'),
    ({'ok': False, 'tool': 'paper.search'}, True, 'need recent evidence'),
    ({'ok': True, 'tool': 'scholar.search'}, True, 'need recent evidence'),
    ({'ok': True, 'tool': 'paper.search'}, False, 'need recent evidence'),
    ({'ok': True, 'tool': 'paper.search'}, True, ''),
])
def test_external_cannot_bypass_current_local_evidence(tmp_path, monkeypatch, receipt, dependency, gap):
    rt, h, rid, state, academic = configured(tmp_path, monkeypatch, receipt=receipt, dependency=dependency)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        apply(h, rid, state, {'local_gap': gap})
    assert state == before and not academic.calls and not rt.db.list_tool_calls(rid)


def test_empty_successful_local_receipt_can_support_an_explicit_gap(tmp_path, monkeypatch):
    rt, h, rid, state, academic = configured(tmp_path, monkeypatch,
        receipt={'ok': True, 'tool': 'paper.search', 'evidence': []})
    apply(h, rid, state, {'local_gap': 'No retrieved text addresses recent temporal alignment.'})
    assert len(academic.calls) == 1 and academic.calls[0][2] == {}
    artifact = state['work']['artifacts'][state['work']['tasks']['search']['output']]
    basis = artifact['content']['local_discovery_basis']
    assert basis['artifact_ids'] == [state['work']['tasks']['local']['output']]
    assert basis['gap_status'] == 'MODEL_ASSESSMENT_NOT_VERIFIED'
    assert 'local_gap' not in rt.db.list_tool_calls(rid)[0]['args']


def test_local_bilingual_queries_use_registered_search_and_preserve_text(tmp_path, monkeypatch):
    from paperpilot_agentic.prototype_analysis import spans
    from paperpilot_agentic.prototype_workspace import digest
    rt, h, rid, state, academic = configured(tmp_path, monkeypatch)
    calls = []

    class Local:
        def search_many(self, queries, *, top_k, paper_sha256s=None):
            calls.append((queries, top_k))
            return [{'content': 'Temporal alignment reduces false changes.',
                     'title': 'Local paper', 'page': 2, 'paper_sha256': 'a'*64}]

    # The registered closure uses the injected adapter at registration time.
    from paperpilot_agentic.builtin_tools import _search_handler
    spec = rt.registry.get('paper.search')
    monkeypatch.setattr(spec, 'handler', _search_handler(Local()))
    h._apply(rid, state, {'action': 'work', 'reason': 'bilingual retrieval', 'arguments': {
        'task_id': 'local', 'content': {'scope': 'local', 'query': 'building change detection',
          'queries': ['building change detection', '建筑变化检测'], 'top_k': 3}}}, 2)
    identity = state['work']['tasks']['local']['output']
    artifact = state['work']['artifacts'][identity]
    assert calls == [(['building change detection', '建筑变化检测'], 3)]
    assert not academic.calls and artifact['content']['tool'] == 'paper.search'
    quote = spans(identity, artifact)[0]
    assert quote['evidence_type'] == 'TEXT'
    assert quote['text'] == 'Temporal alignment reduces false changes.'
    assert digest(artifact) == identity


@pytest.mark.parametrize('queries', [[], ['wrong primary'], ['building change detection']*2,
    ['building change detection', ''], ['building change detection']+['q']*3])
def test_invalid_local_query_lists_do_not_dispatch(tmp_path, monkeypatch, queries):
    rt, h, rid, state, _ = configured(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Local queries'):
        h._apply(rid, state, {'action': 'work', 'reason': 'query check', 'arguments': {
            'task_id': 'local', 'content': {'scope': 'local', 'query': 'building change detection',
                                          'queries': queries, 'top_k': 3}}}, 2)
    assert not rt.db.list_tool_calls(rid)
