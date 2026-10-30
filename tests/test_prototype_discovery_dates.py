import copy

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from test_prototype_workspace import Reader


class DatedAcademic:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, **options):
        self.calls.append((query, top_k, options))
        return {'items': [{'source_id': 'paper', 'title': 'Dated fixture paper', 'year': 2026,
                           'publication_date': '2026-01-02', 'date_basis': 'publication_date'}],
                'search_scope': options, 'errors': [], 'providers': ['fixture']}


def prepared(tmp_path, monkeypatch):
    academic = DatedAcademic()
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db',
                                   llm=Reader(), scholarly_adapter=academic)
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'baseline'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('recent research', 'dated-discovery-fixture', 'parent', ['source'],
                         literature_enabled=True)['run']['run_id']
    state = harness._load(rid)
    state.pop('literature_local_first_policy')  # Frozen pre-local-first protocol coverage.
    state['research_date'] = '2026-10-24'
    harness._apply(rid, state, {'action': 'plan', 'reason': 'search', 'arguments': {'tasks': [
        {'id': 'search', 'goal': 'recent papers', 'capability': 'search_literature',
         'depends_on': [], 'acceptance': 'date-scoped metadata'}]}}, 1)
    return runtime, harness, rid, state, academic


def apply(harness, rid, state, content):
    return harness._apply(rid, state, {'action': 'work', 'reason': 'dated request',
        'arguments': {'task_id': 'search', 'content': {'scope': 'external', 'query': 'building change detection',
                                                      'top_k': 3, **content}}}, 2)


def test_dates_reach_the_registered_scholarly_tool_and_remain_in_receipts(tmp_path, monkeypatch):
    rt, h, rid, state, academic = prepared(tmp_path, monkeypatch)
    apply(h, rid, state, {'published_from': '2025-01-01', 'sort': 'newest', 'require_full_text': True})
    expected = {'published_from': '2025-01-01', 'published_to': '2026-10-24',
                'sort': 'newest', 'require_full_text': True}
    assert academic.calls == [('building change detection', 3, expected)]
    artifact = state['work']['artifacts'][state['work']['tasks']['search']['output']]['content']
    assert artifact['output']['search_scope'] == expected
    evidence = artifact['evidence'][0]
    assert evidence['provenance']['publication_date'] == '2026-01-02'
    assert evidence['provenance']['search_scope'] == expected
    call = rt.db.get_tool_call(artifact['tool_call_id'])
    assert call['status'] == 'COMPLETED' and call['args']['published_to'] == '2026-10-24'


@pytest.mark.parametrize('arguments', [
    {'published_to': '2027-01-01'}, {'published_from': '2027-01-01'},
    {'published_from': '2026-02-30'}, {'published_to': None},
    {'published_from': '2026-01-01', 'published_to': '2025-01-01'},
    {'scope': 'local', 'sort': 'newest'}, {'sort': 'fast'}, {'require_full_text': 'yes'}])
def test_bad_or_future_date_requests_do_not_dispatch(tmp_path, monkeypatch, arguments):
    rt, h, rid, state, academic = prepared(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        apply(h, rid, state, arguments)
    assert not academic.calls and state == before
    assert not rt.db.list_tool_calls(rid)


def test_historical_tasks_do_not_gain_date_parameters_by_resubmission(tmp_path, monkeypatch):
    _, h, rid, state, academic = prepared(tmp_path, monkeypatch)
    state.pop('literature_discovery_policy')
    with pytest.raises(ValueError):
        apply(h, rid, state, {'published_from': '2025-01-01'})
    assert not academic.calls
