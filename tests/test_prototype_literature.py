import copy

import pytest

from paperpilot_agentic.models import ToolCallStatus
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_literature import check_design, CONTEXT_POLICY
from paperpilot_agentic.prototype_workspace import complete, plan, task_rows
from paperpilot_agentic.episode_runner import ResearchStopped
from test_prototype_workspace import Reader


class Academic:
    searches = 0
    fetches = 0

    def search(self, query, top_k):
        self.searches += 1
        return {'items': [{'title': 'Fixture paper', 'abstract': 'Metadata only', 'year': 2025,
                           'full_text_url': 'https://arxiv.org/pdf/2501.00001'}]}

    def fetch_full_text(self, url):
        self.fetches += 1
        return {'url': url, 'sha256': 'f'*64, 'pages': [
            {'page': 1, 'content': 'Original method describes temporal differences.'},
            {'page': 2, 'content': 'Ablation evidence is separate.'}]}


def setup(tmp_path, monkeypatch):
    academic = Academic()
    rt = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'db.sqlite3',
                              llm=Reader(), scholarly_adapter=academic)
    monkeypatch.setattr(rt.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'baseline'}])
    h = PrototypeHarness(rt)
    rid = h.start('research goal', 'literature-test-001', 'parent', ['source'],
                  literature_enabled=True)['run']['run_id']
    state = h._load(rid)
    state.pop('design_contract_policy')  # Historical fixtures predate typed binary design declarations.
    state.pop('literature_local_first_policy')  # Frozen pre-local-first protocol coverage.
    # Preserve coverage of the pre-analysis literal-quote task protocol.
    state.pop('literature_analysis_policy')
    state.pop('literature_review_policy')  # Frozen protocol predates claim review.
    state.pop('analysis_read_policy')  # Keep the historical short-preview coverage.
    state.pop('quote_handle_policy')  # Historical protocol used full canonical quote IDs.
    state.pop('analysis_workset_policy')  # Historical protocol had per-source reading allowances.
    plan(state['work'], [{'id': key, 'goal': key, 'capability': kind, 'depends_on': deps, 'acceptance': 'fixture'}
                        for key, kind, deps in [('read', 'read', []), ('search', 'search_literature', []),
                            ('paper', 'read_literature', ['search']),
                            ('design', 'design', ['read', 'search', 'paper'])]], enforce_capabilities=True)
    complete(state['work'], 'read', 'read', state['sources']['source'])
    return rt, h, rid, state, academic


def work(h, rid, state, key, content):
    return h._apply(rid, state, {'action': 'work', 'reason': 'fixture',
                               'arguments': {'task_id': key, 'content': content}}, 1)


def search(h, rid, state):
    return work(h, rid, state, 'search', {'scope': 'external', 'query': 'building change detection 2025', 'top_k': 3})


def test_existing_scholarly_tools_supply_exact_evidence_and_page_selection(tmp_path, monkeypatch):
    rt, h, rid, state, academic = setup(tmp_path, monkeypatch)
    search(h, rid, state)
    work(h, rid, state, 'paper', {'url': 'https://arxiv.org/pdf/2501.00001', 'pages': [1]})
    identity = state['work']['tasks']['paper']['output']
    content = state['work']['artifacts'][identity]['content']
    assert content['evidence'][0]['content'] == 'Original method describes temporal differences.'
    assert len(content['evidence']) == 1 and content['available_pages'] == [1, 2]
    call = rt.db.get_tool_call(content['tool_call_id'])
    assert len(call['result']['evidence']) == 2  # raw receipt not truncated
    assert academic.searches == academic.fetches == 1
    work(h, rid, state, 'design', {'literature_basis': [{'artifact_id': identity,
        'quote': 'temporal differences', 'interpretation': 'motivation, not a proven new result'}]})


def test_completed_search_recovery_does_not_refetch_or_duplicate_counters(tmp_path, monkeypatch):
    rt, h, rid, state, academic = setup(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    search(h, rid, state)
    counters = rt.db.get_run(rid)['used_tool_calls']
    search(h, rid, before)  # simulate receipt saved but workspace commit lost
    assert academic.searches == 1 and rt.db.get_run(rid)['used_tool_calls'] == counters
    assert before['work']['tasks']['search']['output'] == state['work']['tasks']['search']['output']


def test_unknown_tool_request_stops_instead_of_replaying(tmp_path, monkeypatch):
    rt, h, rid, state, academic = setup(tmp_path, monkeypatch)
    before = copy.deepcopy(state)
    search(h, rid, state)
    call = rt.db.list_tool_calls(rid)[0]
    rt.db.update_tool_call(call['tool_call_id'], ToolCallStatus.RUNNING)
    with pytest.raises(ResearchStopped, match='unknown'):
        search(h, rid, before)
    assert academic.searches == 1


@pytest.mark.parametrize('url', ['https://arxiv.org/pdf/2501.99999', 'http://localhost/secrets'])
def test_read_cannot_fetch_an_undiscovered_url(tmp_path, monkeypatch, url):
    _, h, rid, state, academic = setup(tmp_path, monkeypatch)
    search(h, rid, state)
    with pytest.raises(ValueError, match='direct search dependency'):
        work(h, rid, state, 'paper', {'url': url, 'pages': [1]})
    assert academic.fetches == 0


@pytest.mark.parametrize('case', ['invented_quote', 'foreign_artifact', 'metadata_only'])
def test_design_rejects_fabricated_provenance_and_metadata_as_original_text(tmp_path, monkeypatch, case):
    _, h, rid, state, _ = setup(tmp_path, monkeypatch)
    search(h, rid, state)
    work(h, rid, state, 'paper', {'url': 'https://arxiv.org/pdf/2501.00001', 'pages': [1]})
    row = {'artifact_id': state['work']['tasks']['paper']['output'],
           'quote': 'temporal differences', 'interpretation': 'hypothesis'}
    if case == 'invented_quote':
        row['quote'] = 'invented state of the art result'
    elif case == 'foreign_artifact':
        row['artifact_id'] = 'foreign'
    else:
        row.update(artifact_id=state['work']['tasks']['search']['output'], quote='Metadata only')
    with pytest.raises(ValueError) as rejected:
        check_design(state['work'], state['work']['tasks']['design'], {'literature_basis': [row]})
    if case == 'invented_quote':
        assert 'literature_basis[0]' in str(rejected.value)
        assert 'contiguous literal substring' in str(rejected.value)


def test_legacy_tasks_cannot_silently_gain_literature_capabilities(tmp_path, monkeypatch):
    _, h, _, state, _ = setup(tmp_path, monkeypatch)
    state.pop('literature_policy')
    before = copy.deepcopy(state)
    rows = [{'id': 'search', 'goal': 'search', 'capability': 'search_literature',
             'depends_on': [], 'acceptance': 'fixture'}]
    with pytest.raises(ValueError, match='not registered'):
        h._apply('run', state, {'action': 'plan', 'reason': 'test', 'arguments': {'tasks': rows}}, 1)
    assert state == before


def test_large_full_text_is_projected_without_changing_original_artifacts(tmp_path, monkeypatch):
    import json
    from paperpilot_agentic.prototype_workspace import digest
    _, h, rid, state, _ = setup(tmp_path, monkeypatch)
    search(h, rid, state)
    complete(state['work'], 'paper', 'read_literature', {'ok': True, 'evidence': [
        {'evidence_type': 'TEXT', 'page': 1, 'content': 'original '*12000}]})
    before = copy.deepcopy(state['work']['artifacts'])
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    identity = state['work']['tasks']['paper']['output']
    view = payload['context_artifacts'][identity]['content']
    assert sum(len(e['content']) for e in view['evidence']) <= 1000
    assert view['context_view']['evidence_character_limit'] == 1000
    assert view['context_view']['raw_artifact_id'] == identity
    assert state['work']['artifacts'] == before
    assert digest(before[identity]) == identity


def test_identical_pages_do_not_trigger_a_second_tool_call(tmp_path, monkeypatch):
    _, h, rid, state, academic = setup(tmp_path, monkeypatch)
    search(h, rid, state)
    work(h, rid, state, 'paper', {'url': 'https://arxiv.org/pdf/2501.00001', 'pages': [1]})
    rows = task_rows(state['work'])
    rows.append({'id': 'duplicate', 'goal': 'duplicate', 'capability': 'read_literature',
                 'depends_on': ['search'], 'acceptance': 'fixture'})
    plan(state['work'], rows)
    state['task_context_policy'] = None  # isolate duplicate admission from serial selection
    with pytest.raises(ValueError, match='Identical paper pages'):
        work(h, rid, state, 'duplicate', {'url': 'https://arxiv.org/pdf/2501.00001', 'pages': [1]})
    assert academic.fetches == 1


def test_context_upgrade_refuses_any_turn_or_unknown_request_and_preserves_budget(tmp_path, monkeypatch):
    from scripts.recover_prototype_literature_context import recover
    rt, h, rid, state, _ = setup(tmp_path, monkeypatch)
    state.pop('literature_context_policy')
    state.update(phase='FAILED', stop_reason='ValueError: Required context exceeds window; split the task')
    h._save(rid, state)
    before = copy.deepcopy(rt.db.get_run(rid))
    recover(h, rid)
    assert h._load(rid)['literature_context_policy'] == CONTEXT_POLICY
    after = rt.db.get_run(rid)
    assert (after['budget'], after['used_llm_calls'], after['used_tokens']) == (
        before['budget'], before['used_llm_calls'], before['used_tokens'])
    with pytest.raises(ValueError):
        recover(h, rid)


@pytest.mark.parametrize('condition', ['unknown', 'registered_turn'])
def test_context_upgrade_never_replays_an_existing_or_unknown_decision(tmp_path, monkeypatch, condition):
    from scripts.recover_prototype_literature_context import recover
    rt, h, rid, state, _ = setup(tmp_path, monkeypatch)
    state.pop('literature_context_policy')
    state.update(phase='FAILED', stop_reason='ValueError: Required context exceeds window; split the task')
    h._save(rid, state)
    if condition == 'unknown':
        monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda _: {'unknown_results': 1})
    else:
        with rt.db.transaction() as conn:
            conn.execute('INSERT INTO research_episode_turns VALUES(?,?,?,?,?,?,?,?)',
                         (rid, 1, 'lead', '{}', 'STARTED', 0, None, None))
    before = copy.deepcopy(h._load(rid))
    with pytest.raises(ValueError):
        recover(h, rid)
    assert h._load(rid) == before
