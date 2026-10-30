import copy
import json

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.prototype_workspace import plan, put
from test_token_evidence import fixture_index
from test_prototype_workspace import Reader


def setup(tmp_path, monkeypatch):
    workspace, _, _, adapter, base = fixture_index(tmp_path)
    rt = ResearchAgentRuntime(workspace=workspace, db_path=tmp_path / 'runs.db', search_adapter=base, llm=Reader())
    monkeypatch.setattr(rt.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'owned baseline'}])
    h = PrototypeHarness(rt)
    rid = h.start('Analyze Alpha research assumptions', 'parent-reading', 'parent', ['source'],
                  literature_enabled=True, retrieval_profile='token_evidence_pilot', runtime_profile='research-runtime-v3')['run']['run_id']
    state = h._load(rid)
    plan(state['work'], [
        {'id': 'search', 'goal': 'Find Alpha method', 'capability': 'search_literature', 'depends_on': [], 'acceptance': 'actual local hits'},
        {'id': 'read', 'goal': 'Read original parent paragraphs', 'capability': 'read_literature', 'depends_on': ['search'], 'acceptance': 'source-bound text'},
        {'id': 'audit', 'goal': 'Inspect full source locations', 'capability': 'read_literature', 'depends_on': ['search', 'read'], 'acceptance': 'full source receipt'},
        {'id': 'analysis', 'goal': 'Interpret source with gaps', 'capability': 'analyze_literature', 'depends_on': ['read'], 'acceptance': 'original quote IDs'}],
        enforce_capabilities=True)
    apply(h, rid, state, 'search', {'scope': 'local', 'query': 'Alpha', 'top_k': 3})
    result = state['work']['artifacts'][state['work']['tasks']['search']['output']]['content']
    ids = [e['provenance']['stable_id'] for e in result['evidence']]
    return rt, h, rid, state, ids, adapter


def apply(h, rid, state, task_id, content):
    return h._apply(rid, state, {'action': 'work', 'reason': 'Read source-bound research evidence',
        'arguments': {'task_id': task_id, 'content': content}}, 2)


def request(ids, **changes):
    return {'paper_sha256': 'a' * 64, 'child_ids': ids, 'view': 'source_handles', 'max_tokens': 2048, **changes}


def test_parent_read_bridge_and_full_locator_audit_feed_analysis_quotes(tmp_path, monkeypatch):
    rt, h, rid, state, ids, _ = setup(tmp_path, monkeypatch)
    assert state['literature_parent_read_policy'] == 'discovered-parent-reading-v1'
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert 'child_ids' in messages[0]['content']
    search = next(a for a in payload['context_artifacts'].values() if a['kind'] == 'search_literature')
    assert [r['child_ids'][0] for r in search['content']['discovery_catalog']] == ids
    assert 'quote_catalog' not in search['content'] and 'evidence' not in search['content']
    assert search['content']['context_view']['raw_artifact_id'] == state['work']['tasks']['search']['output']
    apply(h, rid, state, 'read', request(ids))
    read_id = state['work']['tasks']['read']['output']
    read = state['work']['artifacts'][read_id]
    assert read['content']['tool'] == 'paper.read_context'
    assert read['content']['output']['view_policy'] == 'source-handles-v1'
    assert read['content']['local_read_request'] == request(ids)
    apply(h, rid, state, 'audit', request(ids, view='full', max_tokens=4096))
    audit = state['work']['artifacts'][state['work']['tasks']['audit']['output']]
    assert audit['content']['evidence'][0]['provenance']['source_segments']
    original = copy.deepcopy(state)
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert payload['active_task_id'] == 'analysis'
    assert any(a['content'].get('pages') for a in payload['context_artifacts'].values() if a['kind'] == 'read_literature')
    assert state == original
    assert [c['tool_name'] for c in rt.db.list_tool_calls(rid)] == ['paper.search', 'paper.read_context', 'paper.read_context']


@pytest.mark.parametrize('kind', ['foreign', 'undiscovered', 'legacy', 'wrong_profile', 'changed_index', 'bad_view'])
def test_parent_read_rejects_unowned_or_changed_inputs_before_dispatch(tmp_path, monkeypatch, kind):
    rt, h, rid, state, ids, adapter = setup(tmp_path, monkeypatch)
    args = request(ids)
    if kind in {'foreign', 'undiscovered'}:
        sha = 'b' * 64 if kind == 'foreign' else 'a' * 64
        args['child_ids'] = [next(i for i, c in adapter.evidence['children'].items()
                                  if c['paper_sha256'] == sha and i not in ids)]
    elif kind == 'legacy':
        state.pop('literature_parent_read_policy', None)
    elif kind == 'wrong_profile':
        state['retrieval_profile'] = 'legacy'
    elif kind == 'changed_index':
        state['retrieval_identity'] = 'changed'
    else:
        args['view'] = 'model_summary'
    before = len(rt.db.list_tool_calls(rid))
    with pytest.raises(ValueError):
        apply(h, rid, state, 'read', args)
    assert len(rt.db.list_tool_calls(rid)) == before
    assert state['work']['tasks']['read']['status'] != 'DONE'


def test_compiler_stop_recovers_same_task_and_budget_without_model_dispatch(tmp_path, monkeypatch):
    rt, h, rid, state, ids, _ = setup(tmp_path, monkeypatch)
    apply(h, rid, state, 'read', request(ids))
    h._save(rid, state)
    snapshot = rt.llm_runtime.budget_snapshot
    monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda run: {**snapshot(run), 'remaining_tokens': 1})
    h.advance(rid, max_turns=1)
    assert h._load(rid)['phase'] == 'FAILED'
    assert h._load(rid)['context_manifest']['reason'] == 'REQUIRED_CONTEXT_EXCEEDS_BUDGET'
    monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', snapshot)
    before, work = rt.db.get_run(rid), copy.deepcopy(h._load(rid)['work'])
    h.recover_undispatched_context_stop(rid)
    state, after = h._load(rid), rt.db.get_run(rid)
    assert state['phase'] == 'READY' and state['work'] == work
    assert state['context_manifest']['status'] == 'READY'
    assert state['context_recoveries'][-1]['new_model_requests'] == 0
    assert {k: before[k] for k in ('budget', 'created_at', 'used_tokens', 'used_llm_calls')} == {
        k: after[k] for k in ('budget', 'created_at', 'used_tokens', 'used_llm_calls')}


def test_parent_navigation_keeps_external_discovery_readable(tmp_path, monkeypatch):
    _, h, rid, state, _, _ = setup(tmp_path, monkeypatch)
    url = 'https://example.org/paper.pdf'
    identity = put(state['work'], 'search', 'search_literature', {'ok': True, 'tool': 'scholar.search',
        'output': {'candidates': [{'title': 'External source', 'full_text_url': url}]}, 'evidence': []})
    state['work']['tasks']['search'].update(status='DONE', output=identity)
    _, messages = h._context(rid, state)
    payload = json.loads(messages[-1]['content'])
    assert payload['context_artifacts'][identity]['content']['output']['candidates'][0]['full_text_url'] == url
    assert 'External reading' in messages[0]['content']


@pytest.mark.parametrize('case', ['unknown', 'dispatch_event', 'no_budget'])
def test_compiler_recovery_refuses_unknown_dispatch_or_insufficient_budget(tmp_path, monkeypatch, case):
    rt, h, rid, state, ids, _ = setup(tmp_path, monkeypatch)
    apply(h, rid, state, 'read', request(ids))
    h._save(rid, state)
    original = rt.llm_runtime.budget_snapshot
    monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda run: {**original(run), 'remaining_tokens': 1})
    h.advance(rid, max_turns=1)
    monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', original)
    if case == 'unknown':
        monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda run: {**original(run), 'unknown_results': 1})
    elif case == 'dispatch_event':
        rt.db.emit_event(rid, 'llm.attempt.started', {'stage': 'research.lead.1'})
    else:
        monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda run: {**original(run), 'remaining_tokens': 1})
    before = h._load(rid)
    with pytest.raises(ValueError):
        h.recover_undispatched_context_stop(rid)
    assert h._load(rid) == before
