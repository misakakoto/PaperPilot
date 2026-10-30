import copy

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.episode_runner import ResearchStopped
from test_prototype_workspace import Reader


def prepared(tmp_path, monkeypatch, *, sources=1, reads=1):
    rt = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    evidence = [{'evidence_id': 'source'+str(i), 'content': 'Immutable source',
                 'provenance': {'license_text': 'Original license'}} for i in range(sources)]
    monkeypatch.setattr(rt.db, 'list_evidence', lambda _: evidence)
    h = PrototypeHarness(rt)
    rid = h.start('research', 'source-read', 'parent', [e['evidence_id'] for e in evidence])['run']['run_id']
    state = h._load(rid)
    state['automatic_source_read_policy'] = 'single-selected-source-read-v1'
    h._apply(rid, state, {'action': 'plan', 'reason': 'fixture', 'arguments': {'tasks': [
        {'id': 'read'+str(i), 'goal': 'read', 'capability': 'read', 'depends_on': [], 'acceptance': 'source'}
        for i in range(reads)]}}, 1)
    return rt, h, rid, state, evidence


def test_mandatory_single_source_read_preserves_bytes_license_and_budget(tmp_path, monkeypatch):
    rt, h, rid, state, evidence = prepared(tmp_path, monkeypatch)
    before = rt.db.get_run(rid)
    observation = {'status': 'OK'}
    h._after_action(rid, state, 'host', observation)
    artifact = state['work']['artifacts'][state['work']['tasks']['read0']['output']]
    assert artifact['content'] == evidence[0]
    assert 'host_source_read' in observation
    assert rt.db.get_run(rid)['used_llm_calls'] == before['used_llm_calls']
    assert rt.db.get_run(rid)['budget'] == before['budget']
    work = copy.deepcopy(state['work'])
    h._after_action(rid, state, 'host', {'status': 'OK'})
    assert state['work'] == work and not rt.db.list_tool_calls(rid)


@pytest.mark.parametrize('sources,reads', [(2, 1), (1, 2)])
def test_host_does_not_guess_ambiguous_source_assignments(tmp_path, monkeypatch, sources, reads):
    _, h, rid, state, _ = prepared(tmp_path, monkeypatch, sources=sources, reads=reads)
    before = copy.deepcopy(state['work'])
    h._after_action(rid, state, 'host', {'status': 'OK'})
    assert state['work'] == before


def test_unknown_requests_prevent_host_progress(tmp_path, monkeypatch):
    rt, h, rid, state, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda _: {'unknown_results': 1})
    with pytest.raises(ResearchStopped):
        h._after_action(rid, state, 'host', {'status': 'OK'})
    assert not state['work']['artifacts']
