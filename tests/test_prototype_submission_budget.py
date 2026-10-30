import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.workbench import ResearchWorkbench
from test_prototype_workspace import Reader
from test_workbench import Background, completed


def submission(tmp_path, monkeypatch):
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path / 'runtime.db', llm=Reader())
    background = Background()
    workbench = ResearchWorkbench(runtime, background)
    parent = completed(workbench, cid='budget-case')
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda rid, **kw: [
        {'evidence_id': 'source', 'content': 'registered source'}] if rid == parent else [])
    payload = {'action': 'research_prototype', 'request_id': 'bounded-prototype',
               'conversation_id': 'budget-case', 'parent_run_id': parent,
               'source_ids': ['source'], 'question': 'Design a bounded prototype',
               'budget': {'max_llm_calls': 2, 'max_tokens': 10000,
                          'synthesis_reserve_tokens': 0, 'repair_reserve_tokens': 0}}
    return runtime, background, workbench, payload


def test_workbench_prototype_honors_initial_cumulative_budget(tmp_path, monkeypatch):
    runtime, background, wb, payload = submission(tmp_path, monkeypatch)
    rid = wb.submit(payload)['run_id']
    budget = runtime.db.get_run(rid)['budget']
    assert budget['max_tokens'] == 10000
    assert budget['max_llm_calls'] == 2
    assert runtime.llm_runtime.budget_snapshot(rid)['remaining_tokens'] == 10000
    assert PrototypeHarness(runtime)._load(rid)['limits']['max_turns'] == 2
    assert wb.submit(payload)['run_id'] == rid
    assert background.submitted == [rid]
    assert runtime.db.get_run(rid)['used_llm_calls'] == 0


@pytest.mark.parametrize('change', [{'max_llm_calls': 17}, {'max_tokens': -1},
                                  {'max_tokens': True}, {'max_seconds': float('nan')},
                                  {'unlimited_llm_usage': True}])
def test_invalid_prototype_budget_does_not_create_a_task(tmp_path, monkeypatch, change):
    runtime, background, wb, payload = submission(tmp_path, monkeypatch)
    payload['budget'].update(change)
    with runtime.db.connect() as conn:
        before = conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
    with pytest.raises(ValueError):
        wb.submit(payload)
    with runtime.db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == before
    assert background.submitted == []


def test_direct_submission_cannot_reuse_identity_with_another_budget(tmp_path, monkeypatch):
    runtime, _, _, payload = submission(tmp_path, monkeypatch)
    h = PrototypeHarness(runtime)
    args = (payload['question'], payload['request_id'], payload['parent_run_id'], payload['source_ids'])
    result = h.start(*args, budget=payload['budget'])
    with pytest.raises(ValueError, match='identity changed'):
        h.start(*args, budget={**payload['budget'], 'max_tokens': 12000})
    assert runtime.db.get_run(result['run']['run_id'])['budget']['max_tokens'] == 10000
