import copy

import pytest

from test_prototype_analysis import prepared


def blocked(tmp_path, monkeypatch):
    rt, h, rid, state, _ = prepared(tmp_path, monkeypatch)
    h._save(rid, state)
    original = h._context

    def context_stop(*args):
        raise ValueError('Required context exceeds window; split the task')

    monkeypatch.setattr(h, '_context', context_stop)
    h.advance(rid, max_turns=1)
    assert h._load(rid)['phase'] == 'FAILED'
    monkeypatch.setattr(h, '_context', original)
    return rt, h, rid


def test_context_recovery_preserves_paid_usage_and_work(tmp_path, monkeypatch):
    rt, h, rid = blocked(tmp_path, monkeypatch)
    before, work = rt.db.get_run(rid), copy.deepcopy(h._load(rid)['work'])
    h.recover_undispatched_context_stop(rid)
    state, after = h._load(rid), rt.db.get_run(rid)
    assert state['phase'] == 'READY' and state['work'] == work
    assert state['context_recoveries'][-1]['new_model_requests'] == 0
    assert state['analysis_workset_policy'] == 'shared-byte-workset-v1'
    assert {k: after[k] for k in ('budget', 'created_at', 'used_tokens', 'used_llm_calls')} == {
        k: before[k] for k in ('budget', 'created_at', 'used_tokens', 'used_llm_calls')}


@pytest.mark.parametrize('case', ['unknown', 'dispatch_event', 'no_budget'])
def test_context_recovery_refuses_uncertainty_or_insufficient_original_budget(tmp_path, monkeypatch, case):
    rt, h, rid = blocked(tmp_path, monkeypatch)
    original = rt.llm_runtime.budget_snapshot
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
