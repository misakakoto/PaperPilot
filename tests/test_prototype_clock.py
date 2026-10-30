"""Approval waiting is excluded; unverifiable crash intervals remain charged."""
import copy
from types import SimpleNamespace

import pytest

from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.models import Budget


class Database:
    def __init__(self, metadata):
        self.run = {'status': 'PENDING', 'created_at': '2000-01-01T00:00:00+00:00',
                    'budget': Budget(max_seconds=60).to_dict(), 'metadata': metadata}

    def get_run(self, rid):
        return copy.deepcopy(self.run)

    def update_run_metadata(self, rid, values):
        self.run['metadata'].update(values)


def test_approved_next_day_can_call_model_with_remaining_active_budget():
    db = Database({'engine': 'research_prototype_v1', 'prototype_clock_policy': 'active-v1',
                   'research_active_seconds': 12, 'research_active_since': None})
    calls = []

    def invoke(*args, attempt_guard, **kwargs):
        attempt_guard()
        calls.append(1)
        return 'response'

    model = DurableResearchModel(SimpleNamespace(db=db, complete=invoke), 'run', stage='test')
    assert model.complete([], max_tokens=5) == 'response'
    assert calls == [1]
    db.run['metadata']['research_active_seconds'] = 61
    with pytest.raises(LLMBudgetExhausted, match='Time budget'):
        model.complete([], max_tokens=5)
    assert calls == [1]


def test_clock_accumulates_execution_but_excludes_clean_waits(monkeypatch):
    from paperpilot_agentic.prototype_clock import active_time
    now = [100.0]
    monkeypatch.setattr('paperpilot_agentic.prototype_clock.time.time', lambda: now[0])
    monkeypatch.setattr('paperpilot_agentic.research_task.time.time', lambda: now[0])
    db = Database({'prototype_clock_policy': 'active-v1', 'research_active_seconds': 3})
    with active_time(db, 'run'):
        now[0] += 7
    assert db.run['metadata']['research_active_seconds'] == 10
    assert db.run['metadata']['research_active_since'] is None
    now[0] += 86400
    with pytest.raises(RuntimeError):
        with active_time(db, 'run'):
            now[0] += 4
            raise RuntimeError('stop')
    assert db.run['metadata']['research_active_seconds'] == 14
    assert db.run['created_at'] == '2000-01-01T00:00:00+00:00'


def test_crash_interval_is_not_erased_and_legacy_budget_is_not_migrated(monkeypatch):
    from paperpilot_agentic.prototype_clock import active_time
    monkeypatch.setattr('paperpilot_agentic.prototype_clock.time.time', lambda: 100.0)
    monkeypatch.setattr('paperpilot_agentic.research_task.time.time', lambda: 100.0)
    db = Database({'prototype_clock_policy': 'active-v1', 'research_active_seconds': 5,
                   'research_active_since': 10})
    with active_time(db, 'run'):
        assert db.run['metadata']['research_active_seconds'] == 95
    assert db.run['metadata']['research_active_seconds'] == 95
    legacy = Database({'engine': 'research_prototype_v1'})
    before = copy.deepcopy(legacy.run)
    with active_time(legacy, 'run'):
        pass
    assert legacy.run == before


def test_episode_stops_clock_at_approval_and_checks_budget_before_execution(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    run = runtime.db.get_run(rid)
    assert run['metadata']['prototype_clock_policy'] == 'active-v1'
    assert run['metadata']['research_active_seconds'] > 0
    assert run['metadata']['research_active_since'] is None
    with runtime.db.transaction() as connection:
        connection.execute('UPDATE runs SET created_at=? WHERE run_id=?',
                           ('2000-01-01T00:00:00+00:00', rid))
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'WAITING_APPROVAL'
    assert runtime.db.get_run(rid)['used_llm_calls'] == run['used_llm_calls']
    assert runtime.db.get_run(rid)['metadata']['research_active_since'] is None
    runtime.approve(harness._load(rid)['pending']['approval'])
    runtime.db.update_run_metadata(rid, {'research_active_seconds': run['budget']['max_seconds'] + 1})
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run',
                        lambda *args: pytest.fail('Budget exhausted; execution must not start'))
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'FAILED'
    assert 'Time budget exhausted' in harness._load(rid)['stop_reason']
    assert runtime.db.get_run(rid)['metadata']['research_active_since'] is None
