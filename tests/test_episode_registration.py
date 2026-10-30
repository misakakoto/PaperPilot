import sqlite3

import pytest

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget


def register(db, rid, **kwargs):
    return db.create_step_tool_call(run_id=rid, tool_name='research.plan', args=kwargs or {'tasks': []},
        risk_level='LOW', idempotency_key='owned-logical-action', max_attempts=1)


def test_atomic_registration_rolls_back_step_counters_and_events(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.db')
    rid = db.create_run('fixture', Budget())
    events = db.list_events(run_id=rid)
    with db.transaction() as conn:
        conn.execute("""CREATE TRIGGER abort_tool BEFORE INSERT ON tool_calls BEGIN
            SELECT RAISE(ABORT,'injected registration failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match='injected'):
        register(db, rid)
    assert db.list_steps(rid) == [] and db.list_tool_calls(rid) == []
    assert db.get_run(rid)['used_steps'] == db.get_run(rid)['used_tool_calls'] == 0
    assert db.list_events(run_id=rid) == events
    with db.transaction() as conn:
        conn.execute('DROP TRIGGER abort_tool')
    call, reused = register(db, rid)
    assert not reused and db.get_tool_call(call)['run_id'] == rid
    assert db.get_run(rid)['used_steps'] == db.get_run(rid)['used_tool_calls'] == 1


def test_reusing_registration_checks_binding_without_new_step_or_event(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.db')
    rid = db.create_run('fixture', Budget())
    first, reused = register(db, rid, channels=32)
    assert not reused
    before = db.get_run(rid)
    steps, events = db.list_steps(rid), db.list_events(run_id=rid)
    assert register(db, rid, channels=32) == (first, True)
    with pytest.raises(RuntimeError, match='binding'):
        register(db, rid, channels=48)
    assert db.get_run(rid) == before
    assert db.list_steps(rid) == steps and db.list_events(run_id=rid) == events
