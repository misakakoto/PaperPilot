"""Explicit host upgrade after a proven pre-decision context overflow; no dispatch."""
from paperpilot_agentic.prototype_literature import POLICY, CONTEXT_POLICY
from paperpilot_agentic.research_task import task_lock
from paperpilot_agentic.reference_methods import encoded
from paperpilot_agentic.models import utc_now


def recover(harness, rid):
    with task_lock(harness.runtime.workspace, rid) as acquired:
        if not acquired:
            raise RuntimeError('Prototype busy')
        state = harness._load(rid)
        harness._guard(rid, state)
        if (state['phase'] != 'FAILED' or state.get('literature_policy') != POLICY
                or state.get('literature_context_policy') is not None
                or state.get('stop_reason') != 'ValueError: Required context exceeds window; split the task'):
            raise ValueError('Select an unprojected literature task stopped on context overflow')
        if harness.runtime.llm_runtime.budget_snapshot(rid)['unknown_results']:
            raise ValueError('Unknown model request; context upgrade cannot resume it')
        turn = state['turn'] + 1
        with harness.db.transaction() as conn:
            row = conn.execute('SELECT 1 FROM research_episode_turns WHERE run_id=? AND turn=?', (rid, turn)).fetchone()
            attempt = conn.execute('SELECT 1 FROM llm_attempts WHERE run_id=? AND stage=?',
                                   (rid, f'research.{state["role"]}.{turn}')).fetchone()
            if row or attempt:
                raise ValueError('A turn or dispatch already exists; inspect its original request')
            record = {'turn': turn, 'previous_stop_reason': state.pop('stop_reason'),
                      'context_policy': CONTEXT_POLICY, 'budget_reset': False, 'time': utc_now()}
            state.update(phase='READY', literature_context_policy=CONTEXT_POLICY)
            state.setdefault('research_date', harness.db.get_run(rid)['created_at'].split('T')[0])
            state.setdefault('context_recoveries', []).append(record)
            conn.execute('UPDATE research_episodes SET state_json=? WHERE run_id=?', (encoded(state).decode(), rid))
            conn.execute("UPDATE runs SET status='RUNNING',error=NULL,updated_at=? WHERE run_id=?", (utc_now(), rid))
            conn.execute('INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)',
                         (rid, 'research.literature_context_upgraded', encoded(record).decode(), utc_now()))
        harness._save(rid, state)
        return record
