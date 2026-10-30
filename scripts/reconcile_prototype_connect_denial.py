"""Host-only reconciliation of a recorded Windows local socket denial.

No timeout/read/write/HTTP/ambiguous connection failure qualifies. The original
attempt and conservative token reservation remain charged to the same task.
This function never sends a request; a subsequent explicit advance may do so.
"""
import json


def reconcile(harness, rid):
    from paperpilot_agentic.research_task import task_lock
    from paperpilot_agentic.reference_methods import encoded
    from paperpilot_agentic.models import utc_now

    with task_lock(harness.runtime.workspace, rid) as acquired:
        if not acquired:
            raise RuntimeError('Prototype busy')
        state = harness._load(rid)
        harness._guard(rid, state)
        if state['phase'] != 'UNKNOWN':
            raise ValueError('Select an unknown local connection attempt')
        turn = state['turn'] + 1
        stage = f'research.{state["role"]}.{turn}'
        with harness.db.transaction() as conn:
            episode = conn.execute('SELECT * FROM research_episode_turns WHERE run_id=? AND turn=?',
                                   (rid, turn)).fetchone()
            attempts = conn.execute("SELECT * FROM llm_attempts WHERE run_id=? AND state IN ('STARTED','UNKNOWN')",
                                    (rid,)).fetchall()
            if (not episode or episode['state'] != 'STARTED' or episode['raw_json'] is not None
                    or len(attempts) != 1 or attempts[0]['state'] != 'UNKNOWN'
                    or attempts[0]['stage'] != stage or not attempts[0]['llm_call_id']):
                raise ValueError('Ambiguous attempt/turn; recovery forbidden')
            attempt = attempts[0]
            failure = json.loads(attempt['failure_json'] or '{}')
            chain = failure.get('cause_chain', [])
            if (failure.get('error_type') != 'APIConnectionError'
                    or failure.get('transport_phase') != 'connect' or failure.get('http_status') is not None
                    or [row.get('type') for row in chain] != ['APIConnectionError', 'ConnectError', 'ConnectError', 'PermissionError']
                    or chain[-1].get('errno') != 13 or chain[-1].get('winerror') != 10013):
                raise ValueError('Only an exact local socket PermissionError/10013 chain qualifies')
            response = conn.execute("SELECT 1 FROM events WHERE run_id=? AND event_id>? AND event_type='llm.response.saved' LIMIT 1",
                                    (rid, episode['before_event'])).fetchone()
            if response:
                raise ValueError('A response exists; apply it rather than redispatching')
            record = {'attempt_id': attempt['attempt_id'], 'turn': turn,
                      'basis': 'OS denied socket connection before HTTP request transport',
                      'original_failure': failure, 'budget_reset': False,
                      'reservation_retained': True, 'time': utc_now()}
            record['previous_stop_reason'] = state.pop('stop_reason', None)
            failure['host_reconciliation'] = record['basis']
            conn.execute("UPDATE llm_attempts SET state='FAILED',failure_json=? WHERE attempt_id=?",
                         (json.dumps(failure), attempt['attempt_id']))
            conn.execute("UPDATE research_episode_turns SET state='NOT_DISPATCHED' WHERE run_id=? AND turn=?",
                         (rid, turn))
            state['phase'] = 'READY'
            state.setdefault('local_network_reconciliations', []).append(record)
            conn.execute('UPDATE research_episodes SET state_json=? WHERE run_id=?', (encoded(state).decode(), rid))
            conn.execute("UPDATE runs SET status='RUNNING',error=NULL,updated_at=? WHERE run_id=?", (utc_now(), rid))
            conn.execute('INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)',
                         (rid, 'research.local_connect_denial_reconciled', encoded(record).decode(), utc_now()))
        harness._save(rid, state)  # Refresh view; canonical transition above is atomic.
        return record
