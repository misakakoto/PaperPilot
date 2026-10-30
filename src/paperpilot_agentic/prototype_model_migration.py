"""Explicit model migration after a durable, known quota rejection."""
import copy
import json

from .models import RunStatus
from .reference_methods import encoded
from .research_identity import research_model_identity
from .research_task import task_lock


def provider_metadata(rt, run_id, failed_model):
    metadata = copy.deepcopy(rt.db.get_run(run_id)['metadata'])
    identity = getattr(rt.llm_runtime.model, 'policy_identity', None)
    if not callable(identity):
        return metadata
    current = identity(metadata.get('execution_profile', 'frozen_acceptance'))
    previous = metadata.get('model_policy')
    if previous is not None and previous != current:
        if (failed_model not in previous['models'] or any(previous[k] != current[k]
                for k in ('provider', 'endpoint_sha256', 'profile'))):
            raise ValueError('Provider migration may only change the model on the same endpoint/profile')
        metadata.setdefault('model_policy_history', []).append(previous)
    metadata['model_policy'] = current
    return metadata


def migrate_quota_model(harness, run_id):
    rt = harness.runtime
    with task_lock(rt.workspace, run_id) as acquired:
        if not acquired:
            raise ValueError('Prototype busy')
        state = harness._load(run_id)
        if state['phase'] != 'FAILED' or rt.llm_runtime.budget_snapshot(run_id)['unknown_results']:
            raise ValueError('Migration requires a stopped task without unknown requests')
        index = state['turn']+1
        with harness.db.connect() as conn:
            turn = conn.execute('SELECT * FROM research_episode_turns WHERE run_id=? AND turn=?',
                                (run_id, index)).fetchone()
            attempts = conn.execute('SELECT * FROM llm_attempts WHERE run_id=? AND stage=? ORDER BY ordinal',
                                   (run_id, f'research.{state["role"]}.{index}')).fetchall()
        if (not turn or turn['state'] != 'STARTED' or turn['raw_json'] is not None or len(attempts) != 1
                or attempts[0]['state'] != 'FAILED'):
            raise ValueError('Require one durable failed provider attempt without a response')
        failure = json.loads(attempts[0]['failure_json'] or '{}')
        if (failure.get('category') != 'BILLING_OR_QUOTA' or failure.get('http_status') != 403
                or failure.get('provider_code') != 'AllocationQuota.FreeTierOnly'):
            raise ValueError('Only a known free-tier quota rejection permits this migration')
        if index >= state['limits']['max_turns']:
            raise ValueError('No original decision budget remains')
        identity = research_model_identity(rt)
        if identity == state['model_protocol']:
            raise ValueError('Select a different explicitly configured model')
        updated = copy.deepcopy(state)
        updated['model_protocol'] = identity
        harness._guard(run_id, updated)  # All verifier, source, cancellation and contract checks remain.
        updated.setdefault('model_migrations', []).append({
            'from_identity': state['model_protocol'], 'to_identity': identity,
            'failed_model': attempts[0]['model'], 'new_model': rt.llm_runtime.model.model,
            'failed_turn': index, 'attempt_id': attempts[0]['attempt_id'], 'reason': failure,
            'budget_reset': False, 'previous_stop_reason': state.get('stop_reason')})
        # Consume the failed decision slot; never overwrite/replay its request.
        updated.update(turn=index, phase='READY')
        updated.pop('stop_reason', None)
        metadata = provider_metadata(rt, run_id, attempts[0]['model'])
        with harness.db.transaction() as conn:
            conn.execute("UPDATE research_episode_turns SET state='PROVIDER_REJECTED' WHERE run_id=? AND turn=?",
                         (run_id, index))
            conn.execute('UPDATE research_episodes SET state_json=? WHERE run_id=?', (encoded(updated).decode(), run_id))
            conn.execute('UPDATE runs SET metadata_json=? WHERE run_id=?', (encoded(metadata).decode(), run_id))
        harness._save(run_id, updated)
        harness.db.update_run_status(run_id, RunStatus.RUNNING)
        harness.db.emit_event(run_id, 'prototype.model_migrated', updated['model_migrations'][-1])
    return harness.status(run_id)


def reconcile_provider_binding(harness, run_id):
    """Finish a retained migration stopped before any replacement dispatch."""
    rt = harness.runtime
    with task_lock(rt.workspace, run_id) as acquired:
        if not acquired:
            raise ValueError('Prototype busy')
        state = harness._load(run_id)
        migrations = state.get('model_migrations', [])
        if (state['phase'] != 'FAILED' or not migrations
                or not state.get('stop_reason', '').startswith('ValueError: Model policy changed;')
                or migrations[-1]['to_identity'] != research_model_identity(rt)
                or rt.llm_runtime.budget_snapshot(run_id)['unknown_results']):
            raise ValueError('Select a known undispatched provider-binding migration stop')
        index = state['turn']+1
        with harness.db.connect() as conn:
            row = conn.execute('SELECT * FROM research_episode_turns WHERE run_id=? AND turn=?', (run_id,index)).fetchone()
            attempts = conn.execute('SELECT 1 FROM llm_attempts WHERE run_id=? AND stage=?',
                                   (run_id,f'research.{state["role"]}.{index}')).fetchone()
        if not row or row['state'] != 'STARTED' or row['raw_json'] is not None or attempts:
            raise ValueError('Cannot prove absence of replacement dispatch')
        harness._guard(run_id, state)
        metadata = provider_metadata(rt, run_id, migrations[-1]['failed_model'])
        state['phase'] = 'READY'
        state.pop('stop_reason', None)
        with harness.db.transaction() as conn:
            conn.execute('UPDATE runs SET metadata_json=? WHERE run_id=?', (encoded(metadata).decode(),run_id))
            conn.execute("UPDATE research_episode_turns SET state='NOT_DISPATCHED' WHERE run_id=? AND turn=?", (run_id,index))
            conn.execute('UPDATE research_episodes SET state_json=? WHERE run_id=?', (encoded(state).decode(),run_id))
        harness._save(run_id,state)
        harness.db.update_run_status(run_id,RunStatus.RUNNING)
        harness.db.emit_event(run_id,'prototype.provider_binding_reconciled',{'new_model_calls':0,'budget_reset':False})
    return harness.status(run_id)
