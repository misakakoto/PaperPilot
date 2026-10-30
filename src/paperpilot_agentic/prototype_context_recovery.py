"""Host recovery for context construction that provably preceded any request."""
from . import prototype_prompt as prompt, prototype_quote_handles as quotes, prototype_selected_text as selected
from .llm_runtime import PersistentLLMRuntime
from .models import Budget, RunStatus
from .prototype_context import task_selection
from .prototype_workspace import digest
from .research_task import task_lock
from .runtime_profile import enabled as runtime_v2

CONTEXT_STOPS = {
    'ValueError: Required context exceeds window; split the task',
    'ValueError: Task context too large; split plan',
}


def recover(harness, rid):
    with task_lock(harness.runtime.workspace, rid) as acquired:
        if not acquired:
            raise ValueError('Prototype busy')
        state = harness._load(rid)
        compiled_stop = (str(state.get('stop_reason', '')).startswith('ContextCompilationBlocked:')
                         and state.get('context_manifest', {}).get('status') == 'BLOCKED'
                         and state['context_manifest'].get('reason') == 'REQUIRED_CONTEXT_EXCEEDS_BUDGET'
                         and runtime_v2(state))
        if state['phase'] != 'FAILED' or (state.get('stop_reason') not in CONTEXT_STOPS and not compiled_stop):
            raise ValueError('Only a known pre-request context construction stop can recover')
        harness._guard(rid, state)
        _, active = task_selection(state)
        if len(active) != 1 or (state['work']['tasks'][active[0]]['capability'] != 'analyze_literature'
                               and not (compiled_stop and state['work']['tasks'][active[0]]['capability'] == 'read_literature')):
            raise ValueError('Shared reading recovery requires a ready literature analysis task')
        runtime = harness.runtime.llm_runtime
        snapshot = runtime.budget_snapshot(rid)
        if snapshot['unknown_results']:
            raise ValueError('An active or unknown model request prevents context recovery')
        index = state['turn'] + 1
        stage = f'research.{state["role"]}.{index}'
        with harness.db.connect() as conn:
            turn = conn.execute('SELECT 1 FROM research_episode_turns WHERE run_id=? AND turn=?', (rid, index)).fetchone()
            attempt = conn.execute('SELECT 1 FROM llm_attempts WHERE run_id=? AND stage=?', (rid, stage)).fetchone()
            call = conn.execute('SELECT 1 FROM llm_calls WHERE run_id=? AND stage=?', (rid, stage)).fetchone()
        events = harness.db.list_events(run_id=rid)
        if turn or attempt or call or any(e['event_type'] in {'llm.attempt.started', 'llm.response.saved'}
                and e['payload'].get('stage') == stage for e in events):
            raise ValueError('Cannot prove context construction preceded dispatch; inspect original request')
        original_work = digest(state['work'])
        previous = {key: state.get(key) for key in ('analysis_read_policy', 'quote_handle_policy',
                                                   'analysis_workset_policy', 'research_source_policy')}
        budget = Budget.from_dict(harness.db.get_run(rid)['budget'])
        protected = runtime._protected_reserve(rid, budget, harness._decision_reserve_class(state))
        old_manifest = state.get('context_manifest')
        if compiled_stop:
            from .context_compiler import ContextCompiler
            from .context_window import window_limit
            frame = ContextCompiler().compile(harness, rid, state, snapshot, protected_tokens=protected,
                protected_calls=runtime._protected_call_reserve(rid, budget, harness._decision_reserve_class(state)),
                window_tokens=window_limit(runtime._provider_for_run(rid)))
            if frame.manifest['status'] != 'READY':
                raise ValueError('Required context still cannot fit the original remaining budget')
            state['context_manifest'] = frame.manifest
            upper, output = frame.manifest['input_upper_bound'], frame.output_tokens
        else:
            state.update(analysis_read_policy=selected.POLICY, quote_handle_policy=quotes.POLICY,
                         analysis_workset_policy=quotes.WORKSET_POLICY, research_source_policy=prompt.DEFERRED_SOURCE_POLICY)
            _, messages = harness._context(rid, state)
            upper = PersistentLLMRuntime._prompt_token_upper_bound(runtime._with_budget(messages, snapshot))
            output = harness._decision_output_tokens(state)
        if snapshot['remaining_tokens'] is not None and upper + output > snapshot['remaining_tokens'] - protected:
            raise ValueError('Shared reading context still cannot fit the original remaining budget')
        assert digest(state['work']) == original_work
        state.setdefault('context_recoveries', []).append({'turn': index, 'reason': state['stop_reason'],
            'previous_policies': previous, 'analysis_workset_policy': quotes.WORKSET_POLICY,
            'research_source_policy': prompt.DEFERRED_SOURCE_POLICY, 'work_sha256': original_work,
            'prompt_upper_bound': upper, 'max_output_tokens': output, 'protected_tokens': protected,
            'remaining_tokens': snapshot['remaining_tokens'], 'new_model_requests': 0, 'budget_reset': False})
        if compiled_stop:
            state['context_recoveries'][-1]['previous_context_manifest'] = old_manifest
            from .run_recovery import trace
            trace(state, 'RECOMPILE_KNOWN_UNDISPATCHED_CONTEXT', turn=index, new_dispatches=0, budget_reset=False)
        state.pop('stop_reason')
        state['phase'] = 'READY'
        harness._save(rid, state)
        harness.db.update_run_status(rid, RunStatus.RUNNING)
        harness.db.emit_event(rid, 'research.context_recovered', {'turn': index, 'new_model_requests': 0,
                                                               'budget_reset': False, 'work_sha256': original_work})
    return harness.status(rid)
