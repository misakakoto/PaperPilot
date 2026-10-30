"""Durable episode progression shared by research and diagnosis via composition.

Engine adapters own tools and policies. This module alone owns turn receipts,
response reuse and local-action commits; it cannot grant training permissions.
"""
import json
from contextlib import ExitStack
from .models import RunStatus, ToolCallStatus
from .reference_methods import encoded
from .research_assets import immutable
from .research_training import episode_folder
from .research_task import task_lock
from .training_state import TrainingBusy
from .durable_reader import DurableResearchModel
from .provider_diagnostics import safe_error_text
from .research_contracts import validate_state
from .runtime_profile import enabled as runtime_v2
from . import run_recovery
from .llm_runtime import LLMOutputTruncated


class ResearchStopped(RuntimeError):
    pass


class EpisodeRunner:
    def __init__(self, owner):
        self.owner = owner
        with owner.db.transaction() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS research_episodes(
                run_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
                fingerprint TEXT NOT NULL, state_json TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS research_episode_turns(
                run_id TEXT NOT NULL, turn INTEGER NOT NULL, role TEXT NOT NULL,
                request_json TEXT NOT NULL, state TEXT NOT NULL, before_event INTEGER NOT NULL,
                raw_json TEXT, result_json TEXT, PRIMARY KEY(run_id,turn))''')

    def _load(self, run_id):
        owner = self.owner
        run = owner.db.get_run(run_id)
        if run['metadata'].get('engine') not in {'research_loop_v1', 'research_diagnosis_v1', 'research_prototype_v1', 'paper_active_v1'}:
            raise ValueError('Select a research-loop episode')
        if getattr(owner, 'engine', None) and owner.engine != run['metadata'].get('engine'):
            raise ValueError('Select a task owned by this prototype engine')
        with owner.db.connect() as conn:
            row = conn.execute('SELECT state_json FROM research_episodes WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError('Research submission incomplete; inspect original record')
        return json.loads(row['state_json'])


    def _save(self, run_id, state, *, applied_turn=None, result=None):
        owner = self.owner
        state = validate_state(state)
        with owner.db.transaction() as conn:
            conn.execute('UPDATE research_episodes SET state_json=? WHERE run_id=?', (encoded(state).decode(), run_id))
            if applied_turn is not None:
                conn.execute("UPDATE research_episode_turns SET state='APPLIED',result_json=? WHERE run_id=? AND turn=?",
                             (encoded(result).decode(), run_id, applied_turn))
        owner.db.update_run_metadata(run_id, {'research_episode': owner.view(run_id, state=state)})


    def status(self, run_id):
        owner = self.owner
        return owner.runtime.status(run_id) | {'research_episode': owner.view(run_id)}


    def cancel(self, run_id):
        owner = self.owner
        owner.runtime.cancel(run_id)
        return owner.status(run_id)


    def reconcile_cancel(self, run_id):
        owner = self.owner
        """Settle abandoned local actions, never an unknown model or GPU request."""
        with task_lock(owner.runtime.workspace, run_id) as acquired:
            if not acquired:
                return
            if owner.db.get_run(run_id)['status'] != 'CANCELLED':
                return
            state = owner._load(run_id)
            state['phase'] = 'CANCELLED'
            local_tools = {'research.'+action for actions in owner.role_actions.values() for action in actions}
            local_tools.update('research.' + action for action in owner.allowed_actions(state))
            for call in owner.db.list_tool_calls(run_id):
                if call['status'] == 'RUNNING' and call['tool_name'] in local_tools:
                    owner.db.update_tool_call(call['tool_call_id'], ToolCallStatus.FAILED,
                        error='Cancelled local action; no candidate code executed by this tool')
            owner._save(run_id, state)


    def _decision(self, run_id, state):
        owner = self.owner
        index = state['turn']+1
        with owner.db.connect() as conn:
            row = conn.execute('SELECT * FROM research_episode_turns WHERE run_id=? AND turn=?', (run_id, index)).fetchone()
        if row and row['state'] != 'NOT_DISPATCHED':
            if row['state'] == 'STARTED':
                if runtime_v2(state):
                    bound = run_recovery.recover_response(run_id, state, json.loads(row['request_json']),
                                                          owner.db.list_events(run_id=run_id))
                    saved = [bound] if bound is not None else []
                else:
                    saved = [e['payload']['raw_response'] for e in owner.db.list_events(run_id=run_id)
                             if e['event_id'] > row['before_event'] and e['event_type'] == 'llm.response.saved']
                if len(saved) != 1:
                    raise ResearchStopped('Model operation is UNKNOWN or interrupted; do not resend')
                raw = saved[0]
                with owner.db.transaction() as conn:
                    conn.execute("UPDATE research_episode_turns SET state='RETURNED',raw_json=? WHERE run_id=? AND turn=?",
                                 (encoded(raw).decode(), run_id, index))
            elif row['state'] == 'RETURNED':
                raw = json.loads(row['raw_json'])
                if runtime_v2(state):
                    request = json.loads(row['request_json'])
                    if request.get('binding') != run_recovery.bind_request(run_id,index,state,request['messages']):
                        raise ResearchStopped('Saved response input versions changed; do not apply')
                    run_recovery.trace(state,'APPLY_SAVED_RESPONSE',turn=index,new_dispatches=0)
            else:
                raise ResearchStopped('Inconsistent research turn; inspect original operation')
            request = json.loads(row['request_json'])
            # The provider receipt may commit before this file or task state.
            # Restore the canonical response envelope; immutable() also refuses
            # a conflicting existing file instead of silently accepting it.
            immutable(episode_folder(owner.runtime, run_id), f'turn-{index:02d}-response.json',
                      encoded({k: raw.get(k) for k in ('text', 'model', 'finish_reason', 'request_id')}))
            if state.get('state_contract'):
                state['pending_action'] = {'turn':index,'role':state['role'],'receipt_state':'RETURNED'}
            return index, request['context_key'], raw
        compiled_output = None
        if runtime_v2(state):
            from .context_compiler import ContextCompiler, ContextCompilationBlocked
            from .models import Budget
            from .context_window import window_limit
            runtime = owner.runtime.llm_runtime
            reserve = owner._decision_reserve_class(state) if hasattr(owner,'_decision_reserve_class') else 'general'
            protected = runtime._protected_reserve(run_id, Budget.from_dict(owner.db.get_run(run_id)['budget']), reserve)
            frame = ContextCompiler().compile(owner,run_id,state,runtime.budget_snapshot(run_id),
                protected_tokens=protected,
                protected_calls=runtime._protected_call_reserve(run_id,Budget.from_dict(owner.db.get_run(run_id)["budget"]),reserve),
                window_tokens=window_limit(runtime._provider_for_run(run_id)))
            state['context_manifest'] = frame.manifest
            owner._save(run_id,state)
            owner.db.emit_event(run_id,'research.context.compiled',frame.manifest)
            if frame.manifest['status'] != 'READY':
                raise ContextCompilationBlocked(json.dumps(frame.manifest,ensure_ascii=False))
            key,messages,compiled_output = frame.context_key,frame.messages,frame.output_tokens
        else:
            key, messages = owner._context(run_id, state)
        folder = episode_folder(owner.runtime, run_id)
        history = []
        filename = f'turn-{index:02d}-request.json'
        if row:
            previous = json.loads(row['request_json'])
            history = previous.get('preflight_history', []) + [
                {'request_artifact': previous.get('request_artifact', filename),
                 'state': 'NOT_DISPATCHED'}]
            filename = f'turn-{index:02d}-request-revision-{len(history):02d}.json'
        immutable(folder, filename, encoded({'role': state['role'], 'messages': messages}))
        events = owner.db.list_events(run_id=run_id)
        request = {'context_key': key, 'messages': messages, 'request_artifact': filename,
                   'preflight_history': history}
        if runtime_v2(state):
            request['binding'] = run_recovery.bind_request(run_id,index,state,messages)
        with owner.db.transaction() as conn:
            if row:
                conn.execute("UPDATE research_episode_turns SET request_json=?,state='STARTED',before_event=? WHERE run_id=? AND turn=?",
                             (encoded(request).decode(), max((e['event_id'] for e in events), default=0), run_id, index))
            else:
                conn.execute('INSERT INTO research_episode_turns VALUES(?,?,?,?,?,?,NULL,NULL)',
                    (run_id, index, state['role'], encoded(request).decode(),
                     'STARTED', max((e['event_id'] for e in events), default=0)))
        if state.get('state_contract'):
            state['pending_action'] = {'turn':index,'role':state['role'],'receipt_state':'STARTED'}
            owner._save(run_id,state)
        reserve_class = owner._decision_reserve_class(state) if hasattr(owner, '_decision_reserve_class') else 'general'
        output_tokens = owner._decision_output_tokens(state) if hasattr(owner, '_decision_output_tokens') else 5000
        if compiled_output is not None:
            output_tokens = compiled_output
        if type(output_tokens) is not int or not 1 <= output_tokens <= 5000:
            raise ValueError('Invalid host research output allowance')
        try:
            response = DurableResearchModel(owner.runtime.llm_runtime, run_id,
                stage=f'research.{state["role"]}.{index}', reserve_class=reserve_class).complete(messages, max_tokens=output_tokens,
                    json_mode=True, thinking_enabled=False, before_attempt=lambda: owner._guard(run_id, state),
                    **({'correlation_id':request['binding']['correlation_id']} if runtime_v2(state) else {}))
        except LLMOutputTruncated as exc:
            if not runtime_v2(state):
                raise
            response = exc.response
        raw = {'text': response.text, 'model': response.model, 'finish_reason': response.finish_reason,
               'request_id': response.request_id}
        immutable(folder, f'turn-{index:02d}-response.json', encoded(raw))
        with owner.db.transaction() as conn:
            conn.execute("UPDATE research_episode_turns SET state='RETURNED',raw_json=? WHERE run_id=? AND turn=?",
                         (encoded(raw).decode(), run_id, index))
        if state.get('state_contract'):
            state['pending_action'] = {'turn':index,'role':state['role'],'receipt_state':'RETURNED'}
            owner._save(run_id,state)
        return index, key, raw


    def advance(self, run_id, *, max_turns=None):
        owner = self.owner
        with task_lock(owner.runtime.workspace, run_id) as acquired:
            if not acquired:
                return owner.status(run_id)
            state = owner._load(run_id)
            if state['phase'] in {'COMPLETED', 'PARTIAL', 'UNKNOWN', 'FAILED', 'CANCELLED'}:
                run = owner.db.get_run(run_id)
                if (state['phase'] in {'COMPLETED','PARTIAL'} and
                        (run['metadata'].get('final_answer') is None or run['status'] != 'COMPLETED')):
                    # Terminal state and local publication are separate commits.
                    # Recover only the deterministic publication, never another
                    # model/tool decision, and retain the frozen identity guard.
                    owner._guard(run_id,state)
                    if runtime_v2(state):
                        run_recovery.trace(state,'PUBLISH_SAVED_REPORT',new_dispatches=0)
                        owner._save(run_id,state)
                    owner._publish(run_id,state)
                    owner.db.emit_event(run_id,'research.publication_recovered',{
                        'new_model_requests':0,'new_training_starts':0})
                return owner.status(run_id)
            used = 0
            action_call = None
            activity = ExitStack()
            try:
                if hasattr(owner, '_active_time'):
                    activity.enter_context(owner._active_time(run_id))
                owner._guard(run_id, state)
                owner.db.update_run_status(run_id, RunStatus.RUNNING)
                while state['turn'] < state['limits']['max_turns']:
                    action_call = None
                    if max_turns is not None and used >= max_turns:
                        break
                    owner._guard(run_id, state)
                    if state['phase'] in {'WAITING_APPROVAL', 'TRAINING', 'PAUSED', 'WAITING_ENVIRONMENT'}:
                        if not owner._training(run_id, state):
                            break
                    index, key, raw = owner._decision(run_id, state)
                    role = state['role']
                    action_call = None
                    try:
                        if raw.get('finish_reason') not in (None, 'stop'):
                            raise ValueError('Role output did not finish normally')
                        action = json.loads(raw['text'])
                        if (isinstance(action, dict) and action.get('action') in owner.allowed_actions(state)
                                and isinstance(action.get('arguments'), dict)
                                and getattr(owner, 'track_action_calls', True)):
                            action_call, reused = owner.db.create_step_tool_call(run_id=run_id,
                                tool_name='research.'+action['action'], args=action['arguments'],
                                risk_level='LOW', idempotency_key=f'research-action:{run_id}:{index}', max_attempts=1)
                            owner.db.update_tool_call(action_call, ToolCallStatus.RUNNING, increment_attempt=not reused)
                        observation = owner._apply(run_id, state, action, index)
                    except (ValueError, KeyError, TypeError, StopIteration) as exc:
                        observation = {'status': 'REJECTED', 'error': str(exc), 'tool_not_executed': True}
                        if runtime_v2(state) and raw.get('finish_reason') in {'length','max_tokens','max_output_tokens'}:
                            task_id = state.get('context_manifest',{}).get('task_id',state['role'])
                            counts = state.setdefault('output_truncations',{})
                            counts[task_id] = counts.get(task_id,0)+1
                            observation.update(code='OUTPUT_TRUNCATED',error='Response truncated; no action admitted; compact the patch or use the next compiled output allowance')
                        if role == 'engineer' and state['candidates']:
                            candidate = state['candidates'][-1]
                            candidate['repairs'] += 1
                            if candidate['repairs'] > state['limits']['max_patch_repairs']:
                                candidate['status'] = 'PATCH_REJECTED'
                                state['role'] = 'lead'
                        state['format_failures'] = state.get('format_failures', 0)+1
                    if action_call:
                        owner.db.update_tool_call(action_call,
                            ToolCallStatus.FAILED if observation['status'] == 'REJECTED' else ToolCallStatus.COMPLETED,
                            result={'ok': observation['status'] != 'REJECTED', 'output': observation})
                    state['contexts'].setdefault(key, []).extend([
                        {'role': 'assistant', 'content': raw['text']},
                        {'role': 'user', 'content': encoded({'tool_result': observation}).decode()}])
                    state['turn'], state['last_observation'] = index, observation
                    if state.get('state_contract'):
                        state['pending_action'] = None
                    owner._after_action(run_id, state, role, observation)
                    owner._save(run_id, state, applied_turn=index, result=observation)
                    owner.db.emit_event(run_id, 'research.action', {'role': role, 'turn': index,
                        'action': observation.get('action'), 'status': observation.get('status'),
                        'reason': observation.get('reason'), 'error': observation.get('error')})
                    used += 1
                    if state['phase'] in {'COMPLETED', 'PARTIAL', 'WAITING_APPROVAL', 'WAITING_ENVIRONMENT'}:
                        break
                if state['turn'] >= state['limits']['max_turns'] and state['phase'] == 'READY':
                    owner._finish(run_id, state, 'Decision turn limit reached', complete=False)
                owner._save(run_id, state)
                if state['phase'] in {'COMPLETED', 'PARTIAL'}:
                    owner._publish(run_id, state)
            except InterruptedError:
                state['phase'] = 'CANCELLED'
                if runtime_v2(state):
                    run_recovery.trace(state,'CANCELLED',new_dispatches=0)
                if action_call and owner.db.get_tool_call(action_call)['status'] == 'RUNNING':
                    owner.db.update_tool_call(action_call, ToolCallStatus.FAILED,
                                             error='Cancelled before local action commit')
                owner._save(run_id, state)
                owner.db.update_run_status(run_id, RunStatus.CANCELLED)
            except (ResearchStopped, TrainingBusy) as exc:
                state['phase'], state['stop_reason'] = 'UNKNOWN', str(exc)
                owner._save(run_id, state)
                owner.db.emit_event(run_id, 'research.blocked', {'reason': str(exc)})
            except Exception as exc:
                snapshot = owner.runtime.llm_runtime.budget_snapshot(run_id)
                state['phase'] = 'UNKNOWN' if snapshot['unknown_results'] else 'FAILED'
                if runtime_v2(state) and snapshot['unknown_results']:
                    run_recovery.trace(state,'WAIT_UNKNOWN',reason='UNRESOLVED_EXTERNAL_REQUEST',new_dispatches=0)
                state['stop_reason'] = safe_error_text(exc)
                owner._save(run_id, state)
                owner.db.update_run_status(run_id, RunStatus.FAILED, error=state['stop_reason'])
            finally:
                activity.close()
            return owner.status(run_id)


    def _finish(self, run_id, state, reason, *, complete):
        state['phase'], state['completed'], state['stop_reason'] = ('COMPLETED' if complete else 'PARTIAL'), complete, reason

