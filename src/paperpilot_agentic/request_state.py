"""Request-state reducer used by pressure-triggered text Reader memory.

Exact attribution and atomic edits are checked here. Semantic interpretation of
user intent remains unverified; this state never grants tool authorization or
establishes scientific facts. Callers own durable storage and model budgets.
"""
from copy import deepcopy
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _record_id(source_id,index,task_id,key):
    return _digest([source_id,index,task_id,key])[:24]


def empty_request_state():
    return {'version': 1, 'revision': 'initial', 'active_task_id': None,
            'tasks': {}, 'sources': {}, 'events': []}


def _text(value, field='text'):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Expected nonempty text: {field}')
    return value


def apply_request_delta(state, source, delta):
    """Apply one owned user request atomically, or raise ValueError.

    source is {id,text}; delta is {expected_revision,operations}. Each operation
    has kind, task_id and a verbatim quote from THIS source. Replacing/retracting
    a condition requires its current record ID. Completed tool work is outside
    this interface; the model cannot mark experiments completed here.
    """
    if not isinstance(source, dict) or not isinstance(delta, dict):
        raise ValueError('Expected source and delta objects')
    sid, source_text = _text(source.get('id'), 'source.id'), _text(source.get('text'), 'source.text')
    signature = _digest({'source': source, 'delta': delta})
    if sid in state['sources']:
        event = next(e for e in state['events'] if e['source_id'] == sid)
        if event['signature'] != signature:
            raise ValueError('Source request already applied with different contents')
        return deepcopy(state)
    if delta.get('expected_revision') != state['revision']:
        raise ValueError('Stale request-state revision')
    operations = delta.get('operations')
    if not isinstance(operations, list) or len(operations) > 64:
        raise ValueError('Expected at most 64 request operations')
    updated = deepcopy(state)
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ValueError('Expected operation object')
        # Resolve content-based preconditions on a private copy. Keep the
        # original model proposal in the event/signature for exact replay.
        op=deepcopy(op)
        kind, task_id = op.get('kind'), _text(op.get('task_id'), f'operations[{index}].task_id')
        quote = _text(op.get('quote'), f'operations[{index}].quote')
        if quote not in source_text:
            raise ValueError('Quote is not in current user request')
        tasks = updated['tasks']
        if kind == 'create_task':
            if task_id in tasks:
                raise ValueError('Task already exists')
            tasks[task_id] = {'label': _text(op.get('label'), f'operations[{index}].label'),
                              'label_source_id':sid, 'label_quote':quote, 'label_history':[],
                              'conditions': {}, 'todos': {}, 'archive': []}
            continue
        if task_id not in tasks:
            raise ValueError('Unknown task')
        if kind == 'activate_task':
            updated['active_task_id'] = task_id
            continue
        task = tasks[task_id]
        if kind == 'rename_task':
            if op.get('previous_label') != task['label']:
                raise ValueError('Rename must name the current task label')
            label = _text(op.get('label'), f'operations[{index}].label')
            task.setdefault('label_history', []).append({
                'label':task['label'], 'source_id':task.get('label_source_id'),
                'quote':task.get('label_quote'), 'changed_by':sid})
            task.update(label=label, label_source_id=sid, label_quote=quote)
            continue
        if kind not in {'set_condition', 'retract_condition', 'add_todo', 'cancel_todo',
                        'pause_todo', 'resume_todo'}:
            raise ValueError('Unknown operation; experiment completion is not a request edit')
        key = _text(op.get('key'), f'operations[{index}].key')
        slot = task['conditions'] if 'condition' in kind else task['todos']
        old = slot.get(key)
        if 'expect' in op:
            if 'replaces' in op:
                raise ValueError('Use either expect or replaces, not both')
            expected=op['expect']
            if expected is None:
                if old is not None:
                    raise ValueError('Expected record absence but a current record exists')
            else:
                if (not isinstance(expected,dict) or set(expected)!={'text','status','applies_to'} or
                        not old or expected!={'text':old['text'],'status':old['status'],
                                              'applies_to':old.get('applies_to')}):
                    raise ValueError('Current record does not match the expected prior value/status/scope')
            op['replaces']=old['id'] if old else None
        scope = None
        if kind == 'set_condition':
            scope=op.get('applies_to',old.get('applies_to') if old else None)
            if scope is not None and (not isinstance(scope,str) or scope not in task['todos']):
                raise ValueError('Condition activity scope must name an unfinished action in the same task')
        if kind in {'pause_todo', 'resume_todo'}:
            expected = 'pending' if kind == 'pause_todo' else 'paused'
            if not old or old['id'] != op.get('replaces') or old['status'] != expected:
                raise ValueError('Pause/resume must reference the current unfinished action in the expected state')
            status = 'paused' if kind == 'pause_todo' else 'pending'
            task['archive'].append({**old,'status':'transitioned','next_status':status,'changed_by':sid})
            slot[key] = {**old,'id':_record_id(sid,index,task_id,key), 'status':status,
                         'status_source_id':sid,'status_quote':quote}
            continue
        if kind == 'set_condition':
            if op.get('replaces') != (old['id'] if old else None):
                raise ValueError('Replacement must name the current condition')
        elif kind == 'add_todo':
            if old:
                raise ValueError('Pending action already exists')
        elif not old or op.get('replaces') != old['id']:
            raise ValueError('Retraction must name the current record')
        if old:
            task['archive'].append({**old, 'status':'superseded' if kind == 'set_condition' else 'cancelled',
                                    'changed_by': sid, 'change_quote': quote})
        if kind in {'retract_condition', 'cancel_todo'}:
            del slot[key]
            if kind=='cancel_todo':
                # Only explicitly bound parameters follow activity cancellation.
                # Task-wide restrictions must not disappear with a pending action.
                affected=[k for k,r in task['conditions'].items() if r.get('applies_to')==key]
                for condition_key in affected:
                    condition=task['conditions'].pop(condition_key)
                    task['archive'].append({**condition,'status':'activity_cancelled',
                        'changed_by':sid,'change_quote':quote,'cancelled_activity':key})
        else:
            slot[key] = {'id':_record_id(sid,index,task_id,key),
                         'text':_text(op.get('text'), f'operations[{index}].text'), 'source_id':sid, 'quote':quote,
                         'status':'active' if kind == 'set_condition' else 'pending'}
            if kind=='set_condition':
                slot[key]['applies_to']=scope
    updated['sources'][sid] = source_text
    updated['events'].append({'source_id':sid, 'signature':signature,
                              'base_revision':state['revision'], 'operations':deepcopy(operations)})
    updated['revision'] = _digest([state['revision'], signature])
    return updated


def request_compiler_view(state):
    """Project current records of ALL tasks, never recursive event history.

    Current records retain exact text, status, scope and source quotes. Previous
    labels are identity aliases, not effective constraints. This is a lossy view
    of history, NOT a lossy summary of current records. Callers must enforce their
    message budget and reload history when old/cancelled values are requested.
    Many genuinely live tasks can still exceed that budget; none is auto-dropped.
    """
    tasks = {}
    for tid, task in state['tasks'].items():
        aliases = {}
        for old in task.get('label_history', []):
            # Keep the newest source for each distinct former name. Repeated
            # renames must not expand the prompt with identical aliases.
            aliases[old['label']] = old['source_id']
        aliases.pop(task['label'], None)
        tasks[tid] = {
            'label': task['label'], 'label_source_id': task.get('label_source_id'),
            'label_quote': task.get('label_quote'),
            'former_labels': [{'label': label, 'source_id': sid} for label, sid in aliases.items()],
            'conditions': deepcopy(task['conditions']), 'todos': deepcopy(task['todos']),
            'history_available': bool(task.get('archive') or task.get('label_history')),
        }
    return {'version': state['version'], 'revision': state['revision'],
            'active_task_id': state['active_task_id'], 'tasks': tasks,
            'history_policy': 'current_records_only; originals_and_past_values_require_reload'}


def request_work_view(state, *, max_bytes=12000):
    """Bounded active-task view; full sources and past values remain in state.

    Cold paused records may be offloaded with an explicit reload marker. Active
    requirements are never truncated. Oversized active work includes a batching
    plan, not a falsely complete first page.
    """
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError('Expected positive byte budget')
    tid = state['active_task_id']
    task = state['tasks'].get(tid)
    if state['tasks'] and task is None:
        return {'status':'NEEDS_TASK_SELECTION','required_bytes':0,
                'revision':state['revision'],'view':None}
    def project(rows):
        return {k:{field:r[field] for field in ('id','text','source_id','status','status_source_id','applies_to') if field in r}
                for k,r in rows.items()}
    active_conditions,paused_conditions={},{}
    for key,row in (task['conditions'].items() if task else []):
        scope=row.get('applies_to')
        if scope is None:
            active_conditions[key]=row
        else:
            activity=task['todos'].get(scope)
            if activity is None or activity['status'] not in {'pending','paused'}:
                raise ValueError('Condition has an unresolved activity scope')
            (paused_conditions if activity['status']=='paused' else active_conditions)[key]=row
    view = {'active_task_id':tid, 'label':task['label'] if task else None,
            'label_source_id':task.get('label_source_id') if task else None,
            'conditions':project(active_conditions),
            'paused_conditions':project(paused_conditions),
            'pending_actions':project({k:v for k,v in task['todos'].items() if v['status']=='pending'}) if task else {},
            'paused_actions':project({k:v for k,v in task['todos'].items() if v['status']=='paused'}) if task else {},
            'assurance':'derived_user_requirements_not_semantically_verified_or_authorization'}
    size = len(json.dumps(view, ensure_ascii=False).encode())
    if size > max_bytes:
        from .request_worksets import plan_worksets
        plan = plan_worksets(view, state['revision'], max_bytes)
        if plan['status'] == 'READY':
            return {**plan, 'revision': state['revision']}
        return {'status': 'OVERFLOW', 'required_bytes': size,
                'revision': state['revision'], 'view': None, 'work_plan': plan}
    return {'status':'READY' if size <= max_bytes else 'OVERFLOW',
            'required_bytes':size, 'revision':state['revision'],
            'view':view if size <= max_bytes else None}


def apply_request_batch(state, sources, proposal):
    """Apply ordered frames covering every supplied source, including no-ops.

    Models need only emit {base_revision, frames:[{source_id,operations}]}.
    Per-frame revisions are calculated locally, not guessed by the model.
    Coverage is structural, not proof that operations preserve every condition.
    No caller-owned state is mutated on any failure.
    """
    if not isinstance(sources,list) or not sources or len(sources)>32:
        raise ValueError('Expected one to 32 source requests')
    if any(not isinstance(s,dict) for s in sources) or not isinstance(proposal,dict):
        raise ValueError('Expected source and proposal objects')
    ids=[_text(s.get('id'),'source.id') for s in sources]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate source request')
    frames=proposal.get('frames')
    if not isinstance(frames,list) or any(not isinstance(f,dict) for f in frames):
        raise ValueError('Expected request frames')
    if [f.get('source_id') for f in frames]!=ids:
        raise ValueError('Frames must cover every source exactly once in supplied order')
    seen=[sid in state['sources'] for sid in ids]
    if any(seen) and not all(seen):
        raise ValueError('Partial batch replay requires explicit reconciliation')
    old_events={e['source_id']:e for e in state['events']}
    if all(seen):
        if proposal.get('base_revision')!=old_events[ids[0]]['base_revision']:
            raise ValueError('Conflicting batch replay revision')
    elif proposal.get('base_revision')!=state['revision']:
        raise ValueError('Stale batch revision')
    current=state
    references={}
    for source,frame in zip(sources,frames):
        operations=deepcopy(frame.get('operations'))
        if not isinstance(operations,list) or any(not isinstance(op,dict) for op in operations):
            raise ValueError('Expected operation objects')
        for op in operations:
            ref=op.get('replaces')
            if isinstance(ref,dict):
                if (set(ref)!={'frame','operation'} or not isinstance(ref['frame'],str) or
                        type(ref['operation']) is not int or (ref['frame'],ref['operation']) not in references):
                    raise ValueError('Replacement reference must name an earlier frame record')
                op['replaces']=references[(ref['frame'],ref['operation'])]
        base=old_events[source['id']]['base_revision'] if all(seen) else current['revision']
        current=apply_request_delta(current,source,{'expected_revision':base,
                                                  'operations':operations})
        for index,op in enumerate(operations):
            if op['kind'] in {'set_condition','add_todo','pause_todo','resume_todo'}:
                references[(source['id'],index)]=_record_id(source['id'],index,op['task_id'],op['key'])
    return current
