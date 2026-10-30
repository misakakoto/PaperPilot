"""Lossless, deterministic working-set planning for derived user requirements.

This is not a scheduler or an authorization decision. Every page repeats global
constraints and keeps each activity with ALL of its scoped conditions. Full
records remain in the revisioned request store; no model summarization is used.
"""
from copy import deepcopy
import json


def byte_size(value):
    return len(json.dumps(value, ensure_ascii=False).encode('utf-8'))


def plan_worksets(view, revision, max_bytes):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('Expected positive byte budget')
    original = deepcopy(view)
    if byte_size(original) <= max_bytes:
        return {'status': 'READY', 'view': original, 'required_bytes': byte_size(original)}

    # Paused activity settings are cold data, not effective requirements. Keep
    # task-wide constraints even when every activity is paused.
    compact = deepcopy(original)
    paused = compact.pop('paused_actions', {})
    paused_conditions = compact.pop('paused_conditions', {})
    compact['paused_actions'] = {}
    compact['paused_conditions'] = {}
    compact['workset'] = {
        'revision': revision, 'scope': 'all_pending_activities',
        'deferred_paused_actions': len(paused),
        'deferred_paused_conditions': len(paused_conditions),
        'reload_policy': 'Paused records remain in the request store. Resume or inspect '
                         'them from the ledger before use; never infer their content.',
    }
    if (paused or paused_conditions) and byte_size(compact) <= max_bytes:
        return {'status': 'READY', 'view': compact, 'required_bytes': byte_size(compact)}

    actions = compact['pending_actions']
    conditions = compact['conditions']
    global_conditions = {k: v for k, v in conditions.items() if v.get('applies_to') is None}
    scoped = {key: {} for key in actions}
    for key, condition in conditions.items():
        target = condition.get('applies_to')
        if target is not None:
            if target not in scoped:
                raise ValueError('Unresolved activity in working set')
            scoped[target][key] = condition

    def page(keys):
        result = deepcopy({k: v for k, v in compact.items()
                           if k not in {'pending_actions', 'conditions'}})
        result['pending_actions'] = {key: deepcopy(actions[key]) for key in keys}
        result['conditions'] = deepcopy(global_conditions)
        for key in keys:
            result['conditions'].update(deepcopy(scoped[key]))
        result['workset'].update(scope='partial_pending_activities',
            total_pending_actions=len(actions), selected_action_keys=list(keys),
            instruction='This page is NOT the complete task. Do not claim overall completion. '
                        'Planning pages neither executes nor authorizes an activity.')
        return result

    pages, selected = [], []
    if not actions or byte_size(page([])) > max_bytes:
        return {'status': 'BLOCKED', 'reason': 'GLOBAL_REQUIREMENTS_TOO_LARGE', 'pages': []}
    for key in actions:
        trial = page(selected + [key])
        if byte_size(trial) <= max_bytes:
            selected.append(key)
            continue
        if not selected or byte_size(page([key])) > max_bytes:
            return {'status': 'BLOCKED', 'reason': 'ACTIVITY_TOO_LARGE',
                    'activity_key': key, 'pages': []}
        pages.append(page(selected))
        selected = [key]
    if selected:
        pages.append(page(selected))
    return {'status': 'NEEDS_WORK_BATCHES', 'revision': revision, 'pages': pages,
            'total_pending_actions': len(actions),
            'assurance': 'Deterministic projection only; pages are not executed or completed.'}
