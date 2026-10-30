"""Targeted adaptive repairs bound to current failed candidate receipts."""
from .prototype_workspace import PLAN_POLICY, evidence_requirements, plan, task_rows


POLICY = 'receipt-bound-repair-v1'


def repair_options(state):
    policy = state.get('repair_policy')
    if policy not in (None, POLICY):
        raise ValueError('Unknown prototype repair policy')
    work = state['work']
    if (policy is None or state.get('mode') != 'adaptive' or work['repairs'] >= 3
            or state.get('plan_requires_work')):
        return {}
    tasks, artifacts = work['tasks'], work['artifacts']
    options = {}
    for key, task in tasks.items():
        if task['status'] != 'DONE' or task['capability'] not in {'validate', 'review'}:
            continue
        receipt = artifacts[task['output']]
        content = receipt['content']
        if task['capability'] == 'validate':
            if (content.get('passed') is not False
                    or content.get('failure_kind') not in {'IMPLEMENTATION', 'DESIGN_CONSTRAINT'}
                    or state.get('validation_contract') is not None
                    and content.get('validation_contract') != state['validation_contract']):
                continue
        elif not (content.get('issues') or content.get('implementation_consistent') is False):
            continue
        for dep in task['depends_on']:
            code = tasks[dep]
            if (code['capability'] != 'implement' or code['status'] != 'DONE'
                    or code['output'] not in receipt['inputs']):
                continue
            if task['capability'] == 'validate' and content.get('validated_code_id') != code['output']:
                continue
            targets = [dep] + [d for d in code['depends_on'] if tasks[d]['capability'] == 'design']
            for target in targets:
                options.setdefault(target, []).append({
                    'failure_task_id': key, 'receipt_artifact_id': task['output'],
                    'code_artifact_id': code['output'],
                    'failure_kind': content.get('failure_kind', 'MODEL_REVIEW'),
                })
    return options


def repair(state, arguments, reason, index):
    options = repair_options(state)
    if (not isinstance(arguments, dict) or set(arguments) != {'target_task_id'}
            or not isinstance(arguments['target_task_id'], str)
            or arguments['target_task_id'] not in options):
        raise ValueError('Repair requires an eligible target_task_id from repair_options')
    target = arguments['target_task_id']
    work = state['work']
    rows = task_rows(work)
    next(t for t in rows if t['id'] == target)['goal'] += f' Repair {index}: {reason}'
    plan(work, rows, enforce_capabilities=True,
         required_sources=state.get('implementation_source_ids', ()),
         require_validation_coverage=state.get('plan_policy') == PLAN_POLICY,
         required_evidence=evidence_requirements(state))
    state.setdefault('repair_history', []).append({
        'turn': index, 'target_task_id': target, 'reason': reason,
        'failure_bindings': options[target], 'workspace_revision': work['revision'],
    })
    state['plan_requires_work'] = True
