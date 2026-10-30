"""A fixed coding loop may repair its design without editing the task graph."""
from .prototype_workspace import PLAN_POLICY, evidence_requirements, plan, task_rows


def can_repair(state):
    work = state['work']
    if (state.get('mode') != 'simple_loop' or work['repairs'] >= 3
            or state.get('plan_requires_work')):
        return False
    tasks, artifacts = work['tasks'], work['artifacts']
    code = tasks.get('code', {})
    if code.get('status') != 'DONE':
        return False
    for key in ('check', 'review'):
        task = tasks.get(key, {})
        if task.get('status') != 'DONE':
            continue
        artifact = artifacts[task['output']]
        content = artifact['content']
        if code['output'] not in artifact['inputs']:
            continue
        if key == 'check' and (content.get('passed') is False
                and content.get('validated_code_id') == code['output']
                # Match the adaptive loop: an unknown result or timeout alone
                # cannot attribute failure to the current implementation.
                and content.get('failure_kind') in {'IMPLEMENTATION', 'DESIGN_CONSTRAINT'}
                and (state.get('validation_contract') is None
                     or content.get('validation_contract') == state['validation_contract'])):
            return True
        if key == 'review' and (content.get('issues') or content.get('implementation_consistent') is False):
            return True
    return False


def repair(state, arguments, reason, index):
    if (not can_repair(state) or not isinstance(arguments, dict)
            or set(arguments) != {'target'} or arguments['target'] not in {'design', 'implement'}):
        raise ValueError('Repair requires a current failure and target design or implement')
    work = state['work']
    target = 'code' if arguments['target'] == 'implement' else 'design'
    rows = task_rows(work)
    next(t for t in rows if t['id'] == target)['goal'] += f' Repair {index}: {reason}'
    plan(work, rows, enforce_capabilities=True,
         require_validation_coverage=state.get('plan_policy') == PLAN_POLICY,
         required_evidence=evidence_requirements(state))
    state['plan_requires_work'] = True
