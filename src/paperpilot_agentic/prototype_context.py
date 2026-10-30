"""Versioned context projection; stored artifacts and task references stay intact."""
import copy

POLICY = 'artifact-table-v1'
SERIAL_POLICY = 'serial-task-v1'


def task_selection(state):
    policy = state.get('task_context_policy')
    if policy not in (None, SERIAL_POLICY):
        raise ValueError('Unknown prototype task context policy')
    tasks = state['work']['tasks']
    order = state['work'].get('task_order') if policy == SERIAL_POLICY else list(tasks)
    if (not isinstance(order, list) or len(order) != len(tasks)
            or any(not isinstance(k, str) for k in order) or set(order) != set(tasks)):
        raise ValueError('Invalid prototype task context policy order')
    ready = [key for key in order if (task := tasks[key])['status'] != 'DONE'
             and all(tasks[d]['status'] == 'DONE' for d in task['depends_on'])]
    return ready, ready[:1] if policy == SERIAL_POLICY else ready


def artifact_table(payload):
    """Share context views by exact artifact ID, including validation briefs.

    Different IDs are never merged by text similarity. A validation view remains
    explicitly linked to its raw receipt; this table is not a new artifact store.
    """
    result = copy.deepcopy(payload)
    table = {}

    def reference(identity, artifact):
        if identity in table and table[identity] != artifact:
            raise ValueError('Conflicting views for the same context artifact')
        table[identity] = artifact
        return {'artifact_ref': identity}

    for key, value in result['ready_contexts'].items():
        if value.get('goal') != result['goal'] or value['task'] != result['tasks'][key]:
            raise ValueError('Task context definition disagrees with workspace')
        value.pop('goal')
        value['task'] = {'id': key, 'capability': result['tasks'][key]['capability'], 'task_ref': key}
        value['dependencies'] = {i: reference(i, a) for i, a in value['dependencies'].items()}
    for value in result['repair_contexts'].values():
        value['previous_artifact'] = reference(value['previous_artifact_id'], value['previous_artifact'])
    feedback = result.get('validation_feedback')
    if feedback:
        matches = [i for i, a in table.items() if a['kind'] == 'validate' and a['content'] == feedback['result']]
        if matches:
            feedback['result'] = {'artifact_ref': sorted(matches)[0], 'field': 'content'}
    result.update(context_policy=POLICY, context_artifacts=table)
    return result
