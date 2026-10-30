"""Make registered direct dependencies explicit without inventing new work."""
import copy

from .prototype_workspace import plan, new_workspace, digest

POLICY = 'transitive-input-elaboration-v1'


def check_policy(state):
    policy = state.get('plan_edge_policy')
    if policy not in (None, POLICY):
        raise ValueError('Unknown prototype plan edge policy')


def elaborate(state, rows):
    check_policy(state)
    policy = state.get('plan_edge_policy')
    if policy is None:
        return rows, None
    # Validate shapes and cycles before traversing model-proposed identifiers.
    probe = new_workspace(state['work']['goal'])
    plan(probe, rows)
    tasks = {t['id']: t for t in rows}
    proposed = copy.deepcopy(rows)
    added = []
    for task in proposed:
        visited, queue = set(), list(task['depends_on'])
        while queue:
            key = queue.pop()
            if key not in visited:
                visited.add(key)
                queue.extend(tasks[key]['depends_on'])
        required = {'design': {'read'}, 'implement': {'design'},
                    'validate': {'implement'}, 'review': {'read', 'design', 'implement', 'validate'},
                    'read_literature': {'search_literature'}}.get(task['capability'], set())
        if task['capability'] == 'design' and state.get('literature_analysis_policy'):
            required = required | {'analyze_literature', 'review_literature'}
        if task['capability'] == 'implement' and state.get('implementation_source_ids'):
            required = required | {'read'}
        for prior in rows:
            key = prior['id']
            if key in visited and prior['capability'] in required and key not in task['depends_on']:
                task['depends_on'].append(key)
                added.append({'task_id': task['id'], 'input_task_id': key,
                              'reason': 'Existing transitive prerequisite made explicit'})
    return proposed, {'policy': POLICY, 'proposal_sha256': digest(rows),
                      'compiled_sha256': digest(proposed), 'added_edges': added}
