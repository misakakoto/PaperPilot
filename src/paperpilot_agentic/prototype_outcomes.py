"""Read-only current candidate evidence, independent of whole-task completion."""
from .prototype_workspace import digest


def validated_candidates(state):
    work = state['work']
    tasks, artifacts = work['tasks'], work['artifacts']
    contract = state.get('validation_contract')
    if not isinstance(contract, dict):
        return []  # Legacy evidence without a frozen contract is not certified here.
    verified, visiting = {}, set()

    def current(key):
        if key in verified:
            return verified[key]
        if key in visiting:
            raise ValueError('Candidate dependency cycle')
        task = tasks[key]
        if task['status'] != 'DONE':
            verified[key] = False
            return False
        visiting.add(key)
        identity = task.get('output')
        artifact = artifacts.get(identity)
        if (artifact is None or digest(artifact) != identity or artifact['name'] != key
                or artifact['kind'] != task['capability']):
            raise ValueError('Candidate artifact identity changed')
        valid = all(current(d) for d in task['depends_on'])
        if valid and artifact['inputs'] != sorted(tasks[d]['output'] for d in task['depends_on']):
            raise ValueError('Candidate dependency version changed')
        visiting.remove(key)
        verified[key] = valid
        return valid

    result = []
    for key in work.get('task_order', list(tasks)):
        task = tasks[key]
        if task['capability'] != 'validate' or not current(key):
            continue
        artifact = artifacts[task['output']]
        receipt = artifact['content']
        code_tasks = [tasks[d] for d in task['depends_on'] if tasks[d]['capability'] == 'implement']
        if len(code_tasks) != 1:
            continue
        code = code_tasks[0]
        if (receipt.get('passed') is not True or receipt.get('validated_code_id') != code['output']
                or receipt.get('validation_contract') != contract):
            continue
        result.append({'code_artifact_id': code['output'], 'code_task_id': code['id'],
            'validation_artifact_id': task['output'], 'validation_task_id': key,
            'validation_contract_identity': digest(contract),
            'metrics': {k: v for k, v in receipt.get('execution', {}).get('metrics', {}).items()
                        if k in {'parameters', 'seconds', 'peak_rss_kib', 'checks', 'device', 'seed'}},
            'scope': 'SAVED_ENGINEERING_RECEIPT_NOT_TASK_COMPLETION_OR_REQUIREMENTS_ATTESTATION',
            'scientific_validity': 'NOT_VERIFIED'})
    return result
