"""Versioned prototype task graph. No candidate code is executed here.

Artifacts are immutable; task completion binds exact input versions. Replacing an
artifact invalidates transitive consumers rather than deleting historical work.
"""
import ast
import copy
import hashlib
import json


CAPABILITIES = {'read', 'design', 'implement', 'validate', 'review', 'search_literature', 'read_literature', 'analyze_literature', 'review_literature'}
PLAN_POLICY = 'validation-coverage-v1'
SOURCE_POLICY = 'selected-evidence-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def new_workspace(goal):
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError('A prototype goal is required')
    return {'version': 'prototype-workspace-v1', 'goal': goal, 'revision': 0,
            'tasks': {}, 'task_order': [], 'artifacts': {}, 'heads': {}, 'repairs': 0}


def task_rows(work):
    return copy.deepcopy([{field: work['tasks'][key][field]
             for field in ('id', 'goal', 'capability', 'depends_on', 'acceptance')}
            for key in work.get('task_order', list(work['tasks']))])


def require_source_context(work, task, required_sources, *, tasks=None, allow_pending=False):
    if not required_sources:
        return
    tasks = work['tasks'] if tasks is None else tasks
    reads = [tasks[d] for d in set(task['depends_on']) if tasks[d]['capability'] == 'read']
    resolved = set()
    for read in reads:
        if read['status'] == 'DONE':
            content = work['artifacts'][read['output']]['content']
            if isinstance(content, dict):
                resolved.add(content.get('evidence_id'))
    missing = set(required_sources) - resolved
    pending = sum(t['status'] != 'DONE' for t in reads)
    if missing and (not allow_pending or pending < len(missing)):
        raise ValueError(f'Task {task["id"]} must directly depend on read tasks for code source IDs: {sorted(missing)}')


def evidence_requirements(state):
    policy = state.get('source_policy')
    if policy is None:
        return ()
    if policy != SOURCE_POLICY:
        raise ValueError('Unknown prototype source policy')
    ids = state.get('required_evidence_ids')
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids) or set(ids) != set(state['sources'])):
        raise ValueError('Required selected evidence IDs changed')
    return ids


def require_evidence_coverage(work, task, required, *, tasks=None, allow_pending=False):
    if not required:
        return
    tasks = work['tasks'] if tasks is None else tasks
    seen, resolved, pending = set(), set(), set()
    queue = list(task['depends_on'])
    while queue:
        key = queue.pop()
        if key in seen:
            continue
        seen.add(key)
        ancestor = tasks[key]
        queue.extend(ancestor['depends_on'])
        if ancestor['capability'] != 'read':
            continue
        if ancestor['status'] != 'DONE':
            pending.add(key)
            continue
        artifact = work['artifacts'][ancestor['output']]
        if digest(artifact) != ancestor['output']:
            raise ValueError('Read artifact identity changed')
        if artifact['kind'] == 'read' and isinstance(artifact['content'], dict):
            resolved.add(artifact['content'].get('evidence_id'))
    missing = set(required)-resolved
    if missing and (not allow_pending or len(pending) < len(missing)):
        raise ValueError(f'Task {task["id"]}: missing selected evidence reads before design: {sorted(missing)}')


def plan(work, tasks, *, enforce_capabilities=False, required_sources=(), require_validation_coverage=False,
         required_evidence=()):
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 16:
        raise ValueError('Supply 1..16 tasks')
    proposed = {}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {'id', 'goal', 'capability', 'depends_on', 'acceptance'}:
            raise ValueError('Invalid task fields')
        key = task['id']
        if (not isinstance(key, str) or not key.isidentifier() or key in proposed
                or task['capability'] not in CAPABILITIES
                or any(not isinstance(task[k], str) or not task[k].strip()
                       for k in ('goal', 'acceptance'))
                or not isinstance(task['depends_on'], list)
                or any(not isinstance(d, str) for d in task['depends_on'])):
            raise ValueError('Invalid task definition')
        proposed[key] = copy.deepcopy(task)
    visited, active = set(), set()

    def visit(key):
        if key not in proposed:
            raise ValueError(f'Missing dependency or cyclic plan: plan is a full replacement; include retained task {key!r}')
        if key in active:
            raise ValueError(f'Missing dependency or cyclic plan: cycle at {key!r}')
        if key in visited:
            return
        active.add(key)
        for dep in proposed[key]['depends_on']:
            visit(dep)
        active.remove(key)
        visited.add(key)

    for key in proposed:
        visit(key)
    if enforce_capabilities:
        required = {'design': {'read'}, 'implement': {'design'},
                    'validate': {'implement'}, 'review': {'design', 'implement', 'validate'},
                    'read_literature': {'search_literature'}}
        for key, task in proposed.items():
            capabilities = [proposed[d]['capability'] for d in task['depends_on']]
            missing = required.get(task['capability'], set()) - set(capabilities)
            if missing:
                raise ValueError(f'Task {key}: required capability dependency missing: {sorted(missing)}')
            if task['capability'] == 'validate' and capabilities.count('implement') != 1:
                raise ValueError(f'Task {key}: each validation must bind exactly one candidate')
    if require_validation_coverage:
        checks = {k: {d for d in t['depends_on'] if proposed[d]['capability'] == 'implement'}
                  for k, t in proposed.items() if t['capability'] == 'validate'}
        for key, task in proposed.items():
            if task['capability'] == 'implement' and not any(key in deps for deps in checks.values()):
                raise ValueError(f'Task {key}: implementation requires validation coverage in this plan')
            if task['capability'] == 'review':
                codes = {d for d in task['depends_on'] if proposed[d]['capability'] == 'implement'}
                validated = set().union(*(checks[d] for d in task['depends_on'] if d in checks))
                if not codes or codes != validated:
                    raise ValueError(f'Task {key}: review validation binding must match its implementation dependencies')
    # Replanning cannot silently certify an old result under new requirements.
    for key, task in proposed.items():
        previous = work['tasks'].get(key)
        unchanged = previous and all(previous[k] == task[k] for k in task)
        task.update(status=previous['status'] if unchanged else 'PENDING',
                    output=previous.get('output') if unchanged else None)
    changed = {k for k, t in proposed.items() if t['status'] != 'DONE'}
    while True:
        more = {k for k, t in proposed.items() if set(t['depends_on']) & changed}
        if more <= changed:
            break
        changed |= more
    for key in changed:
        proposed[key].update(status='PENDING', output=None)
    for task in proposed.values():
        if task['capability'] == 'design':
            require_evidence_coverage(work, task, required_evidence, tasks=proposed, allow_pending=True)
        if task['capability'] in {'implement', 'review'}:
            require_source_context(work, task, required_sources, tasks=proposed, allow_pending=True)
    work['tasks'] = proposed
    work['task_order'] = [t['id'] for t in tasks]
    work['revision'] += 1


def ready(work, key):
    task = work['tasks'][key]
    if task['status'] == 'DONE' or any(work['tasks'][d]['status'] != 'DONE'
                                     for d in task['depends_on']):
        raise ValueError('Task is complete or dependencies are not ready')
    return task


def put(work, name, kind, content, inputs=()):
    if any(i not in work['artifacts'] for i in inputs):
        raise ValueError('Unknown artifact input')
    artifact = {'name': name, 'kind': kind, 'content': copy.deepcopy(content), 'inputs': sorted(inputs)}
    identity = digest(artifact)
    previous = work['heads'].get(name)
    work['artifacts'][identity] = artifact
    work['heads'][name] = identity
    if previous and previous != identity:
        stale = {previous}
        while True:
            more = {k for k, a in work['artifacts'].items() if set(a['inputs']) & stale}
            if more <= stale:
                break
            stale |= more
        for task in work['tasks'].values():
            if task.get('output') in stale:
                task.update(status='PENDING', output=None)
    return identity


def complete(work, key, kind, content):
    task = ready(work, key)
    inputs = [work['tasks'][d]['output'] for d in task['depends_on']]
    identity = put(work, key, kind, content, inputs)
    task.update(status='DONE', output=identity)
    return identity


def context(work, key, *, limit=48000, project=None):
    task = ready(work, key)
    selected = {work['tasks'][d]['output'] for d in task['depends_on']}
    value = {'goal': work['goal'], 'task': task,
             'dependencies': {i: project(i, work['artifacts'][i]) if project else work['artifacts'][i] for i in selected}}
    if len(json.dumps(value, ensure_ascii=False).encode()) > limit:
        raise ValueError('Required context exceeds window; split the task')
    return value


def check_code(source):
    if not isinstance(source, str) or len(source.encode()) > 100000:
        raise ValueError('Invalid model.py size')
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f'SyntaxError at line {exc.lineno}, column {exc.offset}: {exc.msg}') from exc
    if not any(isinstance(n, ast.FunctionDef) and n.name == 'build_model' for n in tree.body):
        raise ValueError('model.py must define build_model(config)')
    return source
