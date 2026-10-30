"""Select registered source versions and invalidate their exact read consumers."""
from .prototype_workspace import SOURCE_POLICY, digest, task_rows


def prepare(harness, original, updated, replacements):
    if original.get('source_policy') != SOURCE_POLICY or not original.get('source_run_id'):
        raise ValueError('Source version revisions require a task with recorded reading-run ownership')
    if (not isinstance(replacements, dict) or not replacements
            or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in replacements.items())
            or set(replacements) - set(original['sources'])):
        raise ValueError('Replace selected source IDs with registered version IDs')
    available = {s['evidence_id']: s for s in harness.db.list_evidence(original['source_run_id'])}
    targets = [replacements.get(sid, sid) for sid in original['sources']]
    if len(set(targets)) != len(targets) or set(targets) - set(available):
        raise ValueError('Replacement sources must be distinct and owned by the same reading run')
    for old, new in replacements.items():
        if old != new and new in original['sources']:
            raise ValueError('Replacement version must be new to the selected sources')
        if original['sources'][old].get('evidence_type') != available[new].get('evidence_type'):
            raise ValueError('Source replacement must preserve evidence type')
        if original.get('profile_id') and original['sources'][old].get('evidence_type') == 'CODE':
            raise ValueError('A bound prototype baseline cannot be replaced in place')
    changed = {old: new for old, new in replacements.items() if old != new}
    if not changed:
        return None
    updated['sources'] = {changed.get(sid, sid): available[changed[sid]] if sid in changed else record
                          for sid, record in original['sources'].items()}
    updated['required_evidence_ids'] = sorted(updated['sources'])
    updated['implementation_source_ids'] = sorted(changed.get(sid, sid)
                                                  for sid in original.get('implementation_source_ids', []))
    rows = task_rows(original['work'])
    for row in rows:
        task = original['work']['tasks'][row['id']]
        if task['capability'] != 'read' or task['status'] != 'DONE':
            continue
        artifact = original['work']['artifacts'][task['output']]
        if digest(artifact) != task['output']:
            raise ValueError('Read artifact identity changed')
        if artifact['kind'] != 'read' or not isinstance(artifact['content'], dict):
            raise ValueError('Read artifact does not contain a source record')
        sid = artifact['content'].get('evidence_id')
        if sid in changed:
            row['goal'] = 'Read registered source version '+changed[sid]
            row['acceptance'] = 'Read exact source record '+digest(updated['sources'][changed[sid]])
    return rows
