"""Owned immutable artifact navigation, with bounded exact-text pages."""
import json

from .prototype_workspace import digest


def artifact_index(work):
    current = {t['output'] for t in work['tasks'].values() if t['status'] == 'DONE'}
    return {identity: {**{k: artifact[k] for k in ('name', 'kind', 'inputs')},
                       'current': identity in current}
            for identity, artifact in work['artifacts'].items()}


def inspect_artifact(work, arguments):
    if (not isinstance(arguments, dict) or 'artifact_id' not in arguments
            or set(arguments) - {'artifact_id', 'offset', 'limit'}):
        raise ValueError('Expected artifact_id and optional character offset/limit')
    identity = arguments['artifact_id']
    if not isinstance(identity, str) or identity not in work['artifacts']:
        raise ValueError('Unknown artifact ID in this prototype')
    offset, limit = arguments.get('offset', 0), arguments.get('limit', 4000)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 4000:
        raise ValueError('Use nonnegative character offset and limit 1..4000')
    artifact = work['artifacts'][identity]
    if digest(artifact) != identity:
        raise ValueError('Prototype artifact identity changed')
    text = json.dumps(artifact, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if offset > len(text):
        raise ValueError('Offset exceeds artifact length')
    end = min(offset + limit, len(text))
    return {'status': 'OK', 'action': 'inspect_artifact', 'artifact_id': identity,
            **artifact_index(work)[identity], 'workspace_revision': work['revision'],
            'encoding': 'json-unicode-characters', 'offset': offset, 'total_characters': len(text),
            'next_offset': end if end < len(text) else None, 'text': text[offset:end]}
