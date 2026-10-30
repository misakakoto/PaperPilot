"""Versioned source-only submission adapter; engineering acceptance is unchanged."""
import hashlib
from .prototype_workspace import digest
from .runtime_profile import feature

POLICY = 'source-default-config-v1'
AUTO_POLICY = 'host-validation-v1'
PATCH_POLICY = 'literal-patch-v1'


def check_policy(state):
    if state.get('candidate_input_policy') not in (None, POLICY):
        raise ValueError('Unknown prototype candidate input policy')
    if state.get('automatic_validation_policy') not in (None, AUTO_POLICY):
        raise ValueError('Unknown automatic validation policy')
    if state.get('candidate_patch_policy') not in (None, PATCH_POLICY):
        raise ValueError('Unknown candidate patch policy')


def patch_base(state, task_id):
    check_policy(state)
    if state.get('candidate_patch_policy') != PATCH_POLICY or state.get('profile_id') or not task_id:
        return None
    work = state['work']
    task = work['tasks'][task_id]
    identity = work['heads'].get(task_id)
    artifact = work['artifacts'].get(identity)
    if artifact is None:
        return None
    if digest(artifact) != identity:
        raise ValueError('Patch base artifact identity changed')
    if artifact['name'] != task_id or artifact['kind'] != 'implement':
        return None
    if (any(work['tasks'][d]['status'] != 'DONE' for d in task['depends_on'])
            or artifact['inputs'] != sorted(work['tasks'][d]['output'] for d in task['depends_on'])):
        return None
    return identity


def normalize(state, content, *, task_id=None):
    check_policy(state)
    if isinstance(content, dict) and 'base_artifact_id' in content:
        base = patch_base(state, task_id)
        if (not base or set(content) != {'base_artifact_id', 'replacements', 'config'}
                or content['base_artifact_id'] != base or not isinstance(content['config'], dict)
                or not isinstance(content['replacements'], list) or not 1 <= len(content['replacements']) <= 8):
            raise ValueError('Patch requires the current unchanged base, 1..8 replacements and explicit config')
        source = state['work']['artifacts'][base]['content']['source']
        original = source
        for index, edit in enumerate(content['replacements'], 1):
            if (not isinstance(edit, dict) or set(edit) != {'old', 'new'}
                    or not isinstance(edit['old'], str) or not edit['old']
                    or not isinstance(edit['new'], str) or len(edit['new'].encode()) > 100000):
                raise ValueError('Each old patch substring must occur exactly once')
            matches = source.count(edit['old'])
            if matches != 1:
                raise ValueError(f'Patch replacement {index}: old substring must occur exactly once (found {matches}); '
                                 'copy original source only, excluding view metadata and symbol labels')
            source = source.replace(edit['old'], edit['new'], 1)
        return {'source': source, 'config': content['config']}, {
            'policy': PATCH_POLICY, 'base_artifact_id': base, 'config_origin': 'EXPLICIT_MODEL_CONFIG',
            'source_before_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'edit_count': len(content['replacements'])}
    source_object = (feature(state, 'source_object_default_config')
                     and isinstance(content, dict) and set(content) == {'source'}
                     and isinstance(content['source'], str))
    if state.get('candidate_input_policy') == POLICY and (isinstance(content, str) or source_object):
        if state.get('profile_id'):
            raise ValueError('Source-only submissions cannot replace a bound baseline')
        source = content['source'] if source_object else content
        return {'source': source, 'config': {}}, {
            'policy': 'source-object-default-config-v1' if source_object else POLICY,
            'config_origin': 'HOST_DECLARED_EMPTY_DEFAULT',
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'source_unchanged': True}
    return content, None
