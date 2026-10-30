"""Short navigation handles bound to a task definition and exact source versions."""
import copy

from . import prototype_selected_text as selected_text
from .prototype_workspace import digest

POLICY = 'task-local-quote-handles-v1'
WORKSET_POLICY = 'shared-byte-workset-v1'
WORKSET_BYTES = 20000


def enabled(state):
    policy = state.get('quote_handle_policy')
    if state.get('analysis_workset_policy') not in (None, WORKSET_POLICY):
        raise ValueError('Unknown analysis workset policy')
    if state.get('analysis_workset_policy') and policy != POLICY:
        raise ValueError('Shared analysis workset requires quote handles')
    if policy not in (None, POLICY):
        raise ValueError('Unknown quote handle policy')
    if policy and state.get('analysis_read_policy') != selected_text.POLICY:
        raise ValueError('Quote handles require the selected original span view')
    return policy == POLICY


def _catalog(state, task):
    work = state['work']
    definition = {k: task[k] for k in ('id', 'goal', 'capability', 'depends_on', 'acceptance')}
    inputs = {key: work['tasks'][key]['output'] for key in task['depends_on']}
    prefix = 'h_' + digest({'task': definition, 'inputs': inputs,
                           'workset_policy': state.get('analysis_workset_policy')})[:12] + '_'
    result = {}
    for identity in sorted(set(inputs.values())):
        artifact = work['artifacts'][identity]
        if artifact['kind'] != 'read_literature':
            continue
        view = selected_text.project(identity, artifact)
        for page in view['content']['pages']:
            for quote_id, _ in page['quotes']:
                result[prefix + str(len(result))] = {'artifact_id': identity, 'quote_id': quote_id}
    return result


def workset(state, task):
    enabled(state)
    handles = _catalog(state, task)
    identities = sorted({row['artifact_id'] for row in handles.values()})
    views = {identity: project(identity, state['work']['artifacts'][identity], handles) for identity in identities}
    if state.get('analysis_workset_policy') == WORKSET_POLICY and views:
        allowance = WORKSET_BYTES // len(views)
        views = {identity: selected_text.bound_view(view, allowance) for identity, view in views.items()}
        shown = {pair[0] for view in views.values() for page in view['content']['pages'] for pair in page['quotes']}
        handles = {key: row for key, row in handles.items() if key in shown}
    return views, handles


def catalog(state, task):
    return workset(state, task)[1]


def project(identity, artifact, handles):
    view = selected_text.project(identity, artifact)
    if artifact['kind'] != 'read_literature':
        return view
    inverse = {(row['artifact_id'], row['quote_id']): key for key, row in handles.items()}
    for page in view['content']['pages']:
        for pair in page['quotes']:
            pair[0] = inverse[(identity, pair[0])]
    view['content']['context_view']['quote_handle_policy'] = POLICY
    view['content']['context_view']['handle_scope'] = 'This task definition and exact input versions only'
    return view


def normalize(state, task, content):
    if not enabled(state):
        return content, None
    handles = catalog(state, task)
    if not isinstance(content, dict) or not isinstance(content.get('findings'), list):
        return content, None  # The ordinary analysis validator owns the schema.
    value, resolutions = copy.deepcopy(content), []
    for index, row in enumerate(value['findings']):
        if not isinstance(row, dict) or not isinstance(row.get('quote_id'), str):
            continue
        key = row['quote_id']
        if not key.startswith('h_'):
            continue  # Canonical IDs from inspect_quotes remain valid.
        original = handles.get(key)
        if not original or row.get('artifact_id') != original['artifact_id']:
            raise ValueError(f'findings[{index}]: Unknown, stale or foreign quote handle; copy one from current selected pages')
        row['quote_id'] = original['quote_id']
        resolutions.append({'finding_index': index, 'handle': key, **original})
    audit = {'policy': POLICY, 'original_content_sha256': digest(content), 'resolutions': resolutions}
    return value, audit if resolutions else None
