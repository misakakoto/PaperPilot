"""Lossless page sharing and the budget lineage of evidence repairs."""
import copy

from .prototype_context import task_selection
from .prototype_workspace import digest

PAGE_POLICY = 'exact-page-context-sharing-v1'
BUDGET_POLICY = 'evidence-repair-budget-v1'


def check_policy(state):
    if state.get('review_page_sharing_policy') not in (None, PAGE_POLICY):
        raise ValueError('Unknown review page sharing policy')
    if state.get('literature_repair_budget_policy') not in (None, BUDGET_POLICY):
        raise ValueError('Unknown literature repair budget policy')


def project(packet):
    view = copy.deepcopy(packet)
    for finding in view['findings']:
        key = finding.get('page_context_ref')
        page = view['page_contexts'].get(key)
        surrounding = finding.get('surrounding_text')
        if not page or not surrounding:
            continue
        start = page['text'].find(surrounding)
        if start >= 0:
            finding['surrounding_text_ref'] = {'page_context_ref': key, 'start': start, 'end': start+len(surrounding)}
            finding.pop('surrounding_text')
    view['context_projection'] = PAGE_POLICY
    return view


def evidence_repair(state):
    check_policy(state)
    if state.get('literature_repair_budget_policy') != BUDGET_POLICY or state.get('mode') != 'adaptive':
        return False
    work = state['work']
    _, active = task_selection(state)
    if not active or work['tasks'][active[0]]['capability'] not in {'analyze_literature', 'review_literature'}:
        return False
    ancestors, queue = set(), [active[0]]
    while queue:
        key = queue.pop()
        if key in ancestors:
            continue
        ancestors.add(key)
        queue.extend(work['tasks'][key]['depends_on'])
    for extension in state.get('literature_evidence_extensions', []):
        if not ancestors.intersection(extension['added_task_ids']):
            continue
        for binding in extension['failure_bindings']:
            identity = binding['review_artifact_id']
            receipt = work['artifacts'][identity]
            if digest(receipt) != identity or receipt['kind'] != 'review_literature':
                raise ValueError('Evidence repair budget lineage changed')
            if receipt['content'].get('passed') is False:
                return True
    return False
