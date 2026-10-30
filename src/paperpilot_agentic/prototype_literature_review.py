"""Version-bound claim critique. Model judgments are never host scientific facts."""
import copy

from . import prototype_analysis as analysis
from .prototype_workspace import digest, plan, task_rows

POLICY = 'source-claim-review-v1'
EXTENSION_POLICY = 'evidence-gap-extension-v2'
CORRECTION_POLICY = 'supported-null-correction-v1'
CAPABILITY = 'review_literature'
ROLES = {'implemented_method', 'reported_result', 'future_work', 'related_work', 'unclear'}


def enabled(state):
    if state.get('review_correction_policy') not in (None, CORRECTION_POLICY):
        raise ValueError('Unknown review correction policy')
    policy = state.get('literature_review_policy')
    if policy not in (None, POLICY):
        raise ValueError('Unknown literature review policy')
    if policy and state.get('literature_analysis_policy') != analysis.POLICY:
        raise ValueError('Literature review requires the source-span analysis policy')
    return policy == POLICY


def normalize(state, content):
    enabled(state)
    if (state.get('review_correction_policy') != CORRECTION_POLICY
            or not isinstance(content, dict) or not isinstance(content.get('checks'), list)):
        return content, None
    result, changes = copy.deepcopy(content), []
    for index, check in enumerate(result['checks']):
        if (isinstance(check, dict) and check.get('interpretation_supported') is True
                and 'required_correction' in check and check['required_correction'] is None):
            check['required_correction'] = ''
            changes.append({'finding_index': index, 'field': 'required_correction', 'from': None, 'to': ''})
    audit = {'policy': CORRECTION_POLICY, 'original_content_sha256': digest(content), 'changes': changes}
    return result, audit if changes else None


def check_plan(state, rows):
    active = enabled(state)
    tasks = {t['id']: t for t in rows}
    if needs_replanning(state):
        _check_evidence_replan(state, tasks)
    for task in rows:
        if task['capability'] == CAPABILITY:
            if not active:
                raise ValueError('Literature review capability is not registered')
            if sum(tasks[d]['capability'] == 'analyze_literature' for d in task['depends_on']) != 1:
                raise ValueError('Literature review requires exactly one direct analysis dependency')
        if active and task['capability'] == 'design':
            analyses = {d for d in task['depends_on'] if tasks[d]['capability'] == 'analyze_literature'}
            covered = {d for r in task['depends_on'] if tasks[r]['capability'] == CAPABILITY
                       for d in tasks[r]['depends_on'] if tasks[d]['capability'] == 'analyze_literature'}
            if not analyses or analyses != covered:
                raise ValueError('Research design requires direct review coverage of its analysis dependencies')


def _check_evidence_replan(state, proposed):
    """New observations must feed analysis; a review cannot be planned away."""
    current = state['work']['tasks']
    for key, old in current.items():
        new = proposed.get(key)
        if (not new or new['capability'] != old['capability']
                or new['acceptance'] != old['acceptance']):
            raise ValueError('Evidence replanning must retain tasks, capabilities and acceptance')
        if old['capability'] != 'analyze_literature' and new['goal'] != old['goal']:
            raise ValueError('Evidence replanning must preserve existing task goals')
        if not set(old['depends_on']) <= set(new['depends_on']):
            raise ValueError('Evidence replanning must retain existing dependency edges')
        if old['status'] == 'DONE' and old['capability'] in {'read', 'read_literature', 'search_literature'}:
            if any(new[field] != old[field] for field in new):
                raise ValueError('Evidence replanning must preserve completed observations exactly')
    missing = {gap['analysis_artifact_id'] for gap in evidence_gaps(state)}
    for old in current.values():
        if old['output'] not in missing:
            continue
        dependencies = proposed[old['id']]['depends_on']
        if not any(key not in current and key in proposed
                   and proposed[key]['capability'] == 'read_literature' for key in dependencies):
            raise ValueError('Evidence replanning must connect new reading directly to the failed analysis')


def packet(work, task):
    deps = [work['tasks'][d] for d in task['depends_on']
            if work['tasks'][d]['capability'] == 'analyze_literature']
    if len(deps) != 1 or deps[0]['status'] != 'DONE':
        raise ValueError('Literature review requires one completed analysis')
    identity = deps[0]['output']
    artifact = work['artifacts'][identity]
    if digest(artifact) != identity:
        raise ValueError('Analysis artifact identity changed')
    page_contexts = {}
    for current in set(artifact['inputs']) | {work['tasks'][d]['output'] for d in task['depends_on']
            if work['tasks'][d]['status'] == 'DONE'}:
        item = work['artifacts'][current]
        if digest(item) != current:
            raise ValueError('Review context source identity changed')
        if item['kind'] not in analysis.LITERATURE or not item['content'].get('ok'):
            continue
        for row in item['content'].get('evidence', []):
            provenance = row.get('provenance', {})
            if provenance.get('complete_page') is True and provenance.get('paper_sha256'):
                key = f'{provenance["paper_sha256"]}:{row.get("page")}'
                page_contexts[key] = {'artifact_id': current, 'source_uri': row['source_uri'],
                                     'page': row['page'], 'text': row['content'],
                                     'reading_order': provenance.get('reading_order')}
    findings, missing = [], []
    for index, finding in enumerate(artifact['content']['findings']):
        original = analysis._original_span(work, finding)
        if (not original or finding['source_artifact_id'] not in artifact['inputs']
                or original['text'] != finding['quote']):
            raise ValueError('Reviewed analysis source provenance changed')
        source = work['artifacts'][finding['source_artifact_id']]
        row = source['content']['evidence'][original['chunk_index']]
        text = row['content']
        start = max(0, original['start']-400)
        end = min(len(text), original['end']+400)
        context_key = None
        if str(original['source_uri']).startswith('paper://'):
            sha = str(original['source_uri']).split('/')[2]
            context_key = f'{sha}:{original["page"]}'
            if context_key not in page_contexts:
                missing.append({'paper_sha256': sha, 'page': original['page'],
                                'reason': 'Retrieved chunks do not establish the full local page context'})
        findings.append({'finding_index': index, 'quote_id': finding['quote_id'],
            'interpretation': finding['interpretation'], 'quote': original['text'],
            'source_artifact_id': finding['source_artifact_id'], 'source_uri': original['source_uri'],
            'page': original['page'], 'surrounding_text': text[start:end],
            'surrounding_range': [start, end], 'complete_chunk': start == 0 and end == len(text),
            'page_context_ref': context_key if context_key in page_contexts else None})
    # Walk the exact dependency chain, not the whole unrelated history.
    visited, queue, searches = set(), list(artifact['inputs']), []
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        item = work['artifacts'][current]
        if digest(item) != current:
            raise ValueError('Literature dependency identity changed')
        queue.extend(item['inputs'])
        if item['kind'] == 'search_literature':
            evidence = item['content'].get('evidence', [])
            searches.append({'artifact_id': current, 'ok': item['content'].get('ok'),
                             'returned_rows': len(evidence),
                             'source_uris': sorted({str(e.get('source_uri', '')) for e in evidence})})
    referenced = {f['page_context_ref'] for f in findings if f['page_context_ref']}
    return {'analysis_artifact_id': identity, 'findings': findings, 'search_receipts': searches,
            'additional_analysis_fields': {k: copy.deepcopy(v) for k, v in artifact['content'].items()
                                           if k not in {'findings', 'gaps', 'directions'}},
            'page_contexts': {k: v for k, v in page_contexts.items() if k in referenced},
            'missing_page_context': list({(x['paper_sha256'], x['page']): x for x in missing}.values()),
            'gaps': artifact['content']['gaps'], 'directions': artifact['content']['directions'],
            'scope': 'Exact cited passages with bounded surrounding text, not full-paper coverage'}


def resolve(work, task, content):
    source = packet(work, task)
    if (not isinstance(content, dict) or set(content) != {'checks', 'coverage'}
            or not isinstance(content['checks'], list)):
        raise ValueError('Literature review requires checks and coverage')
    expected = {f['finding_index']: f for f in source['findings']}
    seen, issues = set(), []
    for check in content['checks']:
        if (not isinstance(check, dict) or set(check) != {
                'finding_index', 'quote_id', 'source_role', 'interpretation_supported', 'reason', 'required_correction'}
                or type(check['finding_index']) is not int or check['finding_index'] not in expected
                or check['finding_index'] in seen or check['quote_id'] != expected[check['finding_index']]['quote_id']
                or not isinstance(check['source_role'], str) or check['source_role'] not in ROLES
                or type(check['interpretation_supported']) is not bool
                or not isinstance(check['reason'], str) or not check['reason'].strip()
                or not isinstance(check['required_correction'], str)
                or not check['interpretation_supported'] and not check['required_correction'].strip()):
            raise ValueError('Review each exact finding once with source role, judgment, reason and correction')
        seen.add(check['finding_index'])
        if not check['interpretation_supported']:
            issues.append(check['required_correction'])
    coverage = content['coverage']
    if (seen != set(expected) or not isinstance(coverage, dict)
            or set(coverage) != {'adequate_for_design', 'reason', 'limitations'}
            or type(coverage['adequate_for_design']) is not bool
            or not isinstance(coverage['reason'], str) or not coverage['reason'].strip()
            or not isinstance(coverage['limitations'], list)
            or any(not isinstance(x, str) for x in coverage['limitations'])):
        raise ValueError('Review must cover all findings and state actual evidence coverage')
    if not coverage['adequate_for_design']:
        issues.append(coverage['reason'])
    if source['missing_page_context']:
        issues.append('Read the cited local pages before accepting interpretations: '+str(source['missing_page_context']))
    return {**copy.deepcopy(content), 'analysis_artifact_id': source['analysis_artifact_id'],
            'issues': issues, 'passed': not issues, 'missing_page_context': source['missing_page_context'],
            'assurance': 'MODEL_JUDGMENT_NOT_HOST_VERIFIED', 'scientific_validity': 'NOT_VERIFIED'}


def require_design(state, task):
    if not enabled(state):
        return
    work = state['work']
    analyses = {work['tasks'][d]['output'] for d in task['depends_on']
                if work['tasks'][d]['capability'] == 'analyze_literature'}
    covered = set()
    for dep in task['depends_on']:
        current = work['tasks'][dep]
        if current['capability'] != CAPABILITY or current['status'] != 'DONE':
            continue
        artifact = work['artifacts'][current['output']]
        if digest(artifact) != current['output']:
            raise ValueError('Literature review artifact identity changed')
        content = artifact['content']
        if (content.get('passed') is True and not content.get('issues')
                and content['analysis_artifact_id'] in artifact['inputs']):
            covered.add(content['analysis_artifact_id'])
    if not analyses or analyses != covered:
        raise ValueError('Research design requires a passing current literature review; repair analysis or gather evidence')


def repair_options(state):
    if not enabled(state) or state.get('plan_requires_work') or state.get('mode') != 'adaptive':
        return {}
    work, options = state['work'], {}
    if len(state.get('literature_repair_history', [])) >= 2:
        return options
    for key, task in work['tasks'].items():
        if task['capability'] != CAPABILITY or task['status'] != 'DONE':
            continue
        receipt = work['artifacts'][task['output']]
        if receipt['content'].get('passed') is not False:
            continue
        if receipt['content'].get('coverage', {}).get('adequate_for_design') is False:
            # Missing observations cannot be repaired by rewriting the same analysis.
            continue
        for dep in task['depends_on']:
            current = work['tasks'][dep]
            if current['capability'] == 'analyze_literature' and current['output'] in receipt['inputs']:
                options[dep] = [{'failure_task_id': key, 'receipt_artifact_id': task['output'],
                                 'analysis_artifact_id': current['output'], 'failure_kind': 'LITERATURE_REVIEW'}]
    return options


def evidence_gaps(state):
    if not enabled(state):
        return []
    work = state['work']
    gaps = []
    for task in work['tasks'].values():
        if task['capability'] != CAPABILITY or task['status'] != 'DONE':
            continue
        artifact = work['artifacts'][task['output']]
        if digest(artifact) != task['output']:
            raise ValueError('Literature review artifact identity changed')
        content = artifact['content']
        if content.get('coverage', {}).get('adequate_for_design') is False:
            gaps.append({'review_artifact_id': task['output'], 'analysis_artifact_id': content['analysis_artifact_id'],
                         'coverage': content['coverage'], 'missing_page_context': content.get('missing_page_context', []),
                         'required_action': 'Replan additional literature reading; rewriting alone cannot supply missing evidence'})
    return gaps


def needs_replanning(state):
    policy = state.get('evidence_replanning_policy')
    if policy not in (None, 'evidence-gap-planning-v1', EXTENSION_POLICY):
        raise ValueError('Unknown evidence replanning policy')
    if not policy:
        return False
    from .prototype_context import task_selection
    _, selected = task_selection(state)
    if not selected:
        return False
    task = state['work']['tasks'][selected[0]]
    if task['capability'] != 'design':
        return False
    inputs = {state['work']['tasks'][d].get('output') for d in task['depends_on']}
    return any(gap['review_artifact_id'] in inputs for gap in evidence_gaps(state))


def evidence_extension(state, arguments):
    """Compile explicit new observations into a preserving full plan revision."""
    if not needs_replanning(state):
        raise ValueError('Evidence extension requires a current evidence gap')
    if not isinstance(arguments, dict) or set(arguments) != {'analysis_task_id', 'tasks'}:
        raise ValueError('Supply analysis_task_id and new tasks')
    target, additions = arguments['analysis_task_id'], arguments['tasks']
    current = state['work']['tasks']
    gaps = {gap['analysis_artifact_id'] for gap in evidence_gaps(state)}
    if (not isinstance(target, str) or target not in current
            or current[target]['capability'] != 'analyze_literature' or current[target]['output'] not in gaps):
        raise ValueError('Select the analysis bound to a current failed evidence review')
    if not isinstance(additions, list) or not 1 <= len(additions) <= 3:
        raise ValueError('Supply 1..3 new search/read tasks')
    for task in additions:
        if (not isinstance(task, dict) or not isinstance(task.get('id'), str) or task['id'] in current
                or task.get('capability') not in {'search_literature', 'read_literature'}):
            raise ValueError('Evidence extension only adds new search/read tasks')
    reads = [t['id'] for t in additions if t['capability'] == 'read_literature']
    if not reads:
        raise ValueError('Evidence extension requires new reading')
    rows = task_rows(state['work'])
    next(t for t in rows if t['id'] == target)['depends_on'].extend(reads)
    # Insert before the target so serial scheduling follows the declared dependency.
    index = next(i for i, t in enumerate(rows) if t['id'] == target)
    return rows[:index] + copy.deepcopy(additions) + rows[index:]


def reading_memory(state, task_id):
    """Bounded navigation facts for new reading, including invalidated critique."""
    work = state['work']
    history = []
    for task in work['tasks'].values():
        if task['capability'] != 'read_literature' or task['status'] != 'DONE':
            continue
        identity = task['output']
        artifact = work['artifacts'][identity]
        if digest(artifact) != identity:
            raise ValueError('Reading memory artifact identity changed')
        content = artifact['content']
        if content.get('ok'):
            history.append({'task_id': task['id'], 'artifact_id': identity,
                            'local_read_request': copy.deepcopy(content.get('local_read_request')),
                            'requested_pages': content.get('requested_pages'),
                            'observed_pages': sorted({e['page'] for e in content.get('evidence', [])
                                                      if isinstance(e.get('page'), int)})})
    reasons = []
    for extension in state.get('literature_evidence_extensions', []):
        if task_id not in extension['added_task_ids']:
            continue
        for binding in extension['failure_bindings']:
            identity = binding['review_artifact_id']
            artifact = work['artifacts'][identity]
            if digest(artifact) != identity or artifact['kind'] != CAPABILITY:
                raise ValueError('Reading assignment review identity changed')
            reasons.append({'review_artifact_id': identity,
                            'coverage': copy.deepcopy(artifact['content']['coverage']),
                            'status': 'HISTORICAL_FAILED_REVIEW_GUIDANCE_NOT_CURRENT_APPROVAL'})
    return {'completed_reads': history, 'assigned_evidence_gaps': reasons,
            'instruction': 'Reuse already-read observations; choose new relevant pages. History does not establish complete-paper coverage.'}


def repair(state, arguments, reason, index):
    options = repair_options(state)
    if (not isinstance(arguments, dict) or set(arguments) != {'target_task_id'}
            or not isinstance(arguments['target_task_id'], str) or arguments['target_task_id'] not in options):
        raise ValueError('Select a current literature review repair target')
    target = arguments['target_task_id']
    rows = task_rows(state['work'])
    next(t for t in rows if t['id'] == target)['goal'] += f' Review repair {index}: {reason}'
    plan(state['work'], rows)
    state.setdefault('literature_repair_history', []).append({'turn': index, 'target_task_id': target,
        'reason': reason, 'failure_bindings': options[target], 'workspace_revision': state['work']['revision']})
    state['plan_requires_work'] = True
