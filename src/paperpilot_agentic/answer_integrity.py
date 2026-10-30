"""Account for requested content across a source-audited revision.

This is an accountability contract, not semantic proof: model resolutions are
fallible. No prior statement is inserted into the answer or promoted to evidence.
"""
from copy import deepcopy
import re


CONSISTENCY_POLICY = (
    'Also audit the answer AS A WHOLE, not only isolated facts. Compare totals '
    'with enumerated members, summaries with details, and asserted experimental '
    'roles/conditions across statements in the SAME scope. A list of some examples '
    'is not an exclusive total; different explicitly labeled stages or variants '
    'are not a contradiction. Return answer_conflicts: '
    '[{"fact_ids":["conflicting current fact IDs"],"reason":"brief scope-specific conflict"}]. '
    'Each conflict must name at least two current facts; return [] if none. '
    'Do not mark coverage COMPLETE with unresolved conflicts. Isolated source '
    'mentions cannot validate a total contradicted by another requested item. '
    'Do not infer the total merely by counting paragraphs or fact IDs.'
)


def validate_answer_conflicts(audit, draft):
    conflicts = audit.get('answer_conflicts', [])
    if not isinstance(conflicts, list) or len(conflicts) > 12:
        raise ValueError('Answer conflicts must be a bounded list')
    facts = {f['id'] for f in draft['facts']}
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            raise ValueError('Invalid answer conflict')
        ids = conflict.get('fact_ids')
        if (not isinstance(ids, list) or not 2 <= len(ids) <= 12
                or any(not isinstance(i, str) or i not in facts for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Answer conflict requires at least two distinct current fact IDs')
        if not isinstance(conflict.get('reason'), str) or not conflict['reason'].strip():
            raise ValueError('Answer conflict requires a reason')


_DATASET_NAME = r'[A-Za-z][A-Za-z0-9_-]{1,63}(?:\s*\+)?'
_DATASET_LIST = rf'{_DATASET_NAME}(?:\s*(?:、|，|,|和|与|及|\band\b)\s*{_DATASET_NAME})+'
_PAPER_SUBJECT = re.compile(r'\b([A-Za-z][A-Za-z0-9_+-]{1,63})\s*论文|(?<!\w)(本文|本研究)')
_CLOSED_DATASETS = re.compile(
    rf'^(?:共|总共|一共)?(?:使用|采用)了?\s*(?P<names>{_DATASET_LIST})\s*'
    r'(?P<count>[一二三四五六七八九十0-9]{1,3})\s*个数据集')
_ADDITIONAL_DATASET = re.compile(
    rf'^(?:还|也)(?:使用|采用)了?\s*(?P<name>{_DATASET_NAME})\s*数据集')
_DATASET_SEPARATOR = re.compile(r'\s*(?:、|，|,|和|与|及|\band\b)\s*')
_CHINESE_COUNTS = dict(zip('一二三四五六七八九十', range(1, 11)))


def _paper_and_tail(text):
    match = _PAPER_SUBJECT.search(text)
    if not match:
        return None, ''
    return (match.group(1) or match.group(2)).casefold(), text[match.end():].strip()


def _closed_dataset_list(text):
    """Recognize only an explicit, exhaustive named list for one paper."""
    paper, tail = _paper_and_tail(text)
    match = _CLOSED_DATASETS.match(tail)
    if not match:
        return None
    names = {re.sub(r'[^a-z0-9]', '', name.casefold())
             for name in _DATASET_SEPARATOR.split(match['names'])}
    count = int(match['count']) if match['count'].isdigit() else _CHINESE_COUNTS.get(match['count'])
    if not count or len(names) != count:
        return None
    return paper, names


def _additional_dataset(text):
    paper, tail = _paper_and_tail(text)
    match = _ADDITIONAL_DATASET.match(tail)
    if not match:
        return None
    return paper, re.sub(r'[^a-z0-9]', '', match['name'].casefold())


def _closed_dataset_conflicts(draft):
    """Find a narrow count/list contradiction missed by the model audit.

    An unqualified, numerically consistent list followed by 'also uses X' for
    the same explicitly named paper is contradictory when X is not in the list.
    Scoped lists, examples and cross-paper comparisons are deliberately left
    to semantic review. This is not a general NLI or dataset-use proof.
    """
    facts = [fact for fact in draft['facts'] if fact.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT']
    closed = [(fact['id'], _closed_dataset_list(fact['text'])) for fact in facts]
    extra = [(fact['id'], _additional_dataset(fact['text'])) for fact in facts]
    conflicts = []
    for total_id, inventory in closed:
        if inventory is None:
            continue
        paper, names = inventory
        for extra_id, addition in extra:
            if extra_id == total_id or addition is None:
                continue
            added_paper, added_name = addition
            if paper == added_paper and added_name not in names:
                conflicts.append({'fact_ids': [total_id, extra_id],
                                  'reason': 'Closed dataset total excludes an additional dataset used by the same paper.',
                                  'source': 'closed-dataset-list-v1'})
    return conflicts[:12]


def quarantine_answer_conflicts(audit, draft, spans, *, question):
    """Do not publish model-flagged conflicting claims as jointly supported."""
    effective = deepcopy(audit)
    deterministic = _closed_dataset_conflicts(draft)
    conflicts = audit.get('answer_conflicts', []) + deterministic
    supported = {a['fact_id'] for a in audit['assessments'] if a['verdict'] == 'SUPPORTED'}
    # A withheld endpoint cannot form a conflict in the published answer.
    # Deterministic source conflicts retain their conservative treatment.
    active = [r for r in audit.get('answer_conflicts', [])
              if len(supported.intersection(r['fact_ids'])) >= 2] + deterministic
    ids = {fid for row in active for fid in row['fact_ids']}
    report = {'version': 'answer-consistency-v2-closed-dataset-list', 'conflicts': conflicts,
              'deterministic_conflicts': deterministic,
              'assessment_status': 'MODEL_REPORTED' if 'answer_conflicts' in audit else 'NOT_REPORTED',
              'active_conflicts': active, 'changed_fact_ids': [], 'facts': []}
    if not ids:
        return audit, report
    sources = {f['id']: set(f['evidence_ids']) for f in draft['facts']}
    by_span = {s['span_id']: s for s in spans}
    for row in effective['assessments']:
        if row['fact_id'] not in ids:
            continue
        # Only exact originals already associated with this fact guide revision.
        selected = row.get('span_ids') if isinstance(row.get('span_ids'), list) else []
        candidates = [by_span[sid] for sid in selected if isinstance(sid, str) and sid in by_span
                      and by_span[sid]['evidence_id'] in sources[row['fact_id']]]
        report['facts'].append({'fact_id': row['fact_id'], 'candidate_source_spans': candidates})
        if row['verdict'] == 'SUPPORTED':
            row.update(verdict='UNCERTAIN', span_ids=[], consistency_reason='unresolved_answer_conflict')
            report['changed_fact_ids'].append(row['fact_id'])
    note = ('当前回答的部分概述与细项存在未解决的冲突，相关结论暂不发布，仍需核对。'
            if re.search(r'[\u4e00-\u9fff]', question) else
            'The current answer contains unresolved conflicts between statements; affected conclusions are withheld.')
    effective['coverage'] = 'PARTIAL'
    effective['missing_aspects'] = list(dict.fromkeys(effective.get('missing_aspects', []) + [note]))
    return effective, report


REVISION_POLICY = (
    'prior_answer_obligations are fallible earlier DIRECT statements, NOT evidence. '
    'Recheck each against the current original sources. Return revision_resolutions '
    'with one object per previous_fact_id: {"previous_fact_id":"...",'
    '"status":"RETAINED|CORRECTED|WITHDRAWN|UNRESOLVED",'
    '"current_fact_ids":["..."],"source_span_ids":["..."],"reason":"brief justification"}. '
    'RETAINED requires the new answer to cover the same requested information; '
    'CORRECTED requires a source-supported replacement, not deletion of an unrelated '
    'requested detail. WITHDRAWN requires originals establishing why the old claim '
    'was wrong or irrelevant; absence of support alone is UNRESOLVED. '
    'Do not preserve a false old count after adding new items. Do not count newly '
    'added information as resolving a different omitted item. Supporting source '
    'spans for RETAINED/CORRECTED must be selected by the corresponding current '
    'fact assessments. If the change is not resolved, keep coverage PARTIAL. '
    'Never copy a prior claim into the answer on the strength of its old verdict.'
)


def prior_obligations(feedback):
    if not feedback:
        return []
    assessments = {a['fact_id']: a for a in (feedback.get('audit') or {}).get('assessments', [])}
    return [{'previous_fact_id': fact['id'], 'text': fact['text'],
             'evidence_ids': fact['evidence_ids'], 'kind': fact.get('kind', 'SOURCE_FACT')}
            for fact in (feedback.get('draft') or {}).get('facts', [])
            if assessments.get(fact['id'], {}).get('verdict') == 'SUPPORTED'
            and assessments[fact['id']].get('relevance') == 'DIRECT']


def validate_revision_resolutions(audit, draft, spans, obligations):
    resolutions = audit.get('revision_resolutions', [])
    if not isinstance(resolutions, list) or len(resolutions) > 12:
        raise ValueError('Revision resolutions must be a bounded list')
    old_ids = {o['previous_fact_id'] for o in obligations}
    fact_ids = {f['id'] for f in draft['facts']}
    span_ids = {s['span_id'] for s in spans}
    seen = set()
    for row in resolutions:
        if not isinstance(row, dict):
            raise ValueError('Invalid revision resolution')
        previous = row.get('previous_fact_id')
        if not isinstance(previous, str) or previous not in old_ids or previous in seen:
            raise ValueError('Revision must resolve unique supplied previous fact IDs')
        seen.add(previous)
        status = row.get('status')
        if status not in {'RETAINED', 'CORRECTED', 'WITHDRAWN', 'UNRESOLVED'}:
            raise ValueError('Unknown revision resolution status')
        for key, allowed in [('current_fact_ids', fact_ids), ('source_span_ids', span_ids)]:
            ids = row.get(key)
            if (not isinstance(ids, list) or len(ids) > 12
                    or any(not isinstance(i, str) or i not in allowed for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError('Revision resolution requires valid ' + key)
        if not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('Revision resolution requires a reason')
        if status in {'RETAINED', 'CORRECTED'} and not row['current_fact_ids']:
            raise ValueError('Retained/corrected content requires current fact IDs')
        if status != 'UNRESOLVED' and not row['source_span_ids']:
            raise ValueError('Resolved revision requires original source spans')


def enforce_revision_accounting(audit, draft, obligations, *, question):
    """Unexplained loss downgrades coverage, without appending unreviewed facts."""
    report = {'version': 'revision-accounting-v1', 'resolutions': [], 'unresolved_previous_fact_ids': []}
    if not obligations:
        return audit, report
    effective = deepcopy(audit)
    assessments = {a['fact_id']: a for a in effective['assessments']}
    decisions = {r['previous_fact_id']: r for r in effective.get('revision_resolutions', [])}
    for old in obligations:
        previous = old['previous_fact_id']
        decision = decisions.get(previous)
        if decision is not None:
            status = decision['status']
            if status in {'RETAINED', 'CORRECTED'}:
                current = [assessments[i] for i in decision['current_fact_ids']]
                selected = {s for a in current for s in a.get('span_ids', [])}
                if (any(a['verdict'] != 'SUPPORTED' or a.get('relevance') != 'DIRECT' for a in current)
                        or not set(decision['source_span_ids']) <= selected):
                    status = 'UNRESOLVED'
            report['resolutions'].append({**decision, 'effective_status': status})
        else:
            # Exact unchanged content can be accounted for without another model
            # declaration, but it MUST pass the fresh audit and guards again.
            same = [f['id'] for f in draft['facts'] if f['text'] == old['text']
                    and f.get('kind', 'SOURCE_FACT') == old['kind']
                    and set(f['evidence_ids']) == set(old['evidence_ids'])
                    and assessments[f['id']]['verdict'] == 'SUPPORTED'
                    and assessments[f['id']].get('relevance') == 'DIRECT']
            status = 'RETAINED' if same else 'UNRESOLVED'
            report['resolutions'].append({'previous_fact_id': previous, 'effective_status': status,
                                          'current_fact_ids': same, 'mode': 'exact_reaudited_content'})
        if status == 'UNRESOLVED':
            report['unresolved_previous_fact_ids'].append(previous)
    if report['unresolved_previous_fact_ids']:
        note = ('本轮修订尚未交代先前部分回答内容的保留或纠正情况，完整性仍待核对。'
                if re.search(r'[\u4e00-\u9fff]', question) else
                'The current revision leaves earlier requested content unaccounted for; completeness needs review.')
        effective['coverage'] = 'PARTIAL'
        effective['missing_aspects'] = list(dict.fromkeys(effective.get('missing_aspects', []) + [note]))
    return effective, report
