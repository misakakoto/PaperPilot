"""A real but uncited source is a Writer binding gap, not a format-only typo."""
from copy import deepcopy
import re


def quarantine_misbound_support(value, draft, spans, *, question):
    """Downgrade known cross-citation spans without changing facts or citations.

    Unknown IDs and malformed structures still go through normal validation.
    Navigation stays within supplied originals and does not certify entailment.
    """
    report = {'version': 'citation-binding-navigation-v2-revision-basis', 'changed_fact_ids': [], 'facts': []}
    if (not isinstance(value, dict) or not isinstance(value.get('assessments'), list)
            or value.get('coverage') not in {'COMPLETE', 'PARTIAL', 'UNANSWERABLE'}):
        return value, report
    effective = deepcopy(value)
    report.update(original_coverage=value.get('coverage'),
                  original_missing_aspects=deepcopy(value.get('missing_aspects', [])),
                  original_omitted_answer_spans=deepcopy(value.get('omitted_answer_spans', [])))
    facts = {fact['id']: fact for fact in draft['facts']}
    by_span = {span['span_id']: span for span in spans}
    for row in effective['assessments']:
        if not isinstance(row, dict) or row.get('verdict') != 'SUPPORTED':
            continue
        fact_id = row.get('fact_id')
        ids = row.get('span_ids')
        if (not isinstance(fact_id, str) or fact_id not in facts or not isinstance(ids, list)
                or not 1 <= len(ids) <= 12 or any(not isinstance(sid, str) or sid not in by_span for sid in ids)):
            continue
        if not any(by_span[sid]['evidence_id'] not in facts[fact_id]['evidence_ids'] for sid in ids):
            continue
        report['facts'].append({'fact_id': fact_id,
                                'candidate_source_spans': [by_span[sid] for sid in dict.fromkeys(ids)]})
        report['changed_fact_ids'].append(fact_id)
        row.update(verdict='UNCERTAIN', span_ids=[],
                   citation_binding_reason='support_selected_outside_writer_citations')
    if report['changed_fact_ids']:
        effective['coverage'] = 'PARTIAL'
        # Do not hide a malformed gap field; let structural validation reject it.
        gaps = effective.get('missing_aspects', [])
        if isinstance(gaps, list) and all(isinstance(gap, str) for gap in gaps):
            note = ('部分结论的引用未覆盖审核选中的原文，需要重新定位并核对。'
                    if re.search(r'[\u4e00-\u9fff]', question) else
                    'Some claims do not cite the original passages selected in review; citations need rechecking.')
            effective['missing_aspects'] = list(dict.fromkeys(gaps + [note]))
    return effective, report
