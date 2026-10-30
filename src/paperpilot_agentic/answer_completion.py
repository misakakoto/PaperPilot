"""Bind caveat roles to this draft; completion never proves factual accuracy."""


def isolate_disclosure_policy_metadata(value, draft, *, question, request_history=None):
    """Quarantine optional policy annotations without changing any source verdict.

    This is not an evaluator of gaps: real content requirements, omission lists,
    model coverage and missing_aspects retain their original authority. Only an
    exact, user-owned pure disclosure instruction can lose its invented task.
    """
    from copy import deepcopy
    from .answer_coverage import validate_requirement_assessments
    from .answer_gap_display import validate_request_quotes
    from .request_obligations import conditional_evidence_policy
    if not isinstance(value, dict):
        return value
    result = deepcopy(value)
    # These are host-owned diagnostics. A model cannot grant itself a waiver.
    result.pop('withheld_limitation_indices', None)
    result.pop('disclosure_policy_metadata', None)
    if not result.get('requirement_assessments'):
        return result
    validate_requirement_assessments(result, draft)
    policy_rows = [r for r in result['requirement_assessments']
                   if conditional_evidence_policy(r.get('request_quote'))]
    if not policy_rows:
        return result
    # Verify literal ownership before treating an annotation as policy. The
    # original candidate stays in raw_audit and the normalization trace.
    ownership_view = deepcopy(result)
    for row in ownership_view['requirement_assessments']:
        if conditional_evidence_policy(row.get('request_quote')):
            row['gap'] = ''
    validate_request_quotes(ownership_view, question, request_history)
    names = {r['requirement'] for r in policy_rows}
    optional = {i for r in policy_rows for i in r.get('limitation_indices', [])}
    shared = {i for r in result['requirement_assessments'] if r not in policy_rows
              for i in r.get('limitation_indices', [])}
    for row in result.get('limitation_assessments') or []:
        if row.get('requirement') in names and type(row.get('limitation_index')) is int:
            optional.add(row['limitation_index'])
    optional = {i for i in optional - shared if 0 <= i < len(draft.get('limitations', []))}
    changed = [deepcopy(r) for r in policy_rows if r.get('gap') or r.get('limitation_indices')]
    if not changed and not optional:
        return result
    for row in policy_rows:
        row.update(gap='', limitation_indices=[])
    result['withheld_limitation_indices'] = sorted(optional)
    result['disclosure_policy_metadata'] = {
        'version': 'owned-disclosure-policy-v1', 'original_rows': changed,
        'withheld_optional_limitation_indices': sorted(optional),
        'coverage_promoted': False, 'source_verdicts_changed': False}
    return result


def validate_limitation_assessments(audit, draft):
    rows = audit.get('limitation_assessments')
    if rows is None:
        return  # Historical audits keep conservative behavior.
    limitations = draft.get('limitations', [])
    if not isinstance(rows, list) or len(rows) != len(limitations):
        raise ValueError('Limitation assessments must cover each draft limitation once')
    seen = set()
    requirements = {r['requirement']: r for r in audit.get('requirement_assessments', [])}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid limitation assessment')
        index = row.get('limitation_index')
        if (type(index) is not int or not 0 <= index < len(limitations) or index in seen
                or not isinstance(row.get('reason'), str) or not row['reason'].strip()):
            raise ValueError('Limitation role requires a unique owned index and explanation')
        if row.get('role') not in {'SCOPE_BOUNDARY', 'REQUEST_GAP'}:
            raise ValueError(
                f"Invalid limitation role at limitation_index={index}: {row.get('role')!r}. "
                'Allowed limitation roles are SCOPE_BOUNDARY or REQUEST_GAP only; '
                'DIRECT, NECESSARY_CONTEXT and TANGENTIAL classify fact relevance, not limitations. '
                'Reassess this limitation; do not automatically relabel it as a boundary.'
            )
        seen.add(index)
        if index in audit.get('withheld_limitation_indices', []):
            continue  # Quarantined annotations are never displayed or used as coverage.
        if row['role'] == 'REQUEST_GAP':
            requirement = requirements.get(row.get('requirement'))
            if requirement is None or not requirement.get('gap', '').strip():
                raise ValueError('Limitation gap must bind an unresolved requested requirement')
    if any(r['role'] == 'REQUEST_GAP' and r['limitation_index'] not in
           audit.get('withheld_limitation_indices', []) for r in rows):
        audit['coverage'] = 'PARTIAL'


def retained_boundary_indices(draft, audit, *, question=None, request_history=None):
    """Only fresh classified, whole renderable caveats can answer a subpart.

    No inferred mapping, translation or promotion of an assertion to a fact.
    A trimmed uncertainty clause does not cover the original mixed statement.
    """
    from .answer_boundaries import safe_limitations
    from .answer_contracts import requires_chinese_answer
    import re
    limitations = draft.get('limitations', [])
    retained = set()
    withheld_indices = set(audit.get('withheld_limitation_indices', []))
    chinese = requires_chinese_answer(question, request_history)
    for row in audit.get('limitation_assessments') or []:
        index = row.get('limitation_index')
        if (row.get('role') != 'SCOPE_BOUNDARY' or type(index) is not int or index in withheld_indices or
                not 0 <= index < len(limitations)):
            continue
        value = limitations[index].strip()
        visible, withheld_text = safe_limitations([value], reviewed_values=[value])
        if (value in visible and not withheld_text and
                (not chinese or re.search(r'[\u4e00-\u9fff]', value))):
            retained.add(index)
    return retained


def reviewed_boundary_only(draft, audit, *, question=None, request_history=None):
    """Require an explicit fresh semantic classification, not caveat regexes."""
    rows = audit.get('limitation_assessments')
    if rows is None or not audit.get('requirement_assessments'):
        return False
    validate_limitation_assessments(audit, draft)
    retained = retained_boundary_indices(draft, audit, question=question, request_history=request_history)
    from .request_obligations import conditional_evidence_policy
    return (all(r['role'] == 'SCOPE_BOUNDARY' for r in rows)
            and not audit.get('missing_aspects') and not audit.get('omitted_answer_spans')
            and all(not r.get('gap', '').strip() and
                    (conditional_evidence_policy(r.get('request_quote')) or r.get('fact_ids')
                     or retained.intersection(r.get('limitation_indices', [])))
                    for r in audit['requirement_assessments']))
