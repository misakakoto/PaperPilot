"""User-visible gaps refer to user requests; raw audit diagnostics stay internal.

No translation or scientific claim is synthesized here. Missing source evidence,
semantic review and publication coverage retain their original conservative state.
"""
from copy import deepcopy
import re

from .answer_boundaries import safe_limitations

VERSION = 'request-bound-gap-display-v3-active-constraints-and-conditional-policy'


def request_texts(question, request_history=None):
    return [question or ''] + [row.get('user_request', '') for row in request_history or []]


def request_quote_units(question, request_history=None):
    """Owned, literal display anchors; never a semantic decomposition or evidence.

    The full requests remain in the messages. Punctuation and bounded slices only
    provide stable choices so a model need not reproduce a quote character by
    character. A selected anchor does not prove the associated requirement true.
    """
    units = []
    for index, text in enumerate(request_texts(question, request_history)):
        for match in re.finditer(r'[^，。；！？\n]+[，。；！？]?', text):
            for offset in range(match.start(), match.end(), 256):
                quote = text[offset:min(offset + 256, match.end())].strip()
                from .request_obligations import superseded_numeric_quote
                if quote and not (index > 0 and superseded_numeric_quote(quote, question)):
                    units.append({'request_ref': f'request_{index}_{offset}', 'text': quote})
    return units


def resolve_request_refs(audit, question, request_history=None):
    """Project exact host text for explicit selections, preserving the raw audit."""
    result = deepcopy(audit)
    units = {r['request_ref']: r['text'] for r in request_quote_units(question, request_history)}
    bindings = []
    for row in result.get('requirement_assessments', []):
        ref = row.get('request_ref')
        if ref is None:
            continue  # Legacy literal quotes retain their strict validation.
        if not isinstance(ref, str) or ref not in units:
            raise ValueError('Requirement request_ref must belong to an original user request')
        if row.get('request_quote') is not None and row['request_quote'] != units[ref]:
            raise ValueError('Requirement request_quote conflicts with the selected original user request')
        row['request_quote'] = units[ref]
        bindings.append({'request_ref': ref, 'request_quote': units[ref]})
    return result, bindings


def validate_request_quotes(audit, question, request_history=None):
    if question is None:
        return  # Historical callers did not supply a request identity.
    requests = request_texts(question, request_history)
    units = {r['request_ref']: r['text'] for r in request_quote_units(question, request_history)}
    for row in audit.get('requirement_assessments', []):
        ref = row.get('request_ref')
        if ref is not None and (not isinstance(ref, str) or ref not in units or
                row.get('request_quote') is not None and row['request_quote'] != units[ref]):
            raise ValueError('Requirement request_ref must belong to an original user request')
        quote = row.get('request_quote')
        from .request_obligations import conditional_evidence_policy, superseded_numeric_quote
        if quote is not None and superseded_numeric_quote(quote, question):
            raise ValueError('Requirement quote is superseded by the current mechanism-only request')
        if conditional_evidence_policy(quote):
            gap = row.get('gap', '')
            if not isinstance(gap, str):
                raise ValueError('Conditional evidence policy gap must be a text item')
            if gap.strip():
                raise ValueError('A conditional evidence policy is not a missing scientific subpart; '
                                 'bind any real evidence gap to the actual requested content')
        if quote is not None and (not isinstance(quote, str) or not quote.strip()
                or len(quote) > 256 or not any(quote in text for text in requests)):
            raise ValueError('Requirement request_quote must belong to an original user request')


def user_visible_gaps(draft, audit, *, question, request_history, chinese, host_presentation=None):
    """Project gaps to owned request quotes, never model-generated criticism."""
    requests = request_texts(question, request_history)
    from .answer_completion import retained_boundary_indices
    retained_boundaries = retained_boundary_indices(draft, audit, question=question, request_history=request_history)
    from .request_obligations import conditional_evidence_policy
    from .answer_layout import satisfied_layout_quote
    diagnostics = {
        'version': VERSION,
        'missing_aspects': deepcopy(audit.get('missing_aspects', [])),
        'request_gaps': deepcopy([row for row in audit.get('requirement_assessments', [])
                                 if row.get('gap', '').strip() or
                                 (not conditional_evidence_policy(row.get('request_quote'))
                                  and not satisfied_layout_quote(row.get('request_quote'), question,
                                                                host_presentation, draft, audit)
                                  and not row.get('fact_ids') and not retained_boundaries.intersection(
                                     row.get('limitation_indices', [])))]),
        'draft_request_gaps': [], 'unrendered_scope_boundaries': [],
    }
    notices = []
    for row in diagnostics['request_gaps']:
        quote = row.get('request_quote')
        if (isinstance(quote, str) and quote.strip() and len(quote) <= 256
                and any(quote in text for text in requests)
                and (not chinese or re.search(r'[一-鿿]', quote))):
            notices.append(('本轮未完成：' if chinese else 'Not completed in this answer: ') +
                           quote.rstrip('。.!！?？') + ('。' if chinese else '.'))
    roles = {row['limitation_index']: row['role']
             for row in audit.get('limitation_assessments') or []}
    withheld_indices = set(audit.get('withheld_limitation_indices', []))
    diagnostics['unreviewed_limitations'] = []
    boundaries = []
    reviewed_boundaries = []
    for index, value in enumerate(draft.get('limitations', [])):
        if index in withheld_indices:
            diagnostics['unreviewed_limitations'].append(value)
        elif roles.get(index) == 'REQUEST_GAP':
            diagnostics['draft_request_gaps'].append(value)
        else:
            boundaries.append(value)
            if roles.get(index) == 'SCOPE_BOUNDARY':
                reviewed_boundaries.append(value)
    visible, withheld = safe_limitations(list(dict.fromkeys(boundaries)),
                                        reviewed_values=reviewed_boundaries)
    if chinese:
        wrong_language = [value for value in visible if not re.search(r'[一-鿿]', value)]
        if wrong_language:
            visible = [value for value in visible if value not in wrong_language]
            diagnostics['unrendered_scope_boundaries'] = wrong_language
            withheld = list(dict.fromkeys(withheld + wrong_language))
            visible.append('结论边界的说明尚未按中文要求完成，仍需复核。')
    if (diagnostics['missing_aspects'] or diagnostics['request_gaps'] or
            diagnostics['draft_request_gaps']) and not notices:
        notices.append('本轮未完成原问题的全部要求；已审核内容保留，未完成项仍需补充复核。' if chinese else
                       'Not all requested work is complete; reviewed content is retained and gaps still need review.')
    return list(dict.fromkeys(visible + notices)), withheld, diagnostics
