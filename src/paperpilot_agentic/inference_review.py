"""Expose the candidate's premise links without granting entailment or approval."""
import re

INSTRUCTION = (
    'Audit SOURCE_FACT attribution and INFERENCE derivation as different questions. '
    'For each inference_review binding, first independently verify every premise '
    'against its own source, then check whether the stated inference follows. '
    'A derived comparison does not need to appear verbatim in either paper; the '
    'authors need not have compared each other. Lack of an author-written comparison '
    'alone is not grounds for UNSUPPORTED or TANGENTIAL. This never grants support: '
    'reject missing/unsupported premises, unlicensed logical steps, stronger '
    'qualifiers, invented purposes, causal effects, or measured superiority inferred '
    'only from a mechanism. Keep whole-paper absence claims subject to source-scope '
    'checks. Do not rewrite a rejected claim into a correct one inside the verdict.'
)


def review_context(draft):
    bindings = [{'fact_id': fact['id'], 'premise_fact_ids': list(fact.get('depends_on', []))}
                for fact in draft['facts'] if fact.get('kind') == 'INFERENCE']
    if not bindings:
        return None
    return {'policy': 'independent-premises-then-derivation-v1',
            'premise_bindings': bindings, 'author_comparison_required': False,
            'premises_require_independent_support': True,
            'host_approval': False}


def validate_disputed_claims(audit, draft):
    """Bind a rejected inference to words actually present in the current draft.

    This proves only review identity, not the verdict. A valid quotation may
    still receive a wrong semantic verdict and must never imply host approval.
    """
    facts = {f['id']: f for f in draft['facts']}
    for row in audit.get('assessments', []):
        fact = facts.get(row.get('fact_id'), {})
        if fact.get('kind') != 'INFERENCE' or row.get('verdict') == 'SUPPORTED':
            continue
        quote = row.get('disputed_quote')
        if not isinstance(quote, str) or not quote.strip() or quote not in fact.get('text', ''):
            raise ValueError('Rejected inference requires disputed_quote copied exactly from CURRENT '
                             'fact text: fact_id=' + str(row.get('fact_id')) +
                             '. Do not assess a removed claim or invent a negative assertion. '
                             'Recheck the supplied current fact and its independently verified '
                             'premises; a positive operation comparison need not prove the other '
                             'method fails. Keep rejection if a quoted current clause is unsupported.')


def validate_derivation_basis(audit, draft):
    """A reviewer must test derivation, rather than demand an author comparison.

    Invalid review criteria consume the existing bounded audit-repair slot.
    No verdict is changed here; factual, causal and scope checks still apply.
    """
    facts = {f['id']: f for f in draft['facts']}
    rows = audit.get('assessments')
    if not isinstance(rows, list):
        return
    criterion = re.compile(
        r'(?:sources?|papers?|authors?|text).{0,90}(?:not|lack).{0,50}'
        r'(?:explicitly\s+(?:rank|compare)|define.{0,30}directness.{0,30}metric)|'
        r'(?:not|never).{0,30}(?:stated\s+verbatim|author-written\s+comparison)|'
        r'(?:原文|论文|作者).{0,25}未(?:直接|明确)(?:比较|对比|排序)', re.I)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (facts.get(row.get('fact_id'), {}).get('kind') == 'INFERENCE'
                and row.get('verdict') in {'UNSUPPORTED', 'UNCERTAIN'}
                and criterion.search(str(row.get('reason', '')))):
            raise ValueError('Invalid inference review criterion at fact_id=' + str(row['fact_id']) +
                             ': an author-written ranking/comparison or a metric named directness '
                             'is not required for a mechanism-level derivation. Reassess the CURRENT '
                             'claim against its cited premises. If it is still unsupported, quote '
                             'the actual unsupported clause and identify the unverified premise, '
                             'logical step, condition, causal claim or stronger scope. Do not '
                             'automatically approve it or require the authors to state the inference.')
