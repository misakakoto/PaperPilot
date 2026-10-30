"""Answer source admission shared by evaluations; not scientific entailment."""
from .paper_numbers import numeric_values


def admit_answer(answer, sources, question=''):
    if not isinstance(answer, dict) or not {'answer', 'verdict', 'claims', 'gaps'} <= answer.keys():
        raise ValueError('Missing answer/verdict/claims/gaps')
    if answer['verdict'] not in {'ANSWERED', 'INSUFFICIENT', 'NOT_COMPARABLE'}:
        raise ValueError('No automated winner is admitted without independently matching protocols')
    by_id = {r['source_id']: r for r in sources}
    if len(by_id) != len(sources):
        raise ValueError('Source IDs must be unique')
    if not isinstance(answer['claims'], list) or len(answer['claims']) > 8:
        raise ValueError('Use at most eight source-bearing claims')
    if answer['verdict'] == 'ANSWERED' and not answer['claims']:
        raise ValueError('An answered factual question requires source-bearing claims')
    if not isinstance(answer['gaps'], list) or any(not isinstance(g, str) for g in answer['gaps']):
        raise ValueError('Gaps must be explicit text entries')
    cited_numbers = set()
    for claim in answer['claims']:
        if not isinstance(claim, dict) or set(claim) != {'fact', 'source_id', 'quote'}:
            raise ValueError('Each claim needs fact/source_id/quote')
        source = by_id.get(claim['source_id'])
        if (source is None or not isinstance(claim['quote'], str) or not claim['quote'].strip()
                or claim['quote'] not in source['content']):
            raise ValueError('Citation quote absent from the supplied original source')
        numbers, supported = numeric_values(claim['fact']), numeric_values(claim['quote'])
        if numbers - supported:
            raise ValueError('Numeric claim absent from its cited quote')
        if numbers and source['source_kind'] == 'TABLE' and source.get('source_quality',{}).get('table_structure') != 'GRID_VALIDATED_NOT_FACT_VERIFIED':
            raise ValueError('Failed table transcription is not numeric evidence')
        cited_numbers |= supported
    # A refusal may refer to a requested but unavailable year/CI. That is not a result.
    if answer['verdict'] == 'INSUFFICIENT':
        cited_numbers |= numeric_values(question)
    if numeric_values(answer['answer']) - cited_numbers:
        raise ValueError('Answer summary contains an unsupported numeric value')
    return {'admitted': True, 'origin_check': 'LITERAL_ONLY_NOT_ENTAILMENT', 'claims': len(answer['claims']),
            'manual_scientific_review': 'REQUIRED', 'unqualified_winner_review': 'REQUIRED'}
