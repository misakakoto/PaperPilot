"""Per-question evidence needs; model support proposals are never scientific proof."""
import copy

from .paper_answer_admission import admit_answer
from .paper_catalog import compare_reports
from .paper_method_identity import check_claim_identity

GAPS = {'SCOPE_GAP','CONTEXT_GAP','CONDITION_GAP','PARSER_GAP','CONFLICT','NOT_FOUND'}


def quote_spans(source):
    from .answer_contracts import evidence_spans
    return [{'quote_id':'Q'+str(n),'text':s['text'],
        'range':[source['range'][0]+s['start'],source['range'][0]+s['end']]}
        for n,s in enumerate(evidence_spans([{'evidence_id':source['source_id'],'content':source['content']}]))]


def canonical_claims(state,claims):
    if not isinstance(claims,list) or len(claims)>8:
        raise ValueError('Use at most eight source claims')
    result = []
    for claim in claims:
        if not isinstance(claim,dict):
            raise ValueError('A source claim must be an object')
        if set(claim)=={'fact','source_id','quote_id'}:
            source = state['read_handles'].get(claim['source_id'])
            spans = {s['quote_id']:s['text'] for s in quote_spans(source)} if source else {}
            if claim['quote_id'] not in spans:
                raise ValueError('Quote ID is not in this actually read source')
            result.append({'fact':claim['fact'],'source_id':claim['source_id'],'quote':spans[claim['quote_id']]})
        else:
            result.append(copy.deepcopy(claim))
    for claim in result:
        check_claim_identity(state,claim)
    return result


def canonical_answer(state,answer):
    value = copy.deepcopy(answer)
    if isinstance(value,dict) and 'claims' in value:
        value['claims'] = canonical_claims(state,value['claims'])
    return value


def requirements(state, rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 3:
        raise ValueError('Define 1..3 evidence requirements')
    normalized = []
    for n, row in enumerate(rows, 1):
        if isinstance(row,dict) and 'id' in row:
            if row['id']!='D'+str(n):
                raise ValueError('Do not change host-assigned requirement IDs')
            row = {k:v for k,v in row.items() if k!='id'}
        if (not isinstance(row, dict) or set(row) != {'dimension','kind','paper_ids'}
                or row['kind'] not in {'method','result','comparability'}
                or not isinstance(row['dimension'], str) or not 1 <= len(row['dimension']) <= 200
                or not isinstance(row['paper_ids'], list) or not 1 <= len(row['paper_ids']) <= 6
                or any(p not in state['paper_handles'] or state['paper_handles'][p] not in state['scope'] for p in row['paper_ids'])):
            raise ValueError('Each requirement needs dimension, kind and discovered paper_ids')
        normalized.append({'id': 'D'+str(n), **copy.deepcopy(row)})
    if state['requirements'] and state['requirements'] != normalized:
        old = state['requirements']
        if len(old)!=len(normalized) or any(a['dimension']!=b['dimension'] or a['kind']!=b['kind'] for a,b in zip(old,normalized)):
            raise ValueError('Requirement dimensions are frozen; do not delete gaps to declare success')
        changed = [(a,p) for a,b in zip(old,normalized) for p in a['paper_ids'] if p not in b['paper_ids']]
        if not changed or any(state['ledger'].get(a['id']+':'+p,{}).get('status')!='SCOPE_GAP' for a,p in changed):
            raise ValueError('Paper replacement requires an explicit SCOPE_GAP assessment')
        state.setdefault('requirement_revisions',[]).append(copy.deepcopy(old))
        state['ledger'] = {k:v for k,v in state['ledger'].items() if all(k!=a['id']+':'+p for a,p in changed)}
    required = set(state['required_papers'])
    if required - {state['paper_handles'][p] for r in normalized for p in r['paper_ids']}:
        raise ValueError('Requirements must cover every explicitly selected paper')
    state['requirements'] = normalized


def assess(state, proposals):
    if not isinstance(proposals, list) or len(proposals) > 18:
        raise ValueError('Use at most 18 bounded evidence assessments')
    needs = {r['id']: r for r in state['requirements']}
    for p in proposals:
        p = copy.deepcopy(p)
        if (not isinstance(p, dict) or set(p) != {'requirement','paper_id','status','reason','claims'}
                or p['requirement'] not in needs or p['paper_id'] not in needs[p['requirement']]['paper_ids']
                or p['status'] not in GAPS | {'SUPPORT_PROPOSED'}
                or not isinstance(p['reason'], str) or not 1 <= len(p['reason']) <= 600):
            raise ValueError('Assess a registered requirement/paper with a bounded reason')
        if not isinstance(p['claims'], list) or len(p['claims']) > 4:
            raise ValueError('Assessment claims must be bounded source claims')
        p['claims'] = canonical_claims(state,p['claims'])
        if p['status'] == 'SUPPORT_PROPOSED' and not p['claims']:
            raise ValueError('A support proposal requires a read source and exact quote')
        admit_answer({'answer':'', 'verdict':'ANSWERED' if p['claims'] else 'INSUFFICIENT',
            'claims': p['claims'], 'gaps': []}, list(state['read_handles'].values()), state['goal'])
        for claim in p['claims']:
            from .paper_numbers import numeric_values
            source = state['read_handles'][claim['source_id']]
            if numeric_values(claim['fact']) and source.get('source_quality',{}).get('failed_table_transcription'):
                raise ValueError('Failed table transcription cannot become a numeric fact through a text view')
            if state['read_handles'][claim['source_id']]['paper_id'] != p['paper_id']:
                raise ValueError('Claim belongs to a different paper')
        state['ledger'][p['requirement']+':'+p['paper_id']] = copy.deepcopy(p)


def coverage(state):
    rows = []
    for need in state['requirements']:
        for paper in need['paper_ids']:
            proposal = state['ledger'].get(need['id']+':'+paper)
            located = [s for s,u in state['source_handles'].items()
                       if u in state.get('unit_papers', {}) and state['unit_papers'][u] == state['paper_handles'][paper]]
            read = [r for r in state['read_handles'].values() if r['paper_id'] == paper]
            rows.append({'requirement': need['id'], 'dimension': need['dimension'], 'paper_id': paper,
                'status': proposal['status'] if proposal else 'READ' if read else 'LOCATED' if located else 'MISSING',
                'reason': proposal['reason'] if proposal else 'No model support assessment',
                'assurance': 'MODEL_PROPOSAL_NOT_INDEPENDENT_REVIEW'})
    return rows


def ready(state):
    rows = coverage(state)
    return bool(rows) and all(r['status'] == 'SUPPORT_PROPOSED' for r in rows)


def admit_final(state, answer):
    for claim in answer.get('claims',[]):
        check_claim_identity(state,claim)
    result = admit_answer(answer, list(state['read_handles'].values()), state['goal'])
    from .paper_numbers import numeric_values
    if any(numeric_values(c['fact']) and state['read_handles'][c['source_id']].get('source_quality',{}).get('failed_table_transcription')
            for c in answer['claims']):
        raise ValueError('Failed table transcription cannot be laundered through a text view')
    cited = {state['read_handles'][c['source_id']]['paper_sha256'] for c in answer['claims']}
    if answer['verdict'] == 'ANSWERED':
        if set(state['required_papers']) - cited:
            raise ValueError('Answer omitted an explicitly selected paper')
        if state['mode'] == 'active' and not ready(state):
            raise ValueError('Resolve or explicitly report evidence gaps; read coverage alone is not answerability')
    # Preserve the declared unknown experimental conditions; no online judge
    # claims to establish fairness from token overlap.
    return {**result, 'comparison': compare_reports([]),
            'scientific_correctness': 'REQUIRES_INDEPENDENT_CONTENT_REVIEW'}
