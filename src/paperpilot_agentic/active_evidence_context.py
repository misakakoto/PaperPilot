"""Pack exact, source-bound quotes fairly into the final answer view.

Retrieval and reading are unchanged. This selects original quote spans, never
summarizes them, and records every omission. The shared compiler still checks
the complete request and output reserve against the task's cumulative budget.
"""
import copy
import re

from .active_evidence_ledger import quote_spans, ready
from .paper_evidence import query_terms, upper


def final_evidence_view(payload, state, ids, cap=6700):
    payload = copy.deepcopy(payload)
    needed = {p for r in state['requirements'] for p in r['paper_ids']}
    needed |= {r['paper_id'] for r in state['read_handles'].values()}
    payload['omitted_catalogue_papers'] = [p['paper_id'] for p in payload['papers'] if p['paper_id'] not in needed]
    payload['papers'] = [{**p,'method_identity_hints':[{'name':h['name'],'unit_id':h['unit_id']}
        for h in p['method_identity_hints']]} for p in payload['papers'] if p['paper_id'] in needed]
    # Dimensions already live in requirements; counts are durable navigation.
    payload['evidence_coverage'] = [{k:v for k,v in row.items() if k not in {'dimension'}}
        for row in payload.get('evidence_coverage',[])]
    payload['read_passages'] = []
    payload['allowed_verdicts'] = (['ANSWERED','INSUFFICIENT','NOT_COMPARABLE'] if ready(state)
        else ['INSUFFICIENT','NOT_COMPARABLE'])
    payload['read_receipts'] = 'Exact selected quotes only; omitted quotes and full originals remain in receipts.'
    payload['evidence_sources'] = {}
    payload['omitted_read_handles'] = list(ids)
    payload['quote_selection'] = {'policy':'exact-quotes-balanced-by-paper-and-range',
        'omitted_quote_count':0,'view_cap':cap}
    preferred = {(c['source_id'],c['quote']) for p in state['ledger'].values() for c in p['claims']}
    query = query_terms(state['goal'])+' '+' '.join(r['dimension'] for r in state['requirements'])
    for zh,en in (('分辨率','resolution resolutions'),('模态','modality modalities'),('池化','pooling'),('回写','decoder')):
        if zh in query:
            query += ' '+en
    terms = set(re.findall(r'[a-z][a-z0-9_-]{2,}',query.casefold()))-{'the','and','with','for','how','are','from','what'}
    def score(rid,span):
        words=set(re.findall(r'[a-z][a-z0-9_-]{2,}',span['text'].casefold()))
        return (1000 if (rid,span['text']) in preferred else 0)+len(words & terms)
    spans={rid:sorted(quote_spans(state['read_handles'][rid]),key=lambda q:-score(rid,q)) for rid in ids}
    # Give every paper/range a chance before filling one long source. Rank within
    # each paper by useful terms, retaining stable IDs and the exact source text.
    groups={}
    for rid in ids:
        groups.setdefault(state['read_handles'][rid]['paper_id'],[]).append(rid)
    for group in groups.values():
        group.sort(key=lambda rid:-max((score(rid,q) for q in spans[rid]),default=0))
    order=[g[n] for n in range(max((len(g) for g in groups.values()),default=0)) for g in groups.values() if n<len(g)]
    total=sum(map(len,spans.values()))
    selected=0
    for n in range(max((len(q) for q in spans.values()),default=0)):
        for rid in order:
            if n>=len(spans[rid]):
                continue
            trial=copy.deepcopy(payload)
            row=next((r for r in trial['read_passages'] if r['source_id']==rid),None)
            source=state['read_handles'][rid]
            if row is None:
                row={k:source[k] for k in ('source_id','paper_id','page','range','source_kind','source_quality')}
                row.update(quote_spans=[],view='SELECTED_EXACT_QUOTES_NOT_FULL_RANGE')
                trial['read_passages'].append(row)
                trial['omitted_read_handles'].remove(rid)
                trial['evidence_sources'][rid]={'range_identity':source['range_identity'],'index_identity':state['retrieval_identity']}
            row['quote_spans'].append(spans[rid][n])
            trial['quote_selection']['omitted_quote_count']=total-selected-1
            if upper(trial)<=cap:
                payload=trial
                selected+=1
    payload['quote_selection']['omitted_quote_count']=total-selected
    return payload
