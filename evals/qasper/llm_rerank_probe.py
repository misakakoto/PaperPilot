"""Evidence-only LLM ranking from frozen candidate IDs. Gold used only after inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from collections import defaultdict

from evals.qasper.agent_case import paper_records
from evals.qasper.retrieval_diagnostic import text_metrics
from paperpilot_retrieval.store import stable_chunk_id
from paperpilot_agentic.config import RuntimeSettings, read_api_key_file
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.evidence_reranking import rerank_evidence, PROMPT
from paperpilot_agentic.evidence_selection import select_evidence, PROMPT as SELECT_PROMPT


def run(args):
    v2 = args.protocol == 'select-v2'
    rank_function = select_evidence if v2 else rerank_evidence
    prompt = SELECT_PROMPT if v2 else PROMPT
    raw=args.dataset.read_bytes()
    data=json.loads(raw)
    source=json.loads((args.candidates/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==source['dataset_sha256']
    records=[json.loads(l) for l in (args.candidates/'rankings.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==len(source['question_ids']), 'candidate collection must be complete'
    records.sort(key=lambda r:hashlib.sha256(('llm-rerank-v1:'+r['question_id']).encode()).hexdigest())
    if args.limit:
        records=records[:args.limit]
    args.output.mkdir(parents=True,exist_ok=False)
    settings=RuntimeSettings('deepseek',read_api_key_file(args.key_file),'https://api.deepseek.com',
        'deepseek-v4-flash',False,0.0,512,90,args.output,args.output/'unused.sqlite3',None,'probe',())
    manifest=dict(question_ids=[r['question_id'] for r in records], papers=source['papers'],
        dataset_sha256=source['dataset_sha256'], top_k=8, candidate_k=24,
        max_provider_attempts=len(records), model=settings.model, retries=0, prompt=prompt,
        protocol=args.protocol, selection_limit=args.limit, max_output_tokens=256 if v2 else 512,
        error_policy='explicit legacy fallback; all model errors reported' if v2 else 'empty ranking, score zero',
        candidate_manifest_sha256=hashlib.sha256((args.candidates/'manifest.json').read_bytes()).hexdigest(),
        ranking_source_sha256=hashlib.sha256(Path(rank_function.__code__.co_filename).read_bytes()).hexdigest(),
        selection='SHA question order, no label/outcome filter; limit zero means whole cohort',
        input='question and public original passages only; no gold; 3500 chars per passage; shuffled IDs')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def execute(record):
        pid,qid=record['paper_id'],record['question_id']
        paper=data[pid]
        qa=next(q for q in paper['qas'] if q['question_id']==qid)
        assert qa['question']==record['question']
        _,doc,chunks=paper_records({k:v for k,v in paper.items() if k!='qas'},source_id=pid)
        by_id={stable_chunk_id(doc.sha256,c):c for c in chunks}
        ids=record['expanded']['candidates'][:24]
        candidates=[{'stable_id':i,'content':f'{doc.title}\n{by_id[i].section}\n{by_id[i].text}'.strip(),
                     'paper_sha256':doc.sha256} for i in ids]
        row=dict(paper_id=pid,question_id=qid,question=qa['question'],attempts=0, baseline=record['legacy'])
        inner=OpenAICompatibleLLM(settings,max_retries=0)
        class Recorder:
            def complete(self,messages,**kwargs):
                def reserve():
                    row['attempts']+=1
                    (args.output/(qid+'-request.json')).write_text(json.dumps({'messages':messages,**kwargs},ensure_ascii=False),encoding='utf-8')
                response=inner.complete(messages,before_attempt=reserve,**kwargs)
                (args.output/(qid+'-response.json')).write_text(json.dumps(asdict(response),ensure_ascii=False),encoding='utf-8')
                row['tokens']=response.total_tokens
                return response
        try:
            ranked=rank_function(qa['question'],candidates,Recorder())
            row.update(ranks=[i['stable_id'] for i in ranked['hits']], error=None)
        except Exception as exc:
            row.update(ranks=list(record['legacy']['ranks']) if v2 else [],
                       error=type(exc).__name__, degraded=v2)
        # References are first consulted here, never by the provider or ranking function.
        row['metrics']=text_metrics(qa,[by_id[i].text for i in row['ranks']],[c.text for c in chunks])
        (args.output/(qid+'-result.json')).write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
        print(f'Finished {qid}: {row["error"] or "ok"}',flush=True)
        return row
    first=execute(records[0])
    if first['error'] in {'APIConnectionError','AuthenticationError','PermissionDeniedError','NotFoundError'}:
        raise RuntimeError('Provider preflight failed; batch stopped')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[first,*pool.map(execute,records[1:])]
    eligible=[r for r in rows if r['metrics']['best_text_recall'] is not None]
    def score(r,name,k):
        m=r['metrics'] if name=='llm' else r['baseline']['metrics']
        return m['best_text_recall'][str(k)]
    groups=defaultdict(list)
    for r in eligible:
        groups[r['paper_id']].append(score(r,'llm',8)-score(r,'baseline',8))
    rng,draws=random.Random(20260913),[]
    for _ in range(2000):
        vals=[v for p in rng.choices(list(groups),k=len(groups)) for v in groups[p]]
        draws.append(sum(vals)/len(vals))
    draws.sort()
    summary=dict(queries=len(rows),text_denominator=len(eligible), provider_attempts=sum(r['attempts'] for r in rows),
        errors=sum(r['error'] is not None for r in rows), tokens=sum(r.get('tokens',0) for r in rows),
        fallbacks=sum(bool(r.get('degraded')) for r in rows),
        final_errors=sum(r['error'] is not None and (not v2 or r['baseline']['error'] is not None) for r in rows),
        recall={n:{str(k):sum(score(r,n,k) for r in eligible)/len(eligible) for k in (1,3,8)} for n in ('baseline','llm')},
        delta95=[draws[50],draws[1949]])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','candidates','output','key-file'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--protocol',choices=('rank-v1','select-v2'),default='rank-v1')
    print(json.dumps(run(p.parse_args()),indent=2))
