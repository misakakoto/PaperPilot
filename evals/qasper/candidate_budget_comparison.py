"""Frozen candidate-budget comparison: same original question, gold and final top-8."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

from evals.qasper.agent_case import paper_records
from evals.qasper.retrieval_diagnostic import text_metrics
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.retrieval import FastEmbedder
from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
from paperpilot_retrieval.window_reranking import WindowReranker


class CachedRanker:
    def __init__(self, inner):
        self.inner, self.cache = inner, {}
        self.inputs = 0
        self.seconds = 0.0
        self.name = inner.name
    def score(self, query, passages):
        missing = list(dict.fromkeys(p for p in passages if (query,p) not in self.cache))
        if missing:
            started = time.perf_counter()
            scores = self.inner.score(query, missing)
            self.seconds += time.perf_counter()-started
            self.inputs += len(missing)
            self.cache.update({(query,p):s for p,s in zip(missing,scores,strict=True)})
        return [self.cache[(query,p)] for p in passages]


class ProbeAdapter(PaperPilotKnowledgeSearchAdapter):
    def _diversify(self, items, **kwargs):
        self.candidate_ids = [i['stable_id'] for i in items]
        return super()._diversify(items, **kwargs)


def run(args):
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    excluded = set()
    for path in ([args.exclude_manifest] if args.exclude_manifest else []) + args.exclude_manifests:
        excluded.update(json.loads(path.read_text(encoding='utf-8'))['papers'])
    ids = sorted(set(data)-excluded, key=lambda p: hashlib.sha256(('retrieval-v2:'+p).encode()).hexdigest())[:args.papers]
    assert ids and not (set(ids)&excluded)
    arms = args.arms.split(',')
    assert set(arms) <= {'legacy','expanded','windows','relevance','lexical'}
    args.output.mkdir(parents=True, exist_ok=False)
    embedder = FastEmbedder(cache_dir=args.embedding_cache, allow_download=False)
    inner = OnnxCrossEncoderReranker(cache_dir=args.reranker_cache, allow_download=False)
    embedder.embed_query('warmup')
    inner.warmup()
    root = Path(__file__).resolve().parents[2]
    sources = ['src/paperpilot_agentic/adapters.py','src/paperpilot_retrieval/window_reranking.py',
        'src/paperpilot_retrieval/reranking.py','src/paperpilot_retrieval/store.py',
        'evals/qasper/agent_case.py','evals/qasper/retrieval_diagnostic.py','evals/qasper/candidate_budget_comparison.py']
    manifest = dict(dataset_sha256=hashlib.sha256(raw).hexdigest(), papers=ids, arms=arms,
        excluded_papers=sorted(excluded), top_k=8, expanded_candidate_k=64,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        model=dict(embedding=embedder.name, embedding_sha=embedder.model_sha256,
                   reranker=inner.name, reranker_sha=inner.model_sha256, revision=inner.resolved_revision),
        question_ids=[q['question_id'] for p in ids for q in data[p]['qas']], provider_requests=0,
        scoring='unchanged exact paragraph best-annotator macro recall; errors zero, no-text separate',
        score_cache='same query/passage/model scores reused across arms, never gold-dependent',
        scope='known-paper public text; not full benchmark or answer accuracy')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows = []
    with (args.output/'rankings.jsonl').open('x',encoding='utf-8') as out:
        for pi, pid in enumerate(ids):
            paper = data[pid]
            _,doc,chunks = paper_records({k:v for k,v in paper.items() if k!='qas'}, source_id=pid)
            with tempfile.TemporaryDirectory(prefix='paperpilot-candidates-') as tmp:
                store = PaperKnowledgeStore(Path(tmp)/'index.sqlite3')
                store.sync_documents([doc],chunks,embedder=embedder)
                for qa in paper['qas']:
                    cached = CachedRanker(inner)
                    row = dict(paper_id=pid, question_id=qa['question_id'], question=qa['question'])
                    for arm in arms:
                        ranker = WindowReranker(cached) if arm=='windows' else cached
                        adapter = ProbeAdapter(store, embedder=embedder if arm!='lexical' else None,
                            reranker=ranker if arm!='lexical' else None, rerank_stage='final',
                            scoped_candidate_k=64 if arm in {'expanded','windows','relevance'} else None,
                            scoped_diversity=0 if arm=='relevance' else 0.22)
                        started, inputs_before = time.perf_counter(), cached.inputs
                        try:
                            hits = adapter.search(qa['question'],top_k=8,paper_sha256s={doc.sha256})
                            keys = [h['stable_id'] for h in hits]
                            candidates = adapter.candidate_ids
                            original = {c.stable_id:c.text for c in store.get_chunks(list(dict.fromkeys(keys+candidates)))}
                            texts = [original[k] for k in keys]
                            candidate_texts = [original[k] for k in candidates]
                            error = None
                        except Exception as exc:
                            keys, texts, candidates, candidate_texts = [], [], [], []
                            error = type(exc).__name__+': '+str(exc)[:150]
                        row[arm] = dict(ranks=keys, candidates=candidates, error=error,
                            metrics=text_metrics(qa,texts,[c.text for c in chunks]),
                            candidate_coverage=text_metrics(qa,[],candidate_texts)['best_corpus_text_coverage'],
                            extra_scored_inputs=cached.inputs-inputs_before, seconds=time.perf_counter()-started)
                    out.write(json.dumps(row,ensure_ascii=False)+'\n')
                    out.flush()
                    rows.append(row)
            print(f'Finished {pi+1}/{len(ids)} papers',flush=True)
    eligible = [r for r in rows if r[arms[0]]['metrics']['best_text_recall'] is not None]
    def value(r,a,k):
        return r[a]['metrics']['best_text_recall'][str(k)]
    summary = dict(papers=len(ids), queries=len(rows), text_denominator=len(eligible),
        recall={a:{str(k):sum(value(r,a,k) for r in eligible)/len(eligible) for k in (1,3,8)} for a in arms},
        coverage={a:sum(r[a]['candidate_coverage'] for r in eligible)/len(eligible) for a in arms},
        errors={a:sum(r[a]['error'] is not None for r in rows) for a in arms},
        extra_scored_inputs={a:sum(r[a]['extra_scored_inputs'] for r in rows) for a in arms},
        mean_candidates={a:sum(len(r[a]['candidates']) for r in rows)/len(rows) for a in arms}, delta95={})
    for a in arms:
        if a == 'legacy':
            continue
        groups = defaultdict(list)
        for r in eligible:
            groups[r['paper_id']].append(value(r,a,8)-value(r,'legacy',8))
        rng, draws = random.Random(20260913), []
        for _ in range(2000):
            values = [d for p in rng.choices(list(groups),k=len(groups)) for d in groups[p]]
            draws.append(sum(values)/len(values))
        draws.sort()
        summary['delta95'][a] = [draws[50],draws[1949]]
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','output','embedding-cache','reranker-cache'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exclude-manifest',type=Path)
    p.add_argument('--exclude-manifests',type=Path,nargs='*',default=[])
    p.add_argument('--papers',type=int,default=40)
    p.add_argument('--arms',default='legacy,expanded,windows,relevance')
    print(json.dumps(run(p.parse_args()),indent=2))
