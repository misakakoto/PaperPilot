"""Same-query, same-eight-slots retrieval comparison; public text, no chat API."""
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


def run(args):
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    ids = sorted(data, key=lambda p: hashlib.sha256(('retrieval-v2:' + p).encode()).hexdigest())
    if args.papers:
        ids = ids[:args.papers]
    args.output.mkdir(parents=True, exist_ok=False)
    embedder = FastEmbedder(cache_dir=args.embedding_cache, allow_download=False)
    reranker = OnnxCrossEncoderReranker(cache_dir=args.reranker_cache, allow_download=False)
    embedder.embed_query('warmup')
    reranker.warmup()
    arms = ['current', 'bm25', 'dense', 'hybrid', 'hybrid_pool', 'hybrid_final']
    root = Path(__file__).resolve().parents[2]
    paths = ['src/paperpilot_agentic/adapters.py', 'src/paperpilot_retrieval/store.py',
             'src/paperpilot_retrieval/reranking.py', 'evals/qasper/strong_retrieval_comparison.py']
    manifest = dict(dataset_sha256=hashlib.sha256(raw).hexdigest(), papers=ids, arms=arms,
        source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        top_k=8, provider_requests=0, known_paper=True, official_score=False,
        selection='paper SHA order, never score-based; train/dev used to select, test confirm once',
        embedder=dict(name=embedder.name, revision=embedder.resolved_revision, sha=embedder.model_sha256),
        reranker=dict(name=reranker.name, revision=reranker.resolved_revision, sha=reranker.model_sha256))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    with (args.output/'rankings.jsonl').open('x', encoding='utf-8') as out:
        for index, pid in enumerate(ids):
            paper = data[pid]
            _, doc, chunks = paper_records({k:v for k,v in paper.items() if k != 'qas'}, source_id=pid)
            with tempfile.TemporaryDirectory(prefix='paperpilot-strong-') as tmp:
                store = PaperKnowledgeStore(Path(tmp)/'index.sqlite3')
                store.sync_documents([doc], chunks, embedder=embedder)
                raw_store = PaperKnowledgeStore(Path(tmp)/'index.sqlite3', lexical_query_policy='raw')
                adapters = {'current': PaperPilotKnowledgeSearchAdapter(store),
                    'hybrid_pool': PaperPilotKnowledgeSearchAdapter(store, embedder=embedder),
                    'hybrid_final': PaperPilotKnowledgeSearchAdapter(store, embedder=embedder,
                         reranker=reranker, rerank_stage='final')}
                for qa in paper['qas']:
                    row = dict(paper_id=pid, question_id=qa['question_id'], question=qa['question'])
                    for name in arms:
                        started = time.perf_counter()
                        try:
                            if name in adapters:
                                hits = adapters[name].search(qa['question'], top_k=8, paper_sha256s={doc.sha256})
                                keys = [h['stable_id'] for h in hits]
                            else:
                                hits = (raw_store if name == 'bm25' else store).search_chunks(qa['question'], top_k=8,
                                    embedder=embedder if name != 'bm25' else None,
                                    mode=name, paper_sha256s={doc.sha256})
                                keys = [h.stable_id for h in hits]
                            byid = {c.stable_id:c.text for c in store.get_chunks(keys)}
                            texts, error = [byid[k] for k in keys], None
                        except Exception as exc:
                            keys, texts, error = [], [], type(exc).__name__ + ': ' + str(exc)[:200]
                        row[name] = dict(ranks=keys, metrics=text_metrics(qa, texts, [c.text for c in chunks]),
                            seconds=time.perf_counter()-started, error=error)
                    out.write(json.dumps(row, ensure_ascii=False)+'\n')
                    out.flush()
                    rows.append(row)
            print(f'Finished {index+1}/{len(ids)} papers', flush=True)
    eligible = [r for r in rows if r['current']['metrics']['best_text_recall'] is not None]
    def score(r, name, k):
        return r[name]['metrics']['best_text_recall'][str(k)]
    summary = dict(papers=len(ids), queries=len(rows), text_denominator=len(eligible),
        recall={name:{str(k):sum(score(r,name,k) for r in eligible)/len(eligible) for k in (1,3,8)} for name in arms},
        errors={name:sum(r[name]['error'] is not None for r in rows) for name in arms})
    groups = defaultdict(list)
    for r in eligible:
        groups[r['paper_id']].append(score(r,'hybrid_final',8)-score(r,'current',8))
    rng, draws = random.Random(20260913), []
    for _ in range(2000):
        values = [v for pid in rng.choices(list(groups), k=len(groups)) for v in groups[pid]]
        draws.append(sum(values)/len(values))
    draws.sort()
    summary['delta95ci'] = [draws[50],draws[1949]]
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--embedding-cache',type=Path,required=True)
    p.add_argument('--reranker-cache',type=Path,required=True)
    p.add_argument('--papers',type=int,default=0)
    print(json.dumps(run(p.parse_args()),indent=2))
