"""Frozen paired lexical-query ablation; no model, gold in retrieval, or live DB."""
import argparse
from collections import Counter
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
from paperpilot_retrieval.lexical_query import FUNCTION_WORDS, content_query

class ContentQueryStore(PaperKnowledgeStore):
    def __init__(self, path, *, namespace='default'):
        super().__init__(path, namespace=namespace, lexical_query_policy='content_v1')


def run(dataset_path, output):
    dataset_path, output = Path(dataset_path), Path(output)
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)
    questions = [(pid, q['question_id']) for pid, p in dataset.items() for q in p['qas']]
    if len(questions) != len(set(questions)):
        raise ValueError('Duplicate paper/question IDs')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'evals/qasper/agent_case.py', root/'evals/qasper/retrieval_diagnostic.py',
             root/'src/paperpilot_agentic/adapters.py', root/'src/paperpilot_retrieval/store.py',
             root/'src/paperpilot_retrieval/lexical_query.py']
    manifest = {'protocol': 'lexical-function-word-ablation-v1', 'dataset': dataset_path.name,
        'dataset_sha256': hashlib.sha256(raw).hexdigest(),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'function_words': sorted(FUNCTION_WORDS), 'top_k': 8, 'primary_metric': 'macro_best_text_recall@8',
        'question_ids': questions, 'provider_requests': 0,
        'controlled': 'same index, query, paper scope, candidate limits, fusion and diversity; lexical query only',
        'errors': 'retain zero in eligible denominator', 'held_out_claim': False,
        'selection_rule': 'Do not deploy unless paired paper-bootstrap primary delta lower 95% bound > 0; '
                          'validate chosen method on a separate frozen split before resume claims.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    scores, timing = [], Counter()
    with (output/'rankings.jsonl').open('x', encoding='utf-8') as stream:
        for paper_index, (pid, paper) in enumerate(dataset.items()):
            public = {k: v for k, v in paper.items() if k != 'qas'}
            _, document, chunks = paper_records(public, source_id=pid)
            with tempfile.TemporaryDirectory(prefix='paperpilot-query-ablation-') as temporary:
                path = Path(temporary)/'index.sqlite3'
                baseline = PaperKnowledgeStore(path, namespace='paired', lexical_query_policy='raw')
                baseline.sync_documents([document], chunks)
                variant = ContentQueryStore(path, namespace='paired')
                adapters = {'baseline': PaperPilotKnowledgeSearchAdapter(baseline),
                            'content_query': PaperPilotKnowledgeSearchAdapter(variant)}
                for qi, qa in enumerate(paper['qas']):
                    row = {'paper_id': pid, 'question_id': qa['question_id'], 'question': qa['question'],
                           'lexical_variant_query': content_query(qa['question'])}
                    # Alternate order to reduce warm-cache timing bias.
                    order = list(adapters) if qi % 2 == 0 else list(reversed(adapters))
                    for name in order:
                        started = time.perf_counter()
                        try:
                            hits = list(adapters[name].search(qa['question'], top_k=8,
                                                            paper_sha256s={document.sha256}))
                            original = {c.stable_id: c.text for c in baseline.get_chunks([h['stable_id'] for h in hits])}
                            texts = [original[h['stable_id']] for h in hits]
                            error = None
                        except Exception as exc:
                            hits, texts, error = [], [], type(exc).__name__
                        elapsed = time.perf_counter() - started
                        timing[name] += elapsed
                        row[name] = {'metrics': text_metrics(qa, texts, [c.text for c in chunks]),
                                     'ranks': [h['stable_id'] for h in hits], 'texts': texts,
                                     'error': error, 'seconds': elapsed}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    scores.append(row)
            if (paper_index + 1) % 25 == 0:
                print(f'Finished {paper_index + 1}/{len(dataset)} papers', flush=True)
    eligible = [r for r in scores if r['baseline']['metrics']['best_text_recall'] is not None]
    def score(row, name, k=8):
        return row[name]['metrics']['best_text_recall'][str(k)]
    groups = {}
    for row in eligible:
        groups.setdefault(row['paper_id'], []).append(score(row, 'content_query') - score(row, 'baseline'))
    ids = sorted(groups)
    rng = random.Random(20260913)
    bootstrap = []
    for _ in range(2000):
        values = [d for pid in rng.choices(ids, k=len(ids)) for d in groups[pid]]
        bootstrap.append(sum(values)/len(values))
    bootstrap.sort()
    deltas = [score(r, 'content_query') - score(r, 'baseline') for r in eligible]
    summary = {'queries': len(scores), 'papers': len(dataset), 'text_denominator': len(eligible),
        'mean_recall': {name: {str(k): sum(score(r, name, k) for r in eligible)/len(eligible)
                              for k in (1, 3, 8)} for name in adapters},
        'delta_top8': sum(deltas)/len(deltas), 'paper_bootstrap_delta_95_ci': [bootstrap[50], bootstrap[1949]],
        'paired': {'improved': sum(d > 0 for d in deltas), 'regressed': sum(d < 0 for d in deltas),
                   'unchanged': sum(d == 0 for d in deltas)},
        'errors': {name: sum(r[name]['error'] is not None for r in scores) for name in adapters},
        'retrieval_seconds': dict(timing), 'provider_requests': 0, 'official_qasper_score': False}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output), indent=2))
