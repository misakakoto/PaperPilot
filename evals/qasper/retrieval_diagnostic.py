"""Offline full-dev text retrieval diagnostic; not QASPER answer/official Evidence F1."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from evals.qasper.agent_case import paper_records
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore


class RankingProbe(PaperPilotKnowledgeSearchAdapter):
    """Evaluation-only paired selection from identical candidates, no gold access."""

    def _diversify(self, items, **kwargs):
        self.candidates = [dict(item) for item in items]
        self.no_similarity_penalty = super()._diversify(items, **{**kwargs, 'diversity': 0.0})
        return super()._diversify(items, **kwargs)


def text_metrics(qa, ranked_texts, corpus_texts):
    """Best-annotator exact-paragraph recall; unmatched gold stays in denominator."""
    references, figures = [], 0
    for annotation in qa['answers']:
        answer = annotation['answer']
        evidence = answer.get('evidence', []) if not answer.get('unanswerable') else []
        figures += sum('FLOAT SELECTED' in text for text in evidence)
        texts = set(text for text in evidence if text and 'FLOAT SELECTED' not in text)
        if texts:
            references.append(texts)
    if not references:
        return {'text_reference_count': 0, 'figure_annotations': figures,
                'best_text_recall': None, 'best_corpus_text_coverage': None}
    recall = {str(k): max(len(set(ranked_texts[:k]) & gold) / len(gold)
                         for gold in references) for k in (1, 3, 8)}
    return {'text_reference_count': len(references), 'figure_annotations': figures,
            'best_text_recall': recall,
            'best_corpus_text_coverage': max(len(set(corpus_texts) & gold) / len(gold)
                                           for gold in references)}


def run(dataset_path, output, *, compare_ranking=False):
    dataset_path, output = Path(dataset_path), Path(output)
    # No test-set tuning and no sampling/cherry-picking switch in this diagnostic.
    if dataset_path.name != 'qasper-dev-v0.3.json':
        raise ValueError('This diagnostic accepts qasper-dev-v0.3.json only')
    raw = dataset_path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    questions = [{'paper_id': pid, 'question_id': qa['question_id'], 'question': qa['question']}
                 for pid, paper in data.items() for qa in paper['qas']]
    if not questions or len({q['question_id'] for q in questions}) != len(questions):
        raise ValueError('Empty dataset or duplicate question ID')
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), root/'evals/qasper/agent_case.py']
    sources += sorted((root/'src/paperpilot_retrieval').glob('*.py'))
    sources.append(root/'src/paperpilot_agentic/adapters.py')
    manifest = {'protocol': 'qasper-full-dev-scoped-text-retrieval-v1',
                'dataset_sha256': hashlib.sha256(raw).hexdigest(),
                'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                'questions': questions, 'provider_requests': 0, 'top_k': 8,
                'scope': 'one supplied paper; public text and captions; no gold in retrieval',
                'dense': False, 'reranker': False, 'memory': 'fresh empty namespace per paper',
                'official_benchmark_comparable': False,
                'held_out_claim': False, 'compare_ranking': compare_ranking,
                'ranking_comparison': ('same candidate pool and source quotas; similarity penalty 0.22 vs 0'
                                       if compare_ranking else None)}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Only disposable databases; no runtime settings, credentials or provider constructed.
    rows = []
    with (output/'rankings.jsonl').open('x', encoding='utf-8') as stream:
        for pid, paper in data.items():
            public = {key: paper[key] for key in ('title', 'abstract', 'full_text', 'figures_and_tables')
                      if key in paper}
            _, document, chunks = paper_records(public, source_id=pid)
            with tempfile.TemporaryDirectory(prefix='paperpilot-qasper-retrieval-') as temporary:
                store = PaperKnowledgeStore(Path(temporary)/'index.sqlite3', namespace='diagnostic')
                store.sync_documents([document], chunks)
                adapter = RankingProbe(store) if compare_ranking else PaperPilotKnowledgeSearchAdapter(store)
                for qa in paper['qas']:
                    error = None
                    alternative, candidates = [], []
                    try:
                        hits = list(adapter.search(qa['question'], top_k=8,
                                                   paper_sha256s={document.sha256}))
                        # Adapter content includes title/section. Resolve original
                        # chunk text by stable ID without guessing string prefixes.
                        originals = {h.stable_id: h.text for h in store.get_chunks(
                            [h['stable_id'] for h in hits])}
                        hits = [{**h, 'original_chunk_text': originals[h['stable_id']]} for h in hits]
                        if compare_ranking:
                            candidates = adapter.candidates
                            originals = {h.stable_id: h.text for h in store.get_chunks(
                                [h['stable_id'] for h in candidates])}
                            candidates = [{**h, 'original_chunk_text': originals[h['stable_id']]}
                                          for h in candidates]
                            alternative = [{**h, 'rank': rank,
                                            'original_chunk_text': originals[h['stable_id']]}
                                           for rank, h in enumerate(adapter.no_similarity_penalty, 1)]
                    except Exception as exc:
                        error, hits = type(exc).__name__, []
                        alternative, candidates = [], []
                    row = {'paper_id': pid, 'question_id': qa['question_id'], 'question': qa['question'],
                           'status': 'FAILED' if error else 'COMPLETED', 'error_type': error,
                           'hits': hits, **text_metrics(qa, [h['original_chunk_text'] for h in hits],
                                                       [c.text for c in chunks])}
                    if compare_ranking:
                        row['ranking_comparison'] = {
                            'candidate_pool': candidates, 'alternative_hits': alternative,
                            'alternative_metrics': text_metrics(qa,
                                [h['original_chunk_text'] for h in alternative], [c.text for c in chunks]),
                            'candidate_metrics': text_metrics(qa, [],
                                [h['original_chunk_text'] for h in candidates])}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    rows.append(row)
    scored = [row for row in rows if row['best_text_recall'] is not None]
    summary = {'protocol': manifest['protocol'], 'dataset_sha256': manifest['dataset_sha256'],
               'queries': len(rows), 'papers': len(data), 'provider_requests': 0,
               'failed_queries': sum(row['status'] == 'FAILED' for row in rows),
               'empty_rankings': sum(not row['hits'] for row in rows),
               'text_metric_denominator': len(scored),
               'no_text_reference_queries': len(rows) - len(scored),
               'queries_with_figure_annotations': sum(row['figure_annotations'] > 0 for row in rows),
               'macro_best_text_recall': {str(k): sum(r['best_text_recall'][str(k)] for r in scored)
                                         / len(scored) if scored else None for k in (1, 3, 8)},
               'macro_best_corpus_text_coverage': sum(r['best_corpus_text_coverage'] for r in scored)
                                                   / len(scored) if scored else None,
               'official_benchmark_comparable': False, 'held_out_claim': False,
               'limitations': ['Exact original paragraph matching; no normalization or partial credit.',
                               'Best annotation per question; not union of every annotator.',
                               'Text metric excludes FLOAT SELECTED evidence, not a figure/table reader test.',
                               'Questions with no textual references have null recall, not a perfect score.',
                               'Errors and zero hits score zero for questions with text references.',
                               'Ranks 1/3 are prefixes of the top-8 call, not separate configurations.',
                               'No section routing, numeric lookup, model answering or audit.',
                               'Full dev diagnostic, not unseen-paper generalization or official Evidence F1.']}
    if compare_ranking:
        def alternative_value(row, k):
            return row['ranking_comparison']['alternative_metrics']['best_text_recall'][str(k)]

        summary['ranking_comparison'] = {
            'alternative': 'similarity_penalty_zero; source quota and per-paper penalty unchanged',
            'macro_alternative_text_recall': {str(k): sum(alternative_value(r, k) for r in scored)
                                             / len(scored) if scored else None for k in (1, 3, 8)},
            'macro_candidate_text_coverage': sum(r['ranking_comparison']['candidate_metrics']
                ['best_corpus_text_coverage'] for r in scored) / len(scored) if scored else None,
            'paired_top8': {label: sum(predicate(alternative_value(r, 8), r['best_text_recall']['8'])
                                      for r in scored)
                            for label, predicate in [('improved', lambda a, b: a > b),
                                                     ('regressed', lambda a, b: a < b),
                                                     ('unchanged', lambda a, b: a == b)]}}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-ranking', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output, compare_ranking=args.compare_ranking), ensure_ascii=False))


if __name__ == '__main__':
    main()
