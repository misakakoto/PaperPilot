"""Small retrieval regression and read-only replay of historical public scores.

No provider calls, downloads, training, source-library writes or parameter search.
The six method questions are reused development material, not a new holdout.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def historical_replay(output):
    from evals.qasper.audit_llm_rerank import audit
    baseline = ROOT.parent / 'PaperPilot-AnswerFirst' / '.paperpilot'
    source = baseline / 'evidence-select-confirm-v2'
    copied = output / 'historical-copy'
    copied.mkdir()
    # audit writes a new receipt. Give it owned copies; never write the baseline.
    for path in source.iterdir():
        if path.is_file() and path.name != 'independent-audit.json':
            shutil.copyfile(path, copied / path.name)
    report = audit(baseline / 'datasets/qasper/v0.3/qasper-test-v0.3.json',
                   baseline / 'candidate-budget-confirm-v2', copied)
    original = json.loads((source / 'summary.json').read_bytes())
    report.update(baseline_recall=original['recall']['baseline'],
                  new_api_calls=0, historical_provider_calls=original['provider_attempts'],
                  dataset_sha256=json.loads((copied / 'manifest.json').read_bytes())['dataset_sha256'])
    save(output / 'historical-replay.json', report)
    print(json.dumps({'historical_verified': True, 'questions': report['queries'],
                      'text_questions': report['eligible'], 'recall8': report['recall']['8']}), flush=True)


def local_retrieval(output):
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_agentic.paper_evidence import PaperEvidenceAdapter
    from paperpilot_retrieval import PaperKnowledgeStore
    from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
    from build_paper_evidence import local_models
    from check_token_evidence_pilot import covered, pack_hits
    from evals.structure_pilot import normal

    settings = RuntimeSettings.from_env()
    benchmark = ROOT / 'outputs/paper-rag-v2-benchmark-preparation-20261027/benchmark-frozen.json'
    raw = json.loads(benchmark.read_bytes())
    questions = raw if isinstance(raw, list) else raw['questions']
    questions = [q for q in questions if q['category'] == 'method']
    if len(questions) != 6:
        raise ValueError('Expected all six frozen method questions; never select by outcomes')
    save(output / 'questions.json', questions)
    with sqlite3.connect(settings.memory_db.resolve().as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(output / 'source.sqlite3') as target:
            source.backup(target)
    budget, embedder = local_models(settings)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ.get(
        'PAPERPILOT_RERANKER_CACHE', settings.workspace / 'model-cache/reranker')),
        batch_size=4, allow_download=False)
    reranker.warmup()
    store = PaperKnowledgeStore(output / 'source.sqlite3', namespace=settings.memory_namespace)
    with store._connect() as c:
        counts = c.execute('''SELECT count(DISTINCT p.sha256),count(*) FROM chunks c
            JOIN papers p ON p.row_id=c.paper_row_id WHERE c.namespace=? AND p.active=1''',
            (settings.memory_namespace,)).fetchone()
        dense = c.execute('''SELECT count(*) FROM chunk_embeddings e JOIN chunks c ON c.row_id=e.chunk_row_id
            JOIN papers p ON p.row_id=c.paper_row_id WHERE c.namespace=? AND p.active=1 AND e.model=?''',
            (settings.memory_namespace, embedder.cache_identity)).fetchone()[0]
        originals = c.execute('''SELECT p.sha256,c.text FROM chunks c JOIN papers p ON p.row_id=c.paper_row_id
            WHERE c.namespace=? AND p.active=1''', (settings.memory_namespace,)).fetchall()
    if dense != counts[1]:
        raise ValueError('Dense snapshot is incomplete; this smoke test does not re-embed the library')
    if any(not any(row[0] == g['paper_sha256'] and normal(g['quote']) in normal(row[1])
                   for row in originals) for q in questions for g in q['gold_source_spans'] if g.get('quote')):
        raise ValueError('A frozen gold anchor is absent from the current text; stop rather than change labels')
    legacy = PaperPilotKnowledgeSearchAdapter(store, embedder=embedder, reranker=reranker,
                                              rerank_stage='final', paper_roots=())
    index = ROOT / 'outputs/paper-evidence-v2-fixed-text-20261027/new6'
    window = PaperEvidenceAdapter(index, legacy)
    save(output / 'scope.json', {'question_set_sha256': sha(output / 'questions.json'),
        'benchmark_sha256': sha(benchmark), 'papers': counts[0], 'chunks': counts[1], 'dense_vectors': dense,
        'embedding': embedder.cache_identity, 'reranker_sha256': reranker.model_sha256,
        'reranker_tokenizer_sha256': reranker.tokenizer_sha256, 'reranker_batch_size': 4,
        'window_index': window.index_identity, 'top_k': 8, 'context_token_cap': 2048,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'src').rglob('*.py')},
        'scope': 'Six reused method questions. Gold-span retrieval coverage, not semantic precision or answer accuracy.',
        'comparison': 'Legacy full-library vs SHA-scoped diagnostic; scoped window arm is a six-paper pilot profile, not whole-library deployment.',
        'api_calls': 0, 'downloads': False, 'source_library_modified': False})
    print(json.dumps({'models_ready': True, 'papers': counts[0], 'chunks': counts[1]}), flush=True)
    rows = []
    for question in questions:
        for arm, adapter, scope in [('legacy_full_library', legacy, None),
                ('legacy_selected_paper', legacy, set(question['paper_ids'])),
                ('window_selected_paper_pilot', window, set(question['paper_ids']))]:
            started = time.perf_counter()
            hits = list(adapter.search(question['question'], top_k=8, paper_sha256s=scope))
            packet, omitted, tokens = pack_hits(hits, budget, 2048)
            gold = [g for g in question['gold_source_spans'] if g.get('quote')]
            row = {'case': question['id'], 'arm': arm, 'gold': len(gold),
                'hit_count': len(hits), 'recall_hits': sum(covered(g, hits, normalizer=normal) for g in gold),
                'context_hits': sum(covered(g, packet, normalizer=normal) for g in gold),
                'target_paper_found': set(question['paper_ids']) <= {h['paper_sha256'] for h in hits},
                'context_tokens': tokens, 'omitted_ids': omitted, 'seconds': time.perf_counter()-started,
                'first_gold_rank': next((n for n in range(1, len(hits)+1)
                    if any(covered(g, hits[:n], normalizer=normal) for g in gold)), None),
                'precision': 'NOT_MEASURED_PARTIAL_GOLD', 'answer_accuracy': 'NOT_MEASURED'}
            rows.append(row)
            save(output / (question['id']+'-'+arm+'.json'), {'row': row, 'hits': hits, 'packet': packet})
            save(output / 'local-progress.json', rows)
            print(json.dumps({'case': question['id'], 'arm': arm, 'hit': row['recall_hits'],
                              'seconds': round(row['seconds'], 2)}), flush=True)
    totals = {arm: {key: sum(r[key] for r in rows if r['arm'] == arm)
              for key in ['gold', 'recall_hits', 'context_hits', 'target_paper_found', 'seconds']}
              for arm in dict.fromkeys(r['arm'] for r in rows)}
    save(output / 'local-results.json', {'summary': totals, 'rows': rows, 'new_api_calls': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, operation in [('historical', historical_replay), ('local', local_retrieval)]:
        try:
            operation(output)
        except Exception as exc:
            save(output / (name+'-error.json'), {'error': type(exc).__name__, 'message': str(exc)})
            raise
        gc.collect()


if __name__ == '__main__':
    main()
