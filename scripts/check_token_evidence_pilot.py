"""Build/evaluate isolated token retrieval on an immutable structured corpus.

Uses existing local ONNX models only: no API, downloads, training or source-index
writes. Questions and source manifest are frozen before indexing. This reuses a
historical development set; it is not a new independent benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_agentic.token_evidence import TokenBudget, TokenEvidenceAdapter, build_index, encoded
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore, FastEmbedder
from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker


class UncachedEmbedder(FastEmbedder):
    def embed_documents(self, texts):
        # Vectors persist in this evaluation's own corpus DB, never shared caches.
        return self._embed(texts)


def normal(text):
    return re.sub(r'\s+', '', str(text)).casefold()


def covered(gold, rows, *, normalizer=normal):
    for row in rows:
        provenance = row.get('provenance') or {}
        sha = row.get('paper_sha256') or provenance.get('paper_sha256')
        pages = [s['page'] for s in provenance.get('source_segments', [])]
        if not pages:
            span = provenance.get('page_range') or [row.get('page'), row.get('page_end', row.get('page'))]
            if all(type(p) is int for p in span):
                pages = list(range(span[0], span[1] + 1))
        if sha == gold['paper_sha256'] and gold['page'] in pages and normalizer(gold['quote']) in normalizer(row['content']):
            return True
    return False


def pack_hits(hits, budget, limit):
    rows, omitted = [], []
    for hit in hits:
        row = {'content': hit['content'], 'paper_sha256': hit['paper_sha256'],
               'page': hit['page'], 'page_end': hit['page_end'], 'stable_id': hit['stable_id'],
               'provenance': {'source_segments': hit.get('provenance', {}).get('source_segments', []),
                              'page_range': hit.get('provenance', {}).get('page_range')}}
        if budget.count(encoded(rows + [row]).decode()) <= limit:
            rows.append(row)
        else:
            omitted.append(hit['stable_id'])
    return rows, omitted, budget.count(encoded(rows).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pilot', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate-only', action='store_true', help='Reuse a completed frozen build after an evaluator failure; never replace saved rankings')
    args = parser.parse_args()
    settings = RuntimeSettings.from_env()
    output = args.output.resolve()
    if not args.evaluate_only:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir() or (output / 'results.json').exists():
        raise ValueError('Reuse only a completed build without saved rankings; existing results are immutable')
    source = args.source_pilot.resolve()
    questions_raw = args.questions.read_bytes()
    questions = json.loads(questions_raw)
    source_manifest_raw = (source / 'index.json').read_bytes()
    if args.evaluate_only:
        assert (output / 'questions.json').read_bytes() == questions_raw
        assert (output / 'source-index.json').read_bytes() == source_manifest_raw
    else:
        (output / 'questions.json').write_bytes(questions_raw)
        (output / 'source-index.json').write_bytes(source_manifest_raw)
    root = Path(__file__).resolve().parents[1]
    if not args.evaluate_only:
        save_json(output / 'source-freeze.json', {p.relative_to(root).as_posix(): digest(p.read_bytes())
            for p in (root / 'src').rglob('*.py') if '__pycache__' not in p.parts})
    cache = Path(os.environ.get('PAPERPILOT_EMBEDDING_CACHE', settings.workspace / 'model-cache/embedding'))
    tokenizer_paths = list(cache.rglob('tokenizer.json'))
    if len(tokenizer_paths) != 1:
        raise ValueError('Select exactly one configured tokenizer snapshot')
    budget = TokenBudget.from_directory(tokenizer_paths[0].parent)
    embedder = UncachedEmbedder(cache_dir=cache, allow_download=False)
    identity = embedder.cache_identity
    reranker = OnnxCrossEncoderReranker(
        cache_dir=Path(os.environ.get('PAPERPILOT_RERANKER_CACHE', settings.workspace / 'model-cache/reranker')),
        allow_download=False)
    reranker.warmup()
    scope = {'source_index_identity': digest(source_manifest_raw),
        'questions_identity': digest(questions_raw), 'questions': len(questions),
        'embedding_identity': identity, 'reranker': reranker.name,
        'reranker_revision': reranker.resolved_revision, 'reranker_model_sha256': reranker.model_sha256,
        'reranker_tokenizer_sha256': reranker.tokenizer_sha256,
        'runner_sha256': digest(Path(__file__).read_bytes()),
        'tokenizer': budget.identity, 'context_token_cap': 2048, 'top_k': 8,
        'arms': ['long_structured', 'token_children', 'token_children_parent_read'],
        'api_calls': 0, 'training': False, 'source_library_modified': False,
        'question_scope': 'historical development/previous holdout; not newly independent',
        'control': 'Same parsed text, embedding, hybrid retrieval, reranker and context cap; children change prefixes and granularity together; parent arm adds original context expansion'}
    if args.evaluate_only:
        original_scope = json.loads((output / 'scope.json').read_bytes())
        assert all(scope[k] == original_scope[k] for k in ('source_index_identity', 'questions_identity',
            'embedding_identity', 'reranker_model_sha256', 'reranker_tokenizer_sha256', 'tokenizer', 'context_token_cap'))
        save_json(output / 'evaluation-attempt-2.json', {'runner_sha256': scope['runner_sha256'],
            'reason': 'Evaluator used scalar page for a cross-page hit whose page is None; use retained page_range',
            'reused_build': True, 'embedding_recomputed': False, 'api_calls': 0})
    else:
        save_json(output / 'scope.json', scope)
    # Consistent read-only backup, rather than copying an SQLite file with a WAL.
    if not args.evaluate_only:
        with sqlite3.connect((source / 'corpus.sqlite3').as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(output / 'baseline.sqlite3') as copied:
                original.backup(copied)
    baseline_store = PaperKnowledgeStore(output / 'baseline.sqlite3', namespace='structured-pilot-v1')
    if not args.evaluate_only:
        baseline_store.ensure_embeddings(embedder)
    baseline = PaperPilotKnowledgeSearchAdapter(baseline_store, embedder=embedder, reranker=reranker, rerank_stage='final')
    workspace = output / 'workspace'
    if args.evaluate_only:
        build = json.loads((workspace / 'token-evidence-pilot/index.json').read_bytes())
    else:
        build = build_index(source, workspace / 'token-evidence-pilot', budget, embedder=embedder)
    token = TokenEvidenceAdapter(workspace / 'token-evidence-pilot',
        SimpleNamespace(embedder=embedder, reranker=reranker, rerank_stage='final', paper_roots=()))
    assert all(c['input_tokens'] <= budget.limit for c in token.evidence['children'].values())
    for parent_id, parent in token.evidence['parents'].items():
        parts = sorted((c for c in token.evidence['children'].values() if c['parent_id'] == parent_id),
                       key=lambda c: c['start'])
        assert parts[0]['start'] == 0 and parts[-1]['end'] == len(parent['text'])
        assert all(a['end'] == b['start'] for a, b in zip(parts, parts[1:]))
    if not args.evaluate_only:
        save_json(output / 'build.json', build)
    print(json.dumps({'built_children': build['children'], 'parents': build['parents'], 'token_limit': budget.limit}), flush=True)
    rows = []
    for case in questions:
        gold = [g for g in case['gold_source_spans'] if g.get('quote')]
        for arm, adapter in [('long_structured', baseline), ('token_children', token)]:
            started = time.monotonic()
            hits = list(adapter.search(case['question'], top_k=8, paper_sha256s=set(case['paper_ids'])))
            packed, omitted, size = pack_hits(hits, budget, 2048)
            row = {'case': case['id'], 'arm': arm, 'previously_heldout': case.get('heldout', False),
                   'gold': len(gold), 'unannotated': len(case['gold_source_spans']) - len(gold),
                   'recall_hits': sum(covered(g, hits) for g in gold),
                   'context_hits': sum(covered(g, packed) for g in gold),
                   'context_tokens': size, 'omitted_ids': omitted,
                   'elapsed_ms': (time.monotonic() - started) * 1000,
                   'hit_ids': [h['stable_id'] for h in hits]}
            rows.append(row)
            if arm == 'token_children':
                expanded = token.read_context(row['hit_ids'], max_tokens=2048,
                                              paper_sha256s=set(case['paper_ids'])) if hits else {'evidence': [], 'context_tokens': 0, 'omitted_parent_ids': []}
                rows.append({**row, 'arm': 'token_children_parent_read',
                    'context_hits': sum(covered(g, expanded['evidence']) for g in gold),
                    'context_tokens': expanded['context_tokens'],
                    'omitted_ids': expanded['omitted_parent_ids'],
                    'note': 'Same child ranking; full parents replace child context, original search receipt retained'})
        save_json(output / 'results.json', {'rows': rows, 'api_calls': 0})
        print(json.dumps({'case': case['id'], 'complete': True}), flush=True)
    assert (source / 'index.json').read_bytes() == source_manifest_raw
    freeze = json.loads((output / 'source-freeze.json').read_bytes())
    assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == sha for p, sha in freeze.items())
    summary = {}
    for arm in ('long_structured', 'token_children', 'token_children_parent_read'):
        selected = [r for r in rows if r['arm'] == arm]
        summary[arm] = {'queries': len(selected), 'gold': sum(r['gold'] for r in selected),
            'recall_hits': sum(r['recall_hits'] for r in selected),
            'context_hits': sum(r['context_hits'] for r in selected),
            'mean_context_tokens': sum(r['context_tokens'] for r in selected) / len(selected)}
    save_json(output / 'summary.json', {'arms': summary, 'api_calls': 0,
        'scientific_answer_quality': 'NOT_EVALUATED', 'original_source_unchanged': True,
        'children_exceeding_embedding_limit': 0, 'complete_parsed_parent_character_coverage': True})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
