"""Frozen retrieval measurements; never substitute coverage for answer correctness."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.paper_evidence import PaperEvidenceAdapter
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
from build_paper_evidence import local_models
from check_token_evidence_pilot import covered, pack_hits
from evals.structure_pilot import normal
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True, type=Path)
    p.add_argument('--questions', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--old-results', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = RuntimeSettings.from_env()
    budget, embedder = local_models(settings)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ.get('PAPERPILOT_RERANKER_CACHE',
        settings.workspace / 'model-cache/reranker')), allow_download=False)
    reranker.warmup()
    adapter = PaperEvidenceAdapter(args.index, SimpleNamespace(embedder=embedder, reranker=reranker, paper_roots=()))
    frozen = args.questions.read_bytes()
    data = json.loads(frozen)
    questions = data if isinstance(data, list) else data['questions']
    (args.output / 'questions.json').write_bytes(frozen)
    save_json(args.output / 'scope.json', {'index_identity': adapter.index_identity,
        'questions_sha256': digest(frozen), 'model': embedder.cache_identity,
        'reranker_sha256': reranker.model_sha256, 'reranker_tokenizer_sha256': reranker.tokenizer_sha256,
        'context_token_cap': 2048, 'top_k': 8, 'api_calls': 0,
        'meaning': 'retrieval/source coverage only; no entailment, numerical column correctness or answer quality verdict',
        'source_freeze': {str(f.relative_to(Path(__file__).resolve().parents[1])): digest(f.read_bytes())
                          for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
    rows = []
    for case in questions:
        start = time.perf_counter()
        category = case.get('category', 'old_development')
        scope = case['paper_ids']
        # Catalogue discovery is measured separately from scoped body recall.
        selected = adapter.library.search_papers(case['question'], limit=6)
        found = {p['paper_id'] for p in selected['papers']}
        hits = [] if category == 'catalog' else adapter.search(case['question'], top_k=8, paper_sha256s=set(scope))
        packet, omitted, size = pack_hits(hits, budget, 2048)
        gold = [g for g in case.get('gold_source_spans', []) if g.get('quote')]
        row = {'case': case['id'], 'category': category, 'split': case.get('split', 'development_regression'),
            'paper_selection_hits': len(found & set(scope)), 'paper_selection_gold': len(scope),
            'hit_ids': [h['stable_id'] for h in hits], 'context_source_ids': [h['stable_id'] for h in packet],
            'gold': len(gold), 'recall_hits': sum(covered(g, hits, normalizer=normal) for g in gold),
            'context_hits': sum(covered(g, packet, normalizer=normal) for g in gold),
            'context_tokens': size, 'omitted_source_ids': omitted,
            'elapsed_seconds': time.perf_counter()-start, 'answer_quality': 'NOT_MEASURED'}
        if category == 'catalog':
            fields = adapter.library.papers[scope[0]]['fields']
            expected = case['expected_metadata']
            row['known_metadata_correct'] = sum(fields[k]['value'] == v and fields[k]['status'] == 'KNOWN' for k, v in expected.items())
            row['metadata_fields'] = len(expected)
            row['metadata_statuses'] = {k: fields[k]['status'] for k in expected}
        if category == 'experiment':
            numbers = case['table_assertions']
            row['numeric_literal_presence_only'] = sum(any(g['value'] in h['content'] and h['paper_sha256'] == scope[0]
                and h['page'] <= case['required_table_page'] <= h['page_end'] for h in packet) for g in numbers)
            row['numeric_gold'] = len(numbers)
            row['header_cell_condition_correctness'] = 'REQUIRES_INDEPENDENT_CONTENT_REVIEW'
        if category == 'no_answer':
            row['abstention_correctness'] = 'REQUIRES_ANSWER_GENERATION_AND_REVIEW; nonempty retrieval is not a positive answer'
        rows.append(row)
        save_json(args.output / (case['id'] + '.json'), {'row': row, 'hits': hits, 'packet': packet, 'catalogue': selected})
        print(json.dumps({'case': case['id'], 'gold': len(gold), 'context_hits': row['context_hits'],
                          'seconds': round(row['elapsed_seconds'], 2)}), flush=True)
    summary = {split: {k: sum(r.get(k, 0) for r in rows if r['split'] == split) for k in
        ('gold', 'recall_hits', 'context_hits', 'paper_selection_hits', 'paper_selection_gold',
         'known_metadata_correct', 'metadata_fields', 'numeric_literal_presence_only', 'numeric_gold')}
        for split in sorted({r['split'] for r in rows})}
    result = {'rows': rows, 'summary': summary, 'answer_quality': 'NOT_MEASURED', 'api_calls': 0}
    if args.old_results:
        old = json.loads(args.old_results.read_bytes())
        result['historical_control'] = {'sha256': digest(args.old_results.read_bytes()), 'summary': old['summary'],
            'comparability': 'Same questions, original text, local models and 2048-token packet schema. Historical rankings retained; encoder window aggregation is the new arm. Historical child index uses paragraph-level parents.'}
    save_json(args.output / 'results.json', result)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
