"""Replay fixed rankings to compare full receipts and source-handle context views."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.structure_pilot import normal
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_agentic.token_evidence import TokenEvidenceAdapter, encoded
from paperpilot_retrieval import PaperKnowledgeStore
from check_token_evidence_pilot import covered, pack_hits


def compact_hits(hits, budget, limit):
    rows, omitted = [], []
    for hit in hits:
        span = hit['provenance']['page_range']
        row = {'evidence_type': 'TEXT', 'content': hit['content'], 'page': span[0],
            'source_uri': hit['source_uri'], 'provenance': {'paper_sha256': hit['paper_sha256'],
                'page_range': span, 'hit_child_ids': [hit['stable_id']]}}
        if budget.count(encoded(rows + [row]).decode()) <= limit:
            rows.append(row)
        else:
            omitted.append(hit['stable_id'])
    return rows, omitted, budget.count(encoded(rows).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.pilot.resolve()
    scope = json.loads((source / 'scope.json').read_bytes())
    questions = {q['id']: q for q in json.loads((source / 'questions.json').read_bytes())}
    rankings = json.loads((source / 'results.json').read_bytes())['rows']
    baseline = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(source / 'baseline.sqlite3', namespace='structured-pilot-v1'))
    token = TokenEvidenceAdapter(source / 'workspace/token-evidence-pilot',
        SimpleNamespace(embedder=SimpleNamespace(cache_identity=scope['embedding_identity']), reranker=None, paper_roots=()))
    root = Path(__file__).resolve().parents[1]
    freeze = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in (root / 'src').rglob('*.py')}
    save_json(args.output / 'scope.json', {'protocol': 'fixed original hit IDs/order, original texts, original tokenizer, 2048-token evidence cap; only context view changes',
        'runner_sha256': digest(Path(__file__).read_bytes()), 'source_freeze': freeze,
        'source_scope': scope, 'ranking_recomputed': False, 'api_calls': 0, 'embedding_inferences': 0,
        'metric': 'existing NFKC+casefold+alphanumeric source containment, not answer accuracy',
        'question_scope': 'historical reused development set, no independent heldout claim',
        'baseline_note': 'Long-block compact arm uses identical visible keys; stable IDs resolve original chunks in its store, not token-parent IDs'})
    rows = []
    for saved in rankings:
        if saved['arm'] == 'token_children_parent_read':
            continue
        question = questions[saved['case']]
        adapter = baseline if saved['arm'] == 'long_structured' else token
        by_id = {h.stable_id: h for h in adapter.store.get_chunks(saved['hit_ids'])}
        hits = [adapter._hit_to_item(by_id[identity], rank=n, memory_ids=[])
                for n, identity in enumerate(saved['hit_ids'], 1)]
        measured = {}
        if saved['arm'] == 'long_structured':
            full, omitted, size = pack_hits(hits, token.budget, 2048)
            assert size == saved['context_tokens']
            compact, compact_omitted, compact_size = compact_hits(hits, token.budget, 2048)
            measured = {'full': (full, omitted, size), 'source_handles': (compact, compact_omitted, compact_size)}
        else:
            for view in ('full', 'source_handles'):
                receipt = token.read_context(saved['hit_ids'], max_tokens=2048,
                    paper_sha256s=set(question['paper_ids']), view=view)
                measured[view] = (receipt['evidence'], receipt['omitted_parent_ids'], receipt['context_tokens'])
            original_parent = next(r for r in rankings if r['case'] == saved['case'] and r['arm'] == 'token_children_parent_read')
            assert measured['full'][2] == original_parent['context_tokens']
        gold = [g for g in question['gold_source_spans'] if g.get('quote')]
        for view, (evidence, omitted, size) in measured.items():
            assert size <= 2048
            rows.append({'case': saved['case'], 'arm': saved['arm'], 'view': view, 'gold': len(gold),
                'context_hits': sum(covered(g, evidence, normalizer=normal) for g in gold),
                'context_tokens': size, 'selected': len(evidence), 'omitted': omitted,
                'hit_ids': saved['hit_ids']})
    summary = {}
    for arm in ('long_structured', 'token_children'):
        summary[arm] = {}
        for view in ('full', 'source_handles'):
            selected = [r for r in rows if r['arm'] == arm and r['view'] == view]
            summary[arm][view] = {'context_hits': sum(r['context_hits'] for r in selected),
                'gold': sum(r['gold'] for r in selected), 'queries': len(selected),
                'mean_tokens': sum(r['context_tokens'] for r in selected) / len(selected)}
    save_json(args.output / 'results.json', {'rows': rows, 'summary': summary})
    assert all(digest((root / p).read_bytes()) == sha for p, sha in freeze.items())
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
