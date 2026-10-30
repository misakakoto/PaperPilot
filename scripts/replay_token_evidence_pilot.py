"""Re-score saved pilot rankings with the historical evaluator's normalization.

No search, embedding, reranking or API calls. Original results are never replaced.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.structure_pilot import normal
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.paper_structure import digest
from paperpilot_agentic.token_evidence import TokenBudget, TokenEvidenceAdapter
from paperpilot_retrieval import PaperKnowledgeStore
from check_token_evidence_pilot import covered, pack_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Replay output must be new; preserve all earlier measurements')
    root = args.directory.resolve()
    scope = json.loads((root / 'scope.json').read_bytes())
    budget = TokenBudget.restore(scope['tokenizer'])
    baseline = PaperPilotKnowledgeSearchAdapter(
        PaperKnowledgeStore(root / 'baseline.sqlite3', namespace='structured-pilot-v1'))
    # This object supplies the pinned identity only; no model is instantiated.
    identity_only = SimpleNamespace(cache_identity=scope['embedding_identity'])
    token = TokenEvidenceAdapter(root / 'workspace/token-evidence-pilot',
        SimpleNamespace(embedder=identity_only, reranker=None, paper_roots=()))
    questions = {q['id']: q for q in json.loads((root / 'questions.json').read_bytes())}
    original = json.loads((root / 'results.json').read_bytes())
    rows = []
    for saved in original['rows']:
        adapter = baseline if saved['arm'] == 'long_structured' else token
        by_id = {h.stable_id: h for h in adapter.store.get_chunks(saved['hit_ids'])}
        hits = [adapter._hit_to_item(by_id[identity], rank=n, memory_ids=[])
                for n, identity in enumerate(saved['hit_ids'], 1)]
        question = questions[saved['case']]
        if saved['arm'] == 'token_children_parent_read':
            expanded = token.read_context(saved['hit_ids'], max_tokens=scope['context_token_cap'],
                paper_sha256s=set(question['paper_ids'])) if hits else {'evidence': [], 'context_tokens': 0}
            packed, size = expanded['evidence'], expanded['context_tokens']
        else:
            packed, _, size = pack_hits(hits, budget, scope['context_token_cap'])
        if size != saved['context_tokens']:
            raise ValueError('Context reconstruction differs from the saved run')
        gold = [g for g in question['gold_source_spans'] if g.get('quote')]
        rows.append({'case': saved['case'], 'arm': saved['arm'], 'gold': len(gold),
            'recall_hits': sum(covered(g, hits, normalizer=normal) for g in gold),
            'context_hits': sum(covered(g, packed, normalizer=normal) for g in gold)})
    summary = {arm: {key: sum(r[key] for r in rows if r['arm'] == arm)
                    for key in ('gold', 'recall_hits', 'context_hits')}
               for arm in scope['arms']}
    result = {'normalization': 'EXISTING evals.structure_pilot.normal: NFKC+casefold+alphanumeric; source location only, not entailment',
        'ranking_recomputed': False, 'api_calls': 0, 'embedding_inferences': 0,
        'original_strict_results_retained': True, 'rows': rows, 'summary': summary,
        'source_results_sha256': digest((root / 'results.json').read_bytes()),
        'runner_sha256': digest(Path(__file__).read_bytes())}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
