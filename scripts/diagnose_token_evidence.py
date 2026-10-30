"""Locate token-pilot evidence loss without changing retrieval or reading gold in it."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from evals.structure_pilot import normal
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_agentic.token_evidence import TokenEvidenceAdapter
from paperpilot_retrieval import FastEmbedder, PaperKnowledgeStore
from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
from check_token_evidence_pilot import covered, pack_hits


class Capture:
    def _diversify(self, items, **kwargs):
        self.candidates = [dict(item) for item in items]
        return super()._diversify(items, **kwargs)


class TokenProbe(Capture, TokenEvidenceAdapter):
    pass


class BaselineProbe(Capture, PaperPilotKnowledgeSearchAdapter):
    pass


def parents_for(adapter, hits):
    ids = dict.fromkeys(adapter.evidence['children'][h['stable_id']]['parent_id'] for h in hits)
    return [{'content': p['text'], 'paper_sha256': p['paper_sha256'],
             'provenance': {'source_segments': p['spans']}}
            for identity in ids for p in [adapter.evidence['parents'][identity]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_dotenv()
    source = args.pilot.resolve()
    scope = json.loads((source / 'scope.json').read_bytes())
    questions = json.loads((source / 'questions.json').read_bytes())
    saved = {(r['case'], r['arm']): r for r in json.loads((source / 'results.json').read_bytes())['rows']}
    embedder = FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']), allow_download=False)
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']), allow_download=False)
    reranker.warmup()
    assert embedder.cache_identity == scope['embedding_identity']
    assert reranker.model_sha256 == scope['reranker_model_sha256']
    assert reranker.tokenizer_sha256 == scope['reranker_tokenizer_sha256']
    baseline = BaselineProbe(PaperKnowledgeStore(source / 'baseline.sqlite3', namespace='structured-pilot-v1'),
                             embedder=embedder, reranker=reranker, rerank_stage='final')
    token = TokenProbe(source / 'workspace/token-evidence-pilot',
                      SimpleNamespace(embedder=embedder, reranker=reranker, rerank_stage='final', paper_roots=()))
    root = Path(__file__).resolve().parents[1]
    freeze = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in (root / 'src').rglob('*.py')}
    save_json(args.output / 'scope.json', {'source_scope': scope, 'source_freeze': freeze,
        'runner_sha256': digest(Path(__file__).read_bytes()), 'api_calls': 0,
        'protocol': 'same frozen queries, same ranking; measure pool/ranking/parent/context independently; gold used only after retrieval'})
    rows = []
    for question in questions:
        for arm, adapter in [('long_structured', baseline), ('token_children', token)]:
            hits = list(adapter.search(question['question'], top_k=8, paper_sha256s=set(question['paper_ids'])))
            ids = [h['stable_id'] for h in hits]
            if ids != saved[(question['id'], arm)]['hit_ids']:
                raise ValueError('Frozen ranking changed: ' + question['id'] + '/' + arm)
            gold = [g for g in question['gold_source_spans'] if g.get('quote')]
            def count(items):
                return sum(covered(g, items, normalizer=normal) for g in gold)
            pool = adapter.candidates
            packed, _, _ = pack_hits(hits, token.budget, scope['context_token_cap'])
            row = {'case': question['id'], 'arm': arm, 'gold': len(gold),
                'pool_size': len(pool), 'pool_coverage': count(pool),
                'rerank_top8_coverage': count(pool[:8]), 'selected_coverage': count(hits),
                'packed_coverage': count(packed), 'hit_ids': ids,
                'candidate_ids': [h['stable_id'] for h in pool]}
            if arm == 'token_children':
                expanded = token.read_context(ids, max_tokens=scope['context_token_cap'],
                    paper_sha256s=set(question['paper_ids'])) if ids else {'evidence': [], 'omitted_parent_ids': []}
                row.update(parent_pool_coverage=count(parents_for(token, pool)),
                    parent_top8_coverage=count(parents_for(token, pool[:8])),
                    parent_selected_coverage=count(parents_for(token, hits)),
                    parent_packed_coverage=count(expanded['evidence']),
                    unique_selected_parents=len(parents_for(token, hits)),
                    omitted_parent_ids=expanded['omitted_parent_ids'])
            rows.append(row)
        save_json(args.output / 'rows.json', rows)
        print(json.dumps({'case': question['id'], 'done': True}), flush=True)
    summary = {}
    keys = ('gold', 'pool_coverage', 'rerank_top8_coverage', 'selected_coverage', 'packed_coverage',
            'parent_pool_coverage', 'parent_top8_coverage', 'parent_selected_coverage', 'parent_packed_coverage')
    for arm in ('long_structured', 'token_children'):
        selected = [r for r in rows if r['arm'] == arm]
        summary[arm] = {key: sum(r[key] for r in selected) for key in keys if key in selected[0]}
    assert all(digest((root / p).read_bytes()) == sha for p, sha in freeze.items())
    save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
