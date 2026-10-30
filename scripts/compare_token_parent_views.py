"""Paired parent-context reranking on saved candidates, never searching gold."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from tokenizers import Tokenizer
from evals.structure_pilot import normal
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_retrieval import FastEmbedder
from paperpilot_retrieval.reranking import OnnxCrossEncoderReranker
from check_token_evidence_pilot import covered
from diagnose_token_evidence import TokenProbe, parents_for


def anchored_window(text, start, end, count_pair, limit=512):
    if not 0 <= start < end <= len(text):
        raise ValueError('Invalid source anchor')
    if count_pair(text[start:end]) > limit:
        raise ValueError('The anchored child and query cannot fit the reranker')
    if count_pair(text) <= limit:
        return 0, len(text)
    low, high, best = 0, len(text), (start, end)
    while low <= high:
        radius = (low + high) // 2
        left, right = max(0, start - radius), min(len(text), end + radius)
        if count_pair(text[left:right]) <= limit:
            best, low = (left, right), radius + 1
        else:
            high = radius - 1
    if count_pair(text[best[0]:best[1]]) > limit:
        raise ValueError('Window does not satisfy the actual tokenizer limit')
    return best


class ViewRanker:
    def __init__(self, inner, tokenizer, *, expand):
        self.inner, self.tokenizer, self.expand = inner, tokenizer, expand

    def score(self, query, passages):
        self.original_passages = list(passages)
        self.manifests = []
        views = []
        for passage in passages:
            matches = self.adapter.items_by_content[passage]
            links = [self.adapter.evidence['children'][i] for i in matches]
            unique = {(c['parent_id'], c['start'], c['end']) for c in links}
            if not self.expand or len(unique) != 1:
                views.append(passage)
                continue
            parent_id, anchor_start, anchor_end = next(iter(unique))
            parent = self.adapter.evidence['parents'][parent_id]
            child = parent['text'][anchor_start:anchor_end].strip()
            if not child or not passage.endswith(child):
                raise ValueError('Original content no longer ends in its source child')
            header = passage[:-len(child)]
            def count(text):
                return len(self.tokenizer.encode(query.strip(), (header + text).strip()).ids)
            start, end = anchored_window(parent['text'], anchor_start, anchor_end, count)
            view = (header + parent['text'][start:end]).strip()
            views.append(view)
            self.manifests.append({'child_ids': sorted(matches), 'parent_id': parent_id,
                'start': start, 'end': end, 'pair_tokens': count(parent['text'][start:end]),
                'text_sha256': digest(view.encode())})
        self.scores = self.inner.score(query, views)
        return self.scores


class ViewProbe(TokenProbe):
    def search(self, *args, **kwargs):
        self.items_by_content = {}
        return super().search(*args, **kwargs)

    def _hit_to_item(self, hit, **kwargs):
        item = super()._hit_to_item(hit, **kwargs)
        self.items_by_content.setdefault(item['content'], set()).add(item['stable_id'])
        return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_dotenv()
    source = args.pilot.resolve()
    scope = json.loads((source / 'scope.json').read_bytes())
    questions = {q['id']: q for q in json.loads((source / 'questions.json').read_bytes())}
    saved = [r for r in json.loads((args.diagnostic / 'rows.json').read_bytes()) if r['arm'] == 'token_children']
    reranker = OnnxCrossEncoderReranker(cache_dir=Path(os.environ['PAPERPILOT_RERANKER_CACHE']), allow_download=False)
    reranker.warmup()
    assert reranker.model_sha256 == scope['reranker_model_sha256']
    assert reranker.tokenizer_sha256 == scope['reranker_tokenizer_sha256']
    tokenizer = Tokenizer.from_str(reranker._tokenizer.to_str())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    embedder = FastEmbedder(cache_dir=Path(os.environ['PAPERPILOT_EMBEDDING_CACHE']), allow_download=False)
    assert embedder.cache_identity == scope['embedding_identity']
    adapters = {}
    for arm, expand in [('child_view', False), ('parent_window_view', True)]:
        wrapper = ViewRanker(reranker, tokenizer, expand=expand)
        adapter = ViewProbe(source / 'workspace/token-evidence-pilot',
            SimpleNamespace(embedder=embedder, reranker=wrapper, rerank_stage='final', paper_roots=()))
        wrapper.adapter = adapter
        adapters[arm] = adapter
    save_json(args.output / 'scope.json', {'protocol': 'fixed saved candidate IDs; only reranker passage view changes; same MMR and full-parent packing',
        'source_scope': scope, 'runner_sha256': digest(Path(__file__).read_bytes()), 'api_calls': 0,
        'diagnostic_sha256': digest((args.diagnostic / 'rows.json').read_bytes()),
        'limit': 512, 'question_scope': 'reused historical development set; no heldout claims',
        'candidate_order': 'Preserve original pre-rerank passage order and batch boundaries; assert baseline and treatment receive identical ordered child inputs',
        'earlier_attempts': 'Post-rerank candidate order did not reproduce baseline scores/selection; discarded, not reported as an ablation'})
    rows = []
    for saved_row in saved:
        question = questions[saved_row['case']]
        query = question['question']
        gold = [g for g in question['gold_source_spans'] if g.get('quote')]
        def count_hits(items):
            return sum(covered(g, items, normalizer=normal) for g in gold)
        measured = {}
        for arm, token in adapters.items():
            selected = token.search(query, top_k=8, paper_sha256s=set(question['paper_ids']))
            ids = [h['stable_id'] for h in selected]
            if arm == 'child_view' and ids != saved_row['hit_ids']:
                save_json(args.output / 'baseline-mismatch.json', {'case': saved_row['case'],
                    'expected': saved_row['hit_ids'], 'actual': ids})
                raise ValueError('Baseline selection did not reproduce')
            expanded = token.read_context(ids, max_tokens=2048, paper_sha256s=set(question['paper_ids']))
            measured[arm] = {'child_coverage': count_hits(selected),
                'parent_coverage': count_hits(parents_for(token, selected)),
                'packed_coverage': count_hits(expanded['evidence']), 'hit_ids': ids,
                'context_tokens': expanded['context_tokens'], 'scores': token.reranker.scores}
        assert adapters['child_view'].reranker.original_passages == adapters['parent_window_view'].reranker.original_passages
        assert set(h['stable_id'] for h in adapters['child_view'].candidates) == set(h['stable_id'] for h in adapters['parent_window_view'].candidates)
        rows.append({'case': saved_row['case'], 'gold': len(gold), 'arms': measured,
            'views': adapters['parent_window_view'].reranker.manifests,
            'original_input_order_sha256': digest(json.dumps(adapters['child_view'].reranker.original_passages, ensure_ascii=False).encode())})
        save_json(args.output / 'rows.json', rows)
        print(json.dumps({'case': saved_row['case'], 'done': True}), flush=True)
    summary = {arm: {key: sum(r['arms'][arm][key] for r in rows)
                     for key in ('child_coverage', 'parent_coverage', 'packed_coverage')}
               for arm in ('child_view', 'parent_window_view')}
    save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
