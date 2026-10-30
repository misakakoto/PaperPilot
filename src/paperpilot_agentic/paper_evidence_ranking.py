"""Bound ranker pairs with the ranker's own tokenizer, without shared mutation."""
import json
from pathlib import Path


def ranker_identity(reranker):
    if reranker is None:
        return None
    if hasattr(reranker, '_ensure_loaded'):
        reranker._ensure_loaded()
    if not getattr(reranker, 'model_sha256', None) or not getattr(reranker, 'tokenizer_sha256', None):
        raise ValueError('Durable evidence retrieval requires pinned reranker weights and tokenizer')
    from .paper_structure import digest
    return {'model': reranker.name, 'model_sha256': reranker.model_sha256,
        'tokenizer_sha256': reranker.tokenizer_sha256, 'pair_limit': reranker._max_length,
        'batch_size': reranker._batch_size, 'view_policy_sha256': digest(Path(__file__).read_bytes())}


def bounded_rerank(reranker, query, passages):
    # The local production adapter exposes its loaded tokenizer. Test adapters can
    # supply pair_count/max_length directly, keeping this internal seam testable.
    if hasattr(reranker, '_ensure_loaded'):
        reranker._ensure_loaded()
    count = getattr(reranker, 'pair_count', None)
    limit = getattr(reranker, '_max_length', getattr(reranker, 'max_length', None))
    if count is None:
        from tokenizers import Tokenizer
        original = getattr(reranker, '_tokenizer', None)
        if original is None:
            raise ValueError('Evidence reranker must expose its actual pair tokenizer')
        tokenizer = Tokenizer.from_str(original.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        def count(q, p):
            return len(tokenizer.encode(q, p).ids)
    if type(limit) is not int or limit < 32:
        raise ValueError('Evidence reranker must expose a finite pair limit')

    def fit(text, fits):
        if fits(text):
            return text
        lo, hi, end = 1, len(text), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(text[:mid]):
                end, lo = mid, mid + 1
            else:
                hi = mid - 1
        if end == 0:
            raise ValueError('Reranker required content cannot fit')
        return text[:end]

    q = fit(query, lambda s: count(s, 'evidence') <= limit // 2)
    bounded = [fit(p, lambda s: count(q, s) <= limit) for p in passages]
    counts = [count(q, p) for p in bounded]
    if any(c > limit for c in counts):
        raise ValueError('Reranker pair exceeded its actual input limit')
    view = {'query_char_range': [0, len(q)], 'query_characters': len(query),
            'passage_char_ranges': [[0, len(p)] for p in bounded], 'pair_tokens': counts,
            'pair_limit': limit, 'truncated': q != query or bounded != passages,
            'model': getattr(reranker, 'name', type(reranker).__name__),
            'model_sha256': getattr(reranker, 'model_sha256', None),
            'tokenizer_sha256': getattr(reranker, 'tokenizer_sha256', None)}
    # Record the exact bounded input identity separately from the original sources.
    from .paper_structure import digest
    view['input_sha256'] = digest(json.dumps([q, bounded], ensure_ascii=False).encode())
    return reranker.score(q, bounded), view
