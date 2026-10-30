"""Bounded literal reading context; compact IDs replace repeated span metadata."""
import copy
import json

from .prototype_analysis import spans

POLICY = 'selected-span-sequence-v1'


def bound_view(view, max_bytes):
    """Keep a whole-quote prefix within a serialized byte budget, never alter storage."""
    if type(max_bytes) is not int or max_bytes < 1024:
        raise ValueError('Selected view byte allowance must be at least 1024')
    flat = [(page, pair) for page in view['content']['pages'] for pair in page['quotes']]

    def prefix(count):
        result = copy.deepcopy(view)
        pages = {}
        for page, pair in flat[:count]:
            key = (page['source_uri'], page['page'])
            pages.setdefault(key, {**{k: v for k, v in page.items() if k != 'quotes'}, 'quotes': []})['quotes'].append(pair)
        result['content']['pages'] = list(pages.values())
        meta = result['content']['context_view']
        meta.update(shown_spans=count, omitted_spans=meta['total_spans']-count,
                    complete_span_sequence=count == meta['total_spans'], serialized_byte_limit=max_bytes)
        return result

    lo, hi, best = 0, len(flat), None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = prefix(mid)
        if len(json.dumps(candidate, ensure_ascii=False).encode()) <= max_bytes:
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError('Selected source navigation metadata exceeds its byte allowance')
    return best


def project(identity, artifact, *, max_characters=24000):
    if artifact['kind'] != 'read_literature':
        return artifact
    if type(max_characters) is not int or max_characters < 1:
        raise ValueError('Selected text allowance must be positive')
    catalog = spans(identity, artifact)
    remaining, pages, shown = max_characters, {}, 0
    for span in catalog:
        if len(span['text']) > remaining:
            break
        index = span['chunk_index']
        page = pages.setdefault(index, {'page': span['page'], 'source_uri': span['source_uri'], 'quotes': []})
        page['quotes'].append([span['quote_id'], span['text']])
        remaining -= len(span['text'])
        shown += 1
    view = copy.deepcopy(artifact)
    view['content'] = {'ok': artifact['content'].get('ok'), 'coverage': artifact['content'].get('coverage'),
        'pages': list(pages.values()), 'quote_columns': ['quote_id', 'literal_text'],
        'context_view': {'policy': POLICY, 'raw_artifact_id': identity,
            'representation': 'Original span sequence from selected pages; not a full-paper read',
            'shown_spans': shown, 'total_spans': len(catalog), 'omitted_spans': len(catalog)-shown,
            'text_character_limit': max_characters, 'complete_span_sequence': shown == len(catalog),
            'read_more': 'inspect_quotes query or inspect_artifact reads original omitted content'}}
    return view
