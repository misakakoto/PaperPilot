"""One truthful projection of exact, ranged and unknown chunk locations."""


def chunk_position(hit):
    start, end = getattr(hit, 'page_start', None), getattr(hit, 'page_end', None)
    known = type(start) is int and type(end) is int and 0 < start <= end
    exact = known and start == end
    result = {'page': start if exact else None,
            'provenance': {'page_precision': 'EXACT' if exact else 'RANGE' if known else 'UNKNOWN',
                           'page_range': [start, end] if known else None,
                           'section_path': [hit.section]}}
    if getattr(hit, 'source_spans', None) is not None:
        result['provenance']['source_segments'] = hit.source_spans
    return result
