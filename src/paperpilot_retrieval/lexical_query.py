"""Versioned lexical-only query cleanup; original questions remain untouched."""
import re

FUNCTION_WORDS = frozenset('a an the am is are was were be been being do does did have has had '
    'what which who whom whose how where when why it its they them their theirs '
    'he she his her hers we us our ours you your yours this that these those '
    'of for to in on at by with from as and or'.split())


def content_query(query: str) -> str:
    # Keep this v1 policy identical to the frozen dev/test ablation.
    filtered = re.sub(r'\b[A-Za-z]+\b', lambda m: '' if (
        m.group().lower() in FUNCTION_WORDS and not m.group().isupper()) else m.group(), query)
    return ' '.join(filtered.split()) if re.search(r'[A-Za-z0-9\u4e00-\u9fff]', filtered) else query
