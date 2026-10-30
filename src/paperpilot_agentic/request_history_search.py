"""Deterministic lexical selection of committed request originals.

Ranking is a locator, never proof that unreturned history is absent. The caller
owns ancestry/version checks and returns whole originals, not generated summaries.
"""
from collections import Counter
import math
import re


def _tokens(text):
    words = re.findall(r'[a-z0-9]+(?:[-_.][a-z0-9]+)*', text.casefold())
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        words.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return words


def history_source_ids(state, task_ids, query=None):
    """Chronological task events, or BM25-ranked lexical hits with stable ties."""
    ids = [e['source_id'] for e in state['events']
           if any(op['task_id'] in task_ids for op in e['operations'])]
    if query is None:
        return ids
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError('History query must contain 1 to 500 characters')
    terms = set(_tokens(query))
    docs = [Counter(_tokens(state['sources'][sid])) for sid in ids]
    if not docs or not terms:
        return []
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(lengths) or 1
    idf = {t: math.log(1 + (len(docs) - sum(t in d for d in docs) + .5)
                      / (sum(t in d for d in docs) + .5)) for t in terms}
    scores = []
    for i, (doc, length) in enumerate(zip(docs, lengths)):
        score = sum(idf[t] * doc[t] * 2.2 /
                    (doc[t] + 1.2 * (.25 + .75 * length / average))
                    for t in terms if doc[t])
        if score > 0:
            scores.append((score, i, ids[i]))
    return [sid for _, _, sid in sorted(scores, key=lambda x: (-x[0], x[1]))]
