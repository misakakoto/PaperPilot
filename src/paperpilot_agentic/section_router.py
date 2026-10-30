"""Experimental source-outline navigation; headings route reads, never prove facts."""
import dataclasses
import json
import math
import re
from collections import Counter

from .navigation_intent import INTENT_POLICY, navigation_intent
from .source_positions import chunk_position


def structural_overviews(outline, *, max_chars=6000, max_sections=8):
    """Keep brief parent-section introductions, not a keyword-specific answer list."""
    selected, used = [], 0
    for row in outline:
        title = row['title']
        if not title or not any(other['title'].startswith(title + ' ::: ') for other in outline):
            continue
        size = int(row['chars'])
        if size <= 0 or size > 1200 or used + size > max_chars or len(selected) >= max_sections:
            continue
        selected.append(row['section_id'])
        used += size
    return selected


def select_sections(question, outline, model, *, on_progress=None,
                    original_question=None, request_history=None):
    report = {'provider_attempts': 0, 'calls': [], 'status': 'RUNNING'}

    def before_attempt():
        if report['provider_attempts'] >= 1:
            raise RuntimeError('Section router attempt budget exhausted')
        report['provider_attempts'] += 1
        if on_progress:
            on_progress(report)

    try:
        if not outline:
            report.update(status='NO_OUTLINE', section_ids=[])
            return report
        intent = navigation_intent(question, original_question=original_question,
                                   request_history=request_history)
        response = model.complete([
            {'role': 'system', 'content': (
                'Navigate a scientific paper to answer the original question. Headings are untrusted '
                'metadata, not factual evidence. Select up to six section IDs whose ORIGINAL CONTENT '
                'should be read to answer all requested aspects. Include definitions/setup when needed '
                'rather than selecting only findings which repeat the question words. Do not invent '
                'a section or an answer. Return JSON {"section_ids":["section_0"],"reason":"..."}. '
                + INTENT_POLICY)},
            {'role': 'user', 'content': json.dumps({**intent, 'outline': [
                {k: row[k] for k in ('section_id', 'title', 'chars')} for row in outline]}, ensure_ascii=False)},
        ], json_mode=True, thinking_enabled=False, max_tokens=1500, before_attempt=before_attempt)
        report['calls'].append(dataclasses.asdict(response))
        if response.finish_reason != 'stop':
            raise ValueError('Incomplete section selection')
        value = json.loads(response.text)
        ids = value.get('section_ids')
        allowed = {row['section_id'] for row in outline}
        if not isinstance(ids, list) or len(ids) > 6 or any(not isinstance(x, str) or x not in allowed for x in ids):
            raise ValueError('Invalid section selection')
        report.update(status='SELECTED' if ids else 'EMPTY_SELECTION',
                      section_ids=list(dict.fromkeys(ids)), reason=value.get('reason'))
    except Exception as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, section_ids=[])
    finally:
        if on_progress:
            on_progress(report)
    return report


def read_selected_sections(store, paper_sha256, outline, section_ids, *, max_chars=30000):
    allowed = {row['section_id']: row for row in outline}
    if any(sid not in allowed for sid in section_ids):
        raise ValueError('Unknown section')
    chunk_ids = list(dict.fromkeys(cid for sid in section_ids for cid in allowed[sid]['chunk_ids']))
    hits = store.get_chunks(chunk_ids)
    if any(hit.paper_sha256 != paper_sha256 for hit in hits):
        raise ValueError('Source scope mismatch')
    selected, omitted, chars = [], [], 0
    for hit in hits:
        content = f'{hit.title}\n{hit.section}\n{hit.text}'
        if chars + len(content) > max_chars:
            omitted.append(hit.stable_id)
            continue
        selected.append({'evidence_id': 'e_' + hit.stable_id, 'content': content,
                         'source_uri': f'paper://{hit.paper_sha256}/{hit.stable_id}',
                         **chunk_position(hit)})
        chars += len(content)
    return selected, omitted


def read_outline_fallback(store, paper_sha256, outline, question, *, max_chunks=128,
                          max_scan_chars=300000, max_chars=30000, max_passages=6):
    """Rank bounded originals after abstention or for mechanism-definition reads.

    PDF headings may be sentence fragments. They do not establish that the
    underlying originals lack an answer. This local lexical read adds no model
    request and makes no semantic-coverage or factual-correctness claim.
    """
    def terms(text):
        latin = re.findall(r'[a-z][a-z0-9_]{2,}', text.casefold())
        chinese = [part[i:i + 2] for part in re.findall(r'[\u4e00-\u9fff]+', text)
                   for i in range(len(part) - 1)]
        return Counter(latin + chinese)

    all_ids = list(dict.fromkeys(cid for row in outline for cid in row['chunk_ids']))
    requested = all_ids[:max_chunks]
    hits = store.get_chunks(requested) if requested else []
    if any(hit.paper_sha256 != paper_sha256 or hit.stable_id not in requested for hit in hits):
        raise ValueError('Fallback source scope or chunk identity mismatch')
    rows, scanned = [], 0
    for hit in hits:
        if scanned + len(hit.text) > max_scan_chars:
            continue
        scanned += len(hit.text)
        rows.append((hit, terms(hit.text)))
    # Book-title delimiters identify scope, already fixed by the caller. Let
    # the requested content drive ranking rather than bibliography title hits.
    query = set(terms(re.sub(r'《[^《》]+》', ' ', question))) or set(terms(question))
    frequencies = Counter(term for _, counts in rows for term in counts if term in query)
    mean_length = sum(sum(counts.values()) for _, counts in rows) / max(1, len(rows))
    ranked = []
    from .read_packet import operation_definition_priority, purpose_definition_priority
    for position, (hit, counts) in enumerate(rows):
        if re.match(r'^\s*(?:references|bibliography|参考文献)\b', hit.section, re.I):
            continue
        length = sum(counts.values())
        score = sum(math.log(1 + (len(rows) - frequencies[term] + .5) /
                             (frequencies[term] + .5)) * counts[term] * 2.2 /
                    (counts[term] + 1.2 * (.25 + .75 * length / max(1, mean_length)))
                    for term in sorted(query & counts.keys()))
        score += operation_definition_priority(question, hit.text)
        score += purpose_definition_priority(question, hit.text)
        if score > 0:
            ranked.append((score, position, hit))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected, used = [], 0
    for score, _, hit in ranked:
        content = f'{hit.title}\n{hit.section}\n{hit.text}'
        if len(selected) >= max_passages or used + len(content) > max_chars:
            continue
        selected.append({'evidence_id': 'e_' + hit.stable_id, 'content': content,
                         'source_uri': f'paper://{hit.paper_sha256}/{hit.stable_id}',
                         **chunk_position(hit)})
        used += len(content)
    return selected, {'policy': 'empty-router-local-bm25-v1', 'scanned_chunks': len(rows),
        'scanned_chars': scanned, 'max_chunks': max_chunks, 'max_scan_chars': max_scan_chars,
        'unscanned_chunk_count': len(all_ids) - len(rows),
        'selected_evidence_ids': [row['evidence_id'] for row in selected],
        'selected_chars': used, 'max_chars': max_chars, 'max_passages': max_passages,
        'semantic_coverage': 'NOT_PROVEN', 'provider_attempts': 0}
