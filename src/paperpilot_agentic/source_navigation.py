"""Scoped exact object/section navigation over existing original chunks."""
import re

from .adapters import PaperPilotKnowledgeSearchAdapter


def locate_requested_objects(store, paper_sha256, outline, question, *, max_chunks=600):
    labels = re.findall(r'(?:\b(?:eq(?:uation)?|fig(?:ure)?|table)\.?|公式|式|图|表)\s*[（(]?\s*(\d+|[IVXLCDM]+)\b',
                        question, re.I)
    wants_limits = bool(re.search(r'limitations?|限制|局限', question, re.I))
    if not labels and not wants_limits:
        return []
    navigator = getattr(store, 'navigator_adapter', None)
    if navigator is not None:
        queries = list(dict.fromkeys(q for label in labels for q in
            ('(' + label + ')', 'Table ' + label, 'Eq. ' + label)))
        if wants_limits:
            queries.append('limitations')
        matches = []
        for query in queries[:6]:
            matches.extend(navigator.find_text(paper_sha256, query)['matches'])
        rows = navigator.read_matches(paper_sha256, matches)
        for row in rows:
            row['provenance']['navigation_anchor'] = {'question':question,
                'policy':'scoped-rg-literal-v1',
                'assurance':'LOCATION_MATCH_ONLY; original read, not factual correctness'}
        return rows
    requested = list(dict.fromkeys(cid for row in outline for cid in row['chunk_ids']))[:max_chunks]
    candidates = store.get_chunks(requested)
    patterns = [re.compile(r'(?:\b(?:eq(?:uation)?|fig(?:ure)?|table)\.?\s*\(?\s*' +
        re.escape(label) + r'\b|[（(]\s*' + re.escape(label) + r'\s*[)）]|\\tag\{' + re.escape(label) + r'\})', re.I)
        for label in labels]
    ranked = []
    for position, hit in enumerate(candidates):
        if hit.paper_sha256 != paper_sha256 or hit.stable_id not in requested:
            raise ValueError('Object navigation escaped scoped source identities')
        if re.search(r'references|bibliography|参考文献', hit.section, re.I):
            continue
        content = hit.section + '\n' + hit.text
        label_hits = sum(bool(pattern.search(content)) for pattern in patterns)
        limitation = wants_limits and bool(re.search(r'\blimitations?\b|局限|限制', content, re.I))
        if not label_hits and not limitation:
            continue
        score = label_hits + 2 * int(limitation)
        ranked.append((-score, position, hit))
    results = []
    for rank, (_, _, hit) in enumerate(sorted(ranked, key=lambda r: r[:2])[:3], 1):
        item = PaperPilotKnowledgeSearchAdapter._hit_to_item(hit, rank=rank, memory_ids=[])
        item['evidence_id'] = 'e_' + hit.stable_id
        item['provenance']['navigation_anchor'] = {
            'question': question, 'policy': 'scoped-object-literal-v1',
            'assurance': 'LOCATION_MATCH_ONLY; not proof of relevance or correctness'}
        results.append(item)
    return results
