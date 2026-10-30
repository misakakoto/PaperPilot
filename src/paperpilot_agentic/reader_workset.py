"""Bound a multi-paper reader window without summarizing source evidence."""
import json
from collections import OrderedDict


def revision_workset(working, previous_context):
    """Revise against the same sources, without reapplying retrieval allocation.

    This does not grant tokens: each actual Writer/Auditor request still passes
    the persistent runtime's input/output budget guard before provider dispatch.
    """
    identifiers = [row['evidence_id'] for row in working]
    if identifiers != previous_context['selected_evidence_ids']:
        raise ValueError('Revision sources differ from the recorded workset')
    return list(working), {**previous_context, 'policy': 'revision-preserve-source-window-v1',
                          'budget_enforcement': 'per-request-input-output-runtime-guard'}


def source_aware_workset(evidence, *, max_bytes):
    """Round-robin already ranked candidates across papers; never cut a passage.

    Candidates must be durably saved by the caller before selection. This does
    not establish relevance or semantic coverage; omitted sources remain reloadable.
    """
    groups = OrderedDict()
    for row in evidence:
        uri = str(row.get('source_uri') or '')
        paper = uri.split('/')[2] if uri.startswith('paper://') else uri or row['evidence_id']
        groups.setdefault(paper, []).append(row)
    selected, omitted, used, covered = [], [], 2, set()
    while any(groups.values()):
        for paper, queue in groups.items():
            if not queue:
                continue
            row = queue.pop(0)
            view = {k: row.get(k) for k in ('evidence_id', 'source_uri', 'page', 'content')}
            size = len(json.dumps(view, ensure_ascii=False).encode('utf-8')) + 2
            if used + size <= max_bytes:
                selected.append(row)
                used += size
                covered.add(paper)
            else:
                omitted.append(row['evidence_id'])
    return selected, {
        'policy': 'whole-passage-source-round-robin-v1',
        'candidate_count': len(evidence), 'selected_count': len(selected),
        'selected_evidence_ids': [row['evidence_id'] for row in selected],
        'omitted_evidence_ids': omitted, 'unrepresented_sources': sorted(set(groups) - covered),
        'context_upper_bytes': used, 'max_bytes': max_bytes,
        'summarized': False, 'semantic_coverage': 'NOT_PROVEN',
    }
