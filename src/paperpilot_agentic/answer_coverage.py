"""Track requested subparts after filtering; no automatic factual PASS."""
from copy import deepcopy
import re

from .numeric_scope import table_references, check_bindings

VERSION = 'retained-request-coverage-v4-host-presentation'


def request_contract(question, catalog, targets=None):
    from .request_obligations import mechanism_only_request
    mechanism_only = mechanism_only_request(question)
    requested = table_references(question)
    numerical = bool(re.search(r'消融|数值|实验结果|消融结果|ablation|numeric|reported result', question, re.I))
    tables = []
    if not mechanism_only and requested and numerical:
        for target in targets or []:
            if any(table_references(target['label'])[0]['table'] == r['table'] for r in requested):
                tables.append(dict(target))
        if not tables:
            for ref in requested:
                owners = {row['paper_sha256'] for row in catalog
                          if table_references(row['table']) and
                          table_references(row['table'])[0]['table'] == ref['table']}
                tables.append({'paper_sha256': next(iter(owners)) if len(owners) == 1 else None,
                               'label': ref['text']})
    return {'version': VERSION, 'original_question': question, 'mechanism_only': mechanism_only,
            'table_requirements': tables,
            'coverage_authority': 'explicit_table_shapes_and_model_semantic_subparts_not_ground_truth'}


def source_presence(index_rows, targets):
    result = []
    for target in targets:
        refs = table_references(target['label'])
        found = []
        for row in index_rows:
            uri = str(row.get('source_uri', ''))
            if not uri.startswith(f"paper://{target['paper_sha256']}/"):
                continue
            for heading in re.finditer(r'(?m)^\s*(?:TABLE|Table|表)\s*([IVXLCDM]+|\d+)\b', row.get('content', '')):
                labels = table_references(heading[0])
                if refs and labels and refs[0]['table'] == labels[0]['table']:
                    found.append(row)
        result.append({**target, 'status': 'PRESENT' if found else 'NOT_LOCATED',
                       'page': found[0].get('page') if found else None,
                       'source_uri': found[0].get('source_uri') if found else None})
    return result


def indexed_table_inventory(store, outline, paper_sha256, requested_targets=None):
    """Bounded local index scan. A missing heading is NOT_LOCATED, never ABSENT."""
    if not callable(getattr(store, 'get_chunks', None)):
        return {'rows': [{**t, 'status': 'NOT_LOCATED', 'page': None, 'source_uri': None}
                         for t in requested_targets or [] if t['paper_sha256'] == paper_sha256],
                'status': 'INDEX_UNAVAILABLE', 'absence_is_not_established': True}
    ids = list(dict.fromkeys(cid for section in outline for cid in section.get('chunk_ids', [])))
    chunks = store.get_chunks(ids[:128])
    source_rows, targets, used = [], [], 0
    for hit in chunks:
        if hit.paper_sha256 != paper_sha256:
            raise ValueError('Source inventory outside selected paper')
        text = hit.text
        if used + len(text) > 300000:
            break
        used += len(text)
        source_rows.append({'source_uri': f'paper://{paper_sha256}/{hit.stable_id}',
                            'content': text, 'page': hit.page_start})
        for heading in re.finditer(r'(?m)^\s*(?:TABLE|Table|表)\s*([IVXLCDM]+|\d+)\b', text):
            targets.append({'paper_sha256': paper_sha256, 'label': 'Table ' + heading[1]})
    unique = {(t['paper_sha256'], t['label']): t for t in targets}
    for target in requested_targets or []:
        if target['paper_sha256'] == paper_sha256:
            unique[(paper_sha256, target['label'])] = {
                'paper_sha256': paper_sha256, 'label': target['label']}
    return {'rows': source_presence(source_rows, list(unique.values())),
            'scanned_chunks': len(source_rows), 'scanned_chars': used,
            'unscanned_chunk_count': len(ids) - len(source_rows), 'absence_is_not_established': True}


def validate_requirement_assessments(audit, draft):
    requirements = audit.get('requirement_assessments', [])
    owned = {fact['id'] for fact in draft['facts']}
    if not isinstance(requirements, list) or len(requirements) > 12:
        raise ValueError('Expected at most 12 requested semantic subparts')
    for row in requirements:
        if (not isinstance(row, dict) or not isinstance(row.get('requirement'), str) or
                not row['requirement'].strip() or not isinstance(row.get('fact_ids'), list) or
                any(not isinstance(fid, str) or fid not in owned for fid in row['fact_ids']) or
                not isinstance(row.get('gap', ''), str)):
            raise ValueError('Requirement assessment must reference original draft fact IDs')
        indices = row.get('limitation_indices', [])
        if (not isinstance(indices, list) or len(indices) > 12 or
                any(type(i) is not int or not 0 <= i < len(draft.get('limitations', [])) for i in indices) or
                len(set(indices)) != len(indices)):
            raise ValueError('Requirement limitation_indices must reference unique owned draft limitations')


def apply_current_scope(draft, audit, contract):
    effective = deepcopy(audit)
    changed = []
    if contract['mechanism_only']:
        facts = {f['id']: f for f in draft['facts']}
        for assessment in effective['assessments']:
            fact = facts[assessment['fact_id']]
            if table_references(fact['text']) and re.search(r'\d+(?:\.\d+)?[%％]', fact['text']):
                assessment['relevance'] = 'TANGENTIAL'
                changed.append(fact['id'])
    return effective, {'version': VERSION, 'unrequested_result_fact_ids': changed}


def apply_final_coverage(draft, audit, evidence, catalog, contract, *, request_history=None,
                         presentation=None):
    effective = deepcopy(audit)
    facts = {f['id']: f for f in draft['facts']}
    retained = {a['fact_id'] for a in effective['assessments'] if a['verdict'] == 'SUPPORTED'
                and a.get('relevance') != 'TANGENTIAL'}
    requirements = []
    from .answer_completion import retained_boundary_indices
    boundaries = retained_boundary_indices(draft, audit, question=contract['original_question'],
                                           request_history=request_history)
    for target in contract['table_requirements']:
        label = table_references(target['label'])[0]['table']
        rows = [r for r in catalog if r['paper_sha256'] == target['paper_sha256'] and
                table_references(r['table']) and table_references(r['table'])[0]['table'] == label]
        source_ids = {r['evidence_id'] for r in rows}
        # Compatibility with existing source-bound table transcriptions that
        # predate the structured row contract. Never use unrelated sources.
        for source in evidence:
            if (target['paper_sha256'] and str(source.get('source_uri', '')).startswith(
                    f"paper://{target['paper_sha256']}/") and
                    any(ref['table'] == label for ref in table_references(source['content']))):
                source_ids.add(source['evidence_id'])
        hits = []
        for fid in retained:
            fact = facts[fid]
            owned_refs = [ref for ref in fact.get('row_refs', []) if ref['evidence_id'] in source_ids]
            bindings = check_bindings(fact, [e for e in evidence if e['evidence_id'] in source_ids])['checks']
            if owned_refs or any(c['kind'] in {'table_cell', 'table_row'} and c['status'] == 'MATCH' for c in bindings):
                hits.append(fid)
        requirements.append({'requirement': f"报告 {target['label']} 的来源绑定消融结果",
            'paper_sha256': target['paper_sha256'], 'table': target['label'],
            'retained_fact_ids': sorted(hits), 'evidence_ids': sorted(source_ids),
            'status': 'COVERED_SHAPE_ONLY' if hits else 'MISSING'})
    for row in audit.get('requirement_assessments', []):
        hits = sorted(retained & set(row['fact_ids']))
        lost = sorted(set(row['fact_ids']) - retained)
        boundary_hits = sorted(boundaries.intersection(row.get('limitation_indices', [])))
        from .request_obligations import conditional_evidence_policy
        policy_only = conditional_evidence_policy(row.get('request_quote'))
        from .answer_layout import satisfied_layout_quote
        layout_only = satisfied_layout_quote(row.get('request_quote'), contract['original_question'],
                                             presentation, draft, audit)
        requirements.append({'requirement': row['requirement'], 'retained_fact_ids': hits,
            'withheld_fact_ids': lost, 'retained_limitation_indices': boundary_hits,
            'evidence_ids': sorted({eid for fid in hits for eid in facts[fid]['evidence_ids']}),
            'status': ('UNRESOLVED' if row.get('gap', '').strip() else
                       'CONDITIONAL_POLICY' if policy_only else
                       'COVERED_HOST_PRESENTATION' if layout_only and not lost else
                       'MISSING' if lost else
                       'COVERED_MODEL_ONLY' if hits or boundary_hits else 'MISSING'),
            'gap': row.get('gap', '')})
    missing = [r for r in requirements if r['status'] in {'MISSING', 'UNRESOLVED'}]
    if missing:
        effective['coverage'] = 'PARTIAL'
        # A declared gap is already projected to the original request for the
        # user. Add host diagnostics only for newly missing retained content.
        effective['missing_aspects'] = list(dict.fromkeys(effective.get('missing_aspects', []) +
            ['当前最终回答尚未覆盖：' + row['requirement'] for row in missing if row['status'] == 'MISSING']))
    report = {'version': VERSION, 'requirements': requirements,
        'retained_fact_ids': sorted(retained), 'coverage': effective['coverage'],
        'retained_limitation_indices': sorted({i for row in requirements
            for i in row.get('retained_limitation_indices', [])}),
        'independent_source_review': 'PENDING', 'semantic_map_status':
        'MODEL_SUBPARTS_RECORDED' if audit.get('requirement_assessments') else 'LEGACY_MODEL_COVERAGE_ONLY'}
    return effective, report
