"""Source-owned row references and necessary checks for single-factor comparisons.

Matching a parsed row never establishes factual truth or statistical causality.
Legacy prose is checked conservatively; no statement is rewritten or upgraded.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

from .structured_extraction import ParsedArtifact

VERSION = 'source-row-attribution-v4-numeric-and-literal-cells'


def _number(value):
    try:
        result = Decimal(str(value).strip().rstrip('%％'))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def _numbers(text):
    return [_number(m[0]) for m in re.finditer(
        r'(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_.])', text)]


def row_catalog(evidence):
    """Give each validated source row an ID bound to crop, parser and config."""
    result = []
    for source in evidence:
        payload = (source.get('provenance') or {}).get('structured_extraction')
        if not payload:
            continue
        try:
            item = ParsedArtifact.model_validate(payload)
        except ValueError:
            continue
        if (item.kind != 'table' or item.parse_status != 'PARSED' or
                item.validation_status != 'VALIDATED' or source.get('source_uri') !=
                f'paper://{item.source.paper_sha256}/layout/{item.source.object_id}'):
            continue
        # Non-evidentiary notes must not change IDs, including after a legacy
        # artifact has been projected into the current typed observation.
        identity_payload = item.model_dump(mode='json', exclude={'parser_notes'})
        identity = hashlib.sha256(json.dumps(identity_payload,
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        components = sorted({name for row in item.table_rows
                             for name in (row.get('checked_columns') or [])})
        from .numeric_scope import metric_references
        metrics = [h for h in item.column_headers if metric_references(h)]
        for index, row in enumerate(item.table_rows):
            cells = row.get('cells')
            if not isinstance(cells, dict):
                continue
            result.append({'evidence_id': source['evidence_id'],
                'row_id': f'row_{identity[:24]}_{index}', 'artifact_sha256': identity,
                'paper_sha256': item.source.paper_sha256, 'table': item.source.label,
                'row_index': index, 'row_label': row.get('row_label', ''),
                'component_columns': components, 'checked_columns': row.get('checked_columns'),
                'metric_headers': metrics,
                'cells': {str(k): str(v) for k, v in cells.items()},
                'factual_review': 'NOT_REVIEWED',
                **({'source_review': source['provenance']['reviewed_source']}
                   if source.get('provenance', {}).get('reviewed_source') else {})})
    return result


def validate_row_refs(draft, catalog):
    owned = {(row['evidence_id'], row['row_id']): row for row in catalog}
    for fact in draft['facts']:
        refs = fact.get('row_refs', [])
        if not isinstance(refs, list) or len(refs) > (2 if fact.get('kind') == 'INFERENCE' else 1):
            raise ValueError('A source fact describes one table configuration; comparison has at most two endpoints')
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'evidence_id', 'row_id', 'metrics'}:
                raise ValueError('row_refs require evidence_id, host row_id and metrics')
            if not isinstance(ref['evidence_id'], str) or not isinstance(ref['row_id'], str):
                raise ValueError('Row identity must be a string')
            row = owned.get((ref['evidence_id'], ref['row_id']))
            metrics = ref['metrics']
            if (not row or ref['evidence_id'] not in fact['evidence_ids'] or
                    not isinstance(metrics, list) or not metrics or
                    any(not isinstance(metric, str) or metric not in row['cells'] for metric in metrics)):
                raise ValueError('Row or selected column outside cited host catalog')
            for metric in metrics:
                raw = row['cells'][metric]
                if _number(raw) is None and (
                        fact.get('kind', 'SOURCE_FACT') != 'SOURCE_FACT'
                        or not isinstance(raw, str) or not raw.strip()
                        or f'{metric}={raw}' not in fact['text']):
                    raise ValueError('Literal row reference requires the exact source column=value binding')


def _endpoints(fact, catalog, facts):
    refs = fact.get('row_refs') or [ref for dep in fact.get('depends_on', [])
                                  for ref in facts.get(dep, {}).get('row_refs', [])]
    selected = [row for ref in refs for row in catalog
                if (row['evidence_id'], row['row_id']) == (ref['evidence_id'], ref['row_id'])]
    if not selected:
        # Only unique same-metric endpoints in the fact's own cited tables.
        from .numeric_scope import metric_references, _metric_name
        metrics = {item['metric'] for item in metric_references(fact['text'])}
        numbers = _numbers(fact['text'])
        selected = [row for row in catalog if row['evidence_id'] in fact['evidence_ids'] and
                    any(_metric_name(metric) in metrics and _number(value) in numbers
                        for metric, value in row['cells'].items())]
    selected = list({(r['evidence_id'], r['row_id']): r for r in selected}.values())
    numbers = _numbers(fact['text'])
    def order(row):
        return min([numbers.index(_number(v)) for v in row['cells'].values()
                    if _number(v) in numbers] or [len(numbers)])
    return sorted(selected, key=order)


def _descriptive_group_configuration(text, cue, components):
    """Recognize an explicit multi-component configuration, not its causal effect.

    Parentheses are presentation, not scientific semantics. Match the bounded
    group preceding a ranking; causal or ambiguous wording keeps the old guard.
    """
    if not re.search(r'(?:同时|共同|联合|\b(?:simultaneously|jointly|together))\s*$',
                     text[:cue.start()], re.I):
        return set()
    ranking = re.search(r'优于|高于|更好|最佳|最优|\b(?:better|higher|best|outperform\w*)\b',
                        text[cue.end():], re.I)
    if not ranking:
        return set()
    label = text[cue.start():cue.end() + ranking.start()]
    names = {name for name in components if re.search(
        rf'(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])', label, re.I)}
    causal = re.search(r'移除|去除|去掉|加入|提升|提高|增加|减少|下降|降低|改善|导致|'
                      r'带来|引起|造成|变化|改变|使|归因|由于|因而|'
                      r'\b(?:remov\w*|add\w*|improv\w*|increas\w*|decreas\w*|'
                      r'reduc\w*|chang\w*|caus\w*|effect|attribut\w*|due\s+to|lead(?:s|ing)?|led)\b',
                      text, re.I)
    return names if len(names) >= 2 and not causal else set()


def _group_scope_error(rows, refs, names):
    """Only scope/configuration checks; a MATCH never proves a prose ranking."""
    if len(rows) < 2 or not refs or any(r.get('checked_columns') is None for r in rows):
        return {'status': 'UNRESOLVED', 'reason': 'group_comparison_scope_unresolved'}
    identity = {(r.get('paper_sha256'), r.get('artifact_sha256'), r.get('table')) for r in rows}
    metric_sets = {tuple(sorted(ref['metrics'])) for ref in refs}
    from .numeric_scope import _metric_name
    components = {c.casefold() for row in rows for c in row['component_columns']}
    headers = {key for row in rows for key in row['cells']
               if key.casefold() not in components and _metric_name(key) is None}
    same_conditions = all(len({r['cells'].get(key) for r in rows}) == 1 for key in headers)
    group_present = any({n.casefold() for n in r['checked_columns']} == names for r in rows)
    if (len(identity) != 1 or any(not item for item in next(iter(identity))) or
            len(metric_sets) != 1 or not same_conditions or not group_present):
        return {'status': 'MISMATCH', 'reason': 'group_comparison_scope_or_configuration_mismatch',
                'row_ids': [r['row_id'] for r in rows]}
    return None


def _single_factor(fact, rows, refs):
    text = fact['text']
    cue = re.search(r'(移除|去除|去掉|加入|启用|remov\w*|add\w*|enabl\w*)\s*([A-Za-z][A-Za-z0-9_-]*)', text, re.I)
    if not cue:
        return None
    target = cue[2].casefold()
    components = {c.casefold() for row in rows for c in row['component_columns']}
    group = _descriptive_group_configuration(text, cue, components)
    if fact.get('kind') == 'INFERENCE' and group:
        return _group_scope_error(rows, refs, group)
    if target not in components:
        return {'status': 'UNRESOLVED', 'reason': 'single_factor_component_not_located'}
    comparative = re.search(r'从.{0,120}(?:到|至)|提升|提高|增加|减少|下降|降低|改变|导致|'
                            r'带来|影响|发生变化|变化(?:了|至|到|为|幅度)|改善|'
                            r'\b(?:improv(?:e[sd]?|ing|ement)|increas(?:e[sd]?|ing)|'
                            r'decreas(?:e[sd]?|ing)|reduc(?:e[sd]?|ing|tion)|chang(?:e[sd]?|ing)|'
                            r'caus(?:e[sd]?|ing)|lead(?:s|ing)?|led|higher|lower|better|worse)\b',
                            text, re.I)
    if (fact.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT' and len(rows) == 1
            and rows[0]['checked_columns'] is not None and not comparative):
        checked = {name.casefold() for name in rows[0]['checked_columns']}
        removing = bool(re.match(r'移除|去除|去掉|remov', cue[1], re.I))
        only = bool(re.search(r'仅|只|\bonly\b', text, re.I))
        valid = target not in checked if removing else target in checked
        if only and not removing:
            valid = valid and checked == {target}
        if not valid:
            return {'status': 'MISMATCH', 'reason': 'single_configuration_component_mismatch'}
        # Leave the actual row values to the independent source-row check.
        return None
    if len(rows) != 2 or any(row['checked_columns'] is None for row in rows):
        return {'status': 'UNRESOLVED', 'reason': 'comparison_endpoints_not_unique'}
    left, right = rows
    a, b = ({name.casefold() for name in row['checked_columns']} for row in rows)
    changed = sorted(a ^ b)
    from .numeric_scope import _metric_name
    metric_headers = {metric for ref in refs for metric in ref['metrics']}
    condition_headers = {key for row in rows for key in row['cells']
                         if key.casefold() not in components and key not in metric_headers
                         and _metric_name(key) is None}
    different = sorted(key for key in condition_headers
                       if left['cells'].get(key) != right['cells'].get(key))
    remove = bool(re.match(r'移除|去除|去掉|remov', cue[1], re.I))
    direction = target in a and target not in b if remove else target not in a and target in b
    same_metrics = not refs or len({tuple(sorted(ref['metrics'])) for ref in refs}) == 1
    valid = (same_metrics and left['artifact_sha256'] == right['artifact_sha256'] and
             changed == [target] and not different and direction)
    return {'status': 'MATCH' if valid else 'MISMATCH',
            'reason': 'single_factor_holds' if valid else 'other_conditions_or_direction_changed',
            'changed_components': changed, 'changed_conditions': different,
            'row_ids': [row['row_id'] for row in rows]}


def apply_row_guard(draft, audit, catalog, *, mode='enforce'):
    if mode not in {'off', 'shadow', 'enforce'}:
        raise ValueError('Unknown row binding mode')
    effective = deepcopy(audit)
    diagnostics = {'version': VERSION, 'mode': mode, 'facts': [], 'changed_fact_ids': []}
    if mode == 'off':
        return effective, diagnostics
    facts = {f['id']: f for f in draft['facts']}
    owned = {(r['evidence_id'], r['row_id']): r for r in catalog}
    for assessment in effective['assessments']:
        fact = facts[assessment['fact_id']]
        refs = fact.get('row_refs') or [ref for dep in fact.get('depends_on', [])
                                      for ref in facts.get(dep, {}).get('row_refs', [])]
        result = _single_factor(fact, _endpoints(fact, catalog, facts), refs)
        if not result and fact.get('row_refs') and fact.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT':
            ref = fact['row_refs'][0]
            row = owned[(ref['evidence_id'], ref['row_id'])]
            expected = {_number(row['cells'][m]) for m in ref['metrics']
                        if _number(row['cells'][m]) is not None}
            literals_match = all(f'{m}={row["cells"][m]}' in fact['text'] for m in ref['metrics']
                                 if _number(row['cells'][m]) is None)
            result = {'status': 'MATCH' if expected <= set(_numbers(fact['text'])) and literals_match else 'MISMATCH',
                      'reason': 'source_row_values', 'row_ids': [row['row_id']]}
        if result is None:
            continue
        diagnostics['facts'].append({'fact_id': fact['id'], **result})
        if mode != 'enforce' or assessment['verdict'] != 'SUPPORTED' or result['status'] == 'MATCH':
            continue
        assessment.update(verdict='UNSUPPORTED' if result['status'] == 'MISMATCH' else 'UNCERTAIN',
                          row_binding_status=result['status'], row_binding_reason=result['reason'])
        diagnostics['changed_fact_ids'].append(fact['id'])
    if diagnostics['changed_fact_ids']:
        effective['coverage'] = 'PARTIAL'
        effective['missing_aspects'] = list(effective.get('missing_aspects', [])) + [
            '部分比较尚未核实是否只改变所述模块，不能把组合变化归因于单一模块。']
    return effective, diagnostics
