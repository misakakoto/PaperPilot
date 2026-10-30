"""Model selects source identities; host projects literal values and Decimal math.

Projection is a provenance contract, never an audit verdict. Causality, mechanism
and significance remain separate statements reviewed against original evidence.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re

VERSION = 'table-fact-projection-v9-literal-cell-values'
INSTRUCTION = '''TABLE_FACT overrides the ordinary schema for table measurements.
Do not write experimental values yourself, even when omitting the metric name.
Return {"id":"f4","kind":"TABLE_FACT","operation":"value",
"row_refs":[{"evidence_id":"supplied ID","row_id":"supplied row_id","metrics":["exact numeric column header"]}]}.
No text, numbers, evidence_ids or extra fields in TABLE_FACT. Host writes source
values, units and configuration, then the independent Auditor checks them.
value reads one configuration's cells, including literal non-numeric markers.
The host preserves a marker as source text, never converts it to zero or assigns
its meaning. Interpret it separately using the source legend. Arithmetic and
ranking operations still require numeric measurements in every selected cell.
max/min selects a set
of rows from ONE table with the SAME requested metrics on every row. Each metric
is ranked independently, never combined across columns or units. Multiple metrics
may share one output configuration ONLY when every column has the same unique
winner. Otherwise use separate single-metric facts; never invent an overall winner.
Rankings apply ONLY to the explicitly selected rows.
difference selects two ordered rows from ONE table with one shared metric and
known equal non-module conditions; reports right minus left, never causality.
Mechanism explanations use separate SOURCE_FACTs with method citations.
Derived causal explanations use separate INFERENCEs depending on source facts.
Do not put numbers from a catalogued table in ordinary prose facts.
This restricts NUMERICAL VALUES, not comparison words. A numbers-free INFERENCE
may explicitly identify the best tested configuration or say one configuration
outperforms another, with source citations and SOURCE_FACT premises, subject to
independent audit. Keep numerical values in the separately projected premises.
For a numerical ranking select max/min and EVERY row in the intended
eligible comparison set, not only the presumed winner. Do not replace a requested
ranking with operation=value: a repeated value row does not identify a winner.
For a budget-constrained ranking use operation=filter_max or filter_min, row_refs
for EVERY row in the source table. EVERY row_ref still requires its own nonempty
metrics array of exact requested numeric column headers; the where.metric is
only the filter column and does not replace the output/ranking metrics.
Use the same metrics on all rows. Add one additional field
where={"metric":"exact source column header","operator":"<=","value":"user-requested bound"}.
Operators are <, <=, ==, >=, >. The host filters source values before ranking.
Never drop ineligible source rows yourself. The bound must come from the current question.
The projected ranking already states the winning configuration and the applied
filter. It can be the direct_answer_id; do not add a redundant prose fact repeating
its numbers or numeric threshold. Keep further interpretation numbers-free.
Neither a ranking nor a difference establishes causation or statistical significance.'''


def measurement_headers(row):
    """Resource measurements are outputs; batch/crop/dataset remain conditions."""
    return {k for k in row['cells'] if re.fullmatch(
        r'\s*(?:(?:[GMKT]?FLOPs?|[GMKT]?MACs?|Param(?:eters?)?s?)\s*(?:\([^)]*\))?|'
        r'(?:Latency|Throughput|FPS|Runtime|Memory)\s*(?:\([^)]*\))?)\s*', k, re.I)}


def configuration_headers(rows):
    # Only a column literally identifying every source row is a configuration
    # selector. This does not relax batch/crop/dataset compatibility.
    keys = set.intersection(*(set(r['cells']) for r in rows))
    return {k for k in keys if (
        re.fullmatch(r'methods?|models?|backbones?|encoders?|architectures?|variants?|networks?|方法|模型|骨干', k, re.I)
        and all(r['cells'][k].strip() for r in rows)) or
        all(r.get('row_label') and r['cells'][k] == r['row_label'] for r in rows)}


def expand_table_value_groups(draft, catalog):
    """Normalize initial selections, never prose or exact-ID repair patches.

    A value group is only a source selection convenience. Every child must pass
    the single-configuration contract. All premises go to independent audit;
    no comparison is inferred here. Rejected groups stay intact for quarantine.
    """
    if not isinstance(draft, dict) or not isinstance(draft.get('facts'), list):
        raise ValueError('Expected a fact draft')
    original_ids = [f.get('id') if isinstance(f, dict) else None for f in draft['facts']]
    if any(not isinstance(i, str) for i in original_ids) or len(set(original_ids)) != len(original_ids):
        raise ValueError('Projection requires unique fact IDs')
    output, trace, aliases = deepcopy(draft), [], {}
    occupied = set(original_ids)
    normalized = []
    total = len(original_ids)
    for fact in draft['facts']:
        refs = fact.get('row_refs')
        if not (fact.get('kind') == 'TABLE_FACT' and fact.get('operation') == 'value'
                and isinstance(refs, list) and len(refs) > 1):
            normalized.append(deepcopy(fact))
            continue
        record = {'version': VERSION, 'raw_selection': deepcopy(fact), 'status': 'REJECTED'}
        try:
            if set(fact) != {'id', 'kind', 'operation', 'row_refs'}:
                raise ValueError('Value groups contain selection identities only')
            if total + len(refs) - 1 > 12:
                raise ValueError('Value group expansion exceeds the 12-fact contract')
            children = []
            for i, ref in enumerate(refs):
                child_id = fact['id'] if i == 0 else fact['id'] + '__value_' + str(i)
                if i and child_id in occupied:
                    raise ValueError('Value group child ID collision')
                child = {'id': child_id, 'kind': 'TABLE_FACT', 'operation': 'value',
                         'row_refs': [deepcopy(ref)]}
                project_table_facts({'facts': [child]}, catalog)
                children.append(child)
            identities = [(ref['evidence_id'], ref['row_id']) for ref in refs]
            if len(set(identities)) != len(identities):
                raise ValueError('Repeated table row selection')
            if len({ref['evidence_id'] for ref in refs}) != 1:
                raise ValueError('Value groups require one source table')
        except (ValueError, TypeError) as exc:
            record['error'] = str(exc)
            normalized.append(deepcopy(fact))
        else:
            child_ids = [child['id'] for child in children]
            occupied.update(child_ids)
            total += len(children) - 1
            aliases[fact['id']] = child_ids
            normalized.extend(children)
            record.update(status='EXPANDED', child_ids=child_ids)
        trace.append(record)
    for fact in normalized:
        if (fact.get('kind') == 'INFERENCE' and isinstance(fact.get('depends_on'), list)
                and all(isinstance(item, str) for item in fact['depends_on'])):
            dependencies = []
            for dependency in fact['depends_on']:
                dependencies.extend(aliases.get(dependency, [dependency]))
            fact['depends_on'] = list(dict.fromkeys(dependencies))
    output['facts'] = normalized
    return output, trace


def projection_audit_context(draft, traces):
    """Expose a current host projection's comparison domain, never old claims.

    Only an unchanged projected fact qualifies. The independent auditor still
    reads source cells and checks interpretation, causality and relevance.
    """
    facts = {f['id']: f for f in draft['facts']}
    result = {}
    for trace in traces:
        projected = trace.get('projected_fact', {})
        fid = projected.get('id')
        if (facts.get(fid) != projected or trace.get('version') != VERSION
                or trace.get('raw_selection', {}).get('operation') not in {'max','min','filter_max','filter_min'}
                or not trace.get('rankings')):
            continue
        result[fid] = {'fact_id': fid, 'operation': trace['raw_selection']['operation'],
            'source_refs': deepcopy(trace['raw_selection']['row_refs']),
            'domain': 'explicit_selected_rows_only', 'rankings': deepcopy(trace['rankings']),
            **({'filter': trace['filter'], 'eligible_row_ids': trace['eligible_row_ids']}
               if 'filter' in trace else {}),
            'assurance': 'SOURCE_ARITHMETIC_NOT_SCIENTIFIC_ENTAILMENT'}
    return list(result.values())


def project_table_facts_locally(draft, catalog, *, chinese=False):
    """Keep valid projections when another selection violates its contract.

    Invalid statements remain unchanged for quarantine/revision, never approved.
    Global ID conflicts still fail rather than guessing ownership.
    """
    if not isinstance(draft, dict) or not isinstance(draft.get('facts'), list):
        raise ValueError('Expected a fact draft')
    ids = [f.get('id') if isinstance(f, dict) else None for f in draft['facts']]
    if any(not isinstance(fid, str) for fid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Projection requires unique fact IDs')
    output, trace, errors = deepcopy(draft), [], {}
    facts = []
    for fact in draft['facts']:
        try:
            projected, entries = project_table_facts({'facts': [fact]}, catalog, chinese=chinese)
        except (ValueError, TypeError) as exc:
            errors[fact['id']] = str(exc)
            facts.append(deepcopy(fact))
        else:
            for item in projected['facts']:
                if item['id'] != fact['id'] and item['id'] in ids:
                    raise ValueError('Projection premise ID collision')
            facts.extend(projected['facts'])
            trace.extend(entries)
    if len({f['id'] for f in facts}) != len(facts):
        raise ValueError('Projection premise ID collision')
    _bind_calculation_premises(facts, trace)
    output['facts'] = facts
    return output, trace, errors


def _bind_calculation_premises(facts, trace):
    # The public fact contract permits SOURCE_FACT premises only. A model may
    # depend on a selected host difference; bind that reference to the exact
    # generated literal endpoints, never an arbitrary model inference chain.
    calculations = {t['projected_fact']['id']: t['projected_fact']['depends_on']
                    for t in trace if t['raw_selection']['operation'] == 'difference'}
    for fact in facts:
        if fact.get('kind') != 'INFERENCE' or fact['id'] in calculations:
            continue
        old = fact.get('depends_on', [])
        if not isinstance(old, list):
            continue  # Existing validator reports malformed model dependencies.
        if any(d in calculations for d in old):
            fact['depends_on'] = list(dict.fromkeys(
                endpoint for dep in old for endpoint in calculations.get(dep, [dep])))
            for item in trace:
                if item['projected_fact']['id'] in old:
                    item.setdefault('dependent_bindings', []).append({
                        'fact_id': fact['id'], 'before': old, 'after': list(fact['depends_on'])})
def number(value):
    try:
        result = Decimal(str(value).strip().rstrip('%％'))
    except InvalidOperation as exc:
        raise ValueError('Source cell is not a numeric measurement') from exc
    if not result.is_finite():
        raise ValueError('Source cell is not finite')
    return result


def column_extrema(selected, metrics, operation):
    """Independent source-column queries; a shared winner is not a weighted score."""
    if len(selected) < 2:
        raise ValueError('Ranking requires at least two selected configurations')
    rankings = []
    winner_indices = []
    for metric in metrics:
        extreme = (max if operation == 'max' else min)(number(r['cells'][metric]) for r in selected)
        winners = [i for i, r in enumerate(selected) if number(r['cells'][metric]) == extreme]
        if len(winners) != 1:
            raise ValueError('Tied extrema require separate value facts for every tied row')
        winner_indices.append(winners[0])
        rankings.append({'metric': metric, 'extreme': str(extreme),
                         'winner_row_ids': [selected[winners[0]]['row_id']],
                         'selected_row_ids': [r['row_id'] for r in selected],
                         'source_values': [r['cells'][metric] for r in selected],
                         'unit_basis': 'original column header and literal source cells'})
    if len(set(winner_indices)) != 1:
        raise ValueError('Different column winners require separate single-metric ranking facts')
    return winner_indices[0], rankings


def project_table_facts(draft, catalog, *, chinese=False):
    if not isinstance(draft, dict) or not isinstance(draft.get('facts'), list):
        raise ValueError('Expected a fact draft')
    output, trace = deepcopy(draft), []
    owned = {(r['evidence_id'], r['row_id']): r for r in catalog}
    table_ids = {r['evidence_id'] for r in catalog}
    facts = []
    for fact in output['facts']:
        if fact.get('kind') != 'TABLE_FACT':
            # Broad necessary contract: metric omission cannot evade ownership.
            if set(fact.get('evidence_ids') or []) & table_ids and re.search(
                    r'(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?\s*[%％]?(?![A-Za-z0-9_]|\.\d)', str(fact.get('text', ''))):
                detail = 'Catalogued table numbers require TABLE_FACT projection'
                if fact.get('kind') == 'INFERENCE':
                    detail += (
                        '; retain the requested winner/comparison, dataset and tested scope '
                        'with the existing source premises, but remove repeated measurement literals. '
                        'For a numerical comparison use TABLE_FACT max/min/difference with the '
                        'complete eligible source rows; do not substitute an unranked value row.')
                raise ValueError(detail)
            facts.append(fact)
            continue
        filtered = fact.get('operation') in {'filter_max', 'filter_min'}
        if set(fact) != ({'id', 'kind', 'operation', 'row_refs', 'where'} if filtered else
                         {'id', 'kind', 'operation', 'row_refs'}):
            raise ValueError('TABLE_FACT contains selection identities only, no model prose or values')
        op, refs = fact['operation'], fact['row_refs']
        if op not in {'value', 'difference', 'max', 'min', 'filter_max', 'filter_min'} or not isinstance(refs, list) or not refs:
            raise ValueError('Invalid table operation')
        selected = []
        cell_value_types = []
        for ref_index, ref in enumerate(refs):
            if (not isinstance(ref, dict) or not {'evidence_id', 'row_id'} <= set(ref)
                    or set(ref) - {'evidence_id', 'row_id', 'metrics'}):
                raise ValueError('Expected source-owned row and metric identities')
            row = owned.get((ref['evidence_id'], ref['row_id']))
            if row is None:
                raise ValueError('Unowned source row')
            if 'metrics' not in ref:
                raise ValueError(
                    f'row_refs[{ref_index}].metrics is required for every operation, including '
                    f'filter_max/filter_min. Select requested numeric columns from the owned '
                    f'row headers {list(row["cells"])}; where.metric only filters rows. '
                    'Do not supply experimental values or omit metrics on later rows.')
            metrics = ref['metrics']
            if (not isinstance(metrics, list) or not metrics or
                    len(set(metrics)) != len(metrics) or any(m not in row['cells'] for m in metrics)):
                raise ValueError('Unowned source row or metric')
            for metric in metrics:
                raw = row['cells'][metric]
                try:
                    number(raw)
                except ValueError:
                    if op != 'value' or not isinstance(raw, str) or not raw.strip():
                        raise
                    kind = 'NON_NUMERIC_LITERAL'
                else:
                    kind = 'NUMERIC'
                cell_value_types.append({'row_id': row['row_id'], 'metric': metric,
                                         'kind': kind, 'source_text': raw})
            selected.append(row)
        if len({(r['evidence_id'], r['row_id']) for r in selected}) != len(selected):
            raise ValueError('Repeated table row selection')
        if op == 'value' and len(selected) != 1:
            raise ValueError('One value fact describes one configuration')
        if op != 'value':
            if (len({r['artifact_sha256'] for r in selected}) != 1 or
                    any(ref['metrics'] != refs[0]['metrics'] for ref in refs) or
                    (op == 'difference' and len(refs[0]['metrics']) != 1)):
                raise ValueError('Comparisons require one source table and shared columns; differences require one metric and unit')
            from .numeric_scope import metric_references
            components = set(selected[0]['component_columns'])
            keys = set().union(*(set(r['cells']) for r in selected))
            metric_keys = set(selected[0].get('metric_headers') or [k for k in keys if metric_references(k)])
            metric_keys.update(refs[0]['metrics'])
            metric_keys.update(measurement_headers(selected[0]))
            condition_keys = keys - components - metric_keys - configuration_headers(selected)
            if any(any(r['cells'].get(k) != selected[0]['cells'].get(k) for r in selected)
                   for k in condition_keys):
                raise ValueError('Other source conditions differ or are unknown')
        def configuration_label(row):
            if row.get('row_label'):
                return row['row_label']
            names = configuration_headers([row])
            if names:
                return ', '.join(row['cells'][k] for k in sorted(names))
            modules = row.get('checked_columns')
            if modules is None:
                return '模块条件未知' if chinese else 'module conditions unknown'
            return ', '.join(modules) or ('无勾选模块' if chinese else 'no checked modules')

        filter_trace = {}
        if filtered:
            where = fact['where']
            if (not isinstance(where, dict) or set(where) != {'metric', 'operator', 'value'}
                    or where['operator'] not in {'<', '<=', '==', '>=', '>'}
                    or not isinstance(where['value'], (str, int, float))):
                raise ValueError('Expected one numeric source-column constraint')
            if {(r['evidence_id'], r['row_id']) for r in selected} != {
                    (r['evidence_id'], r['row_id']) for r in catalog
                    if r['evidence_id'] == selected[0]['evidence_id']}:
                raise ValueError('Filtered ranking requires the complete source table')
            bound = number(where['value'])
            def eligible(row):
                if where['metric'] not in row['cells']:
                    raise ValueError('Constraint column not present in source row')
                value = number(row['cells'][where['metric']])
                return {'<': value < bound, '<=': value <= bound, '==': value == bound,
                        '>=': value >= bound, '>': value > bound}[where['operator']]
            chosen = [(r, ref) for r, ref in zip(selected, refs) if eligible(r)]
            if not chosen:
                raise ValueError('No source configuration satisfies the requested constraint')
            selected, refs = map(list, zip(*chosen))
            filter_trace = {'filter': deepcopy(where), 'eligible_row_ids': [r['row_id'] for r in selected]}
        projected_rows, projected_refs = selected, refs
        suffix = ''
        rankings = []
        metric = refs[0]['metrics'][0]
        if op in {'max', 'min', 'filter_max', 'filter_min'}:
            rank_op = op.removeprefix('filter_')
            if filtered and len(selected) == 1:
                winner = 0
                rankings = [{'metric': m, 'extreme': selected[0]['cells'][m],
                    'winner_row_ids': [selected[0]['row_id']], 'selected_row_ids': [selected[0]['row_id']],
                    'source_values': [selected[0]['cells'][m]]} for m in refs[0]['metrics']]
            else:
                winner, rankings = column_extrema(selected, refs[0]['metrics'], rank_op)
            projected_rows, projected_refs = [selected[winner]], [refs[winner]]
            compared = ('、' if chinese else ', ').join(configuration_label(row) for row in selected)
            metric_names = ('、' if chinese else ', ').join(refs[0]['metrics'])
            plural = '分别' if len(rankings) > 1 else ''
            suffix = (f'；在本次所选表内配置（{compared}）中，{metric_names}{plural}' +
                      ('最高' if rank_op == 'max' else '最低') if chinese else
                      f'; {metric_names} each independently {op} among the selected configurations in this table ({compared})')
            if filtered:
                suffix += '；已按当前请求约束筛选，所有符合条件的配置均参与比较' if chinese else '; after the requested constraint; all eligible rows compared'
        def render(row, ref):
            from .numeric_scope import metric_references
            metric_keys = set(row.get('metric_headers') or
                              [k for k in row['cells'] if metric_references(k)])
            metric_keys.update(ref['metrics'])
            metric_keys.update(measurement_headers(row))
            conditions = [f'{k}={v}' for k, v in row['cells'].items()
                          if k not in metric_keys and k not in row['component_columns']]
            config = configuration_label(row)
            return (row['table'] + ' [' + row['paper_sha256'][:12] + ']，' +
                    ('配置：' if chinese else '') + config +
                    ('，' + ', '.join(conditions) if conditions else '') + '：' +
                    '；'.join(m + '=' + row['cells'][m] for m in ref['metrics']))
        if op == 'difference':
            if len(selected) != 2 or any(r.get('checked_columns') is None and not configuration_headers([r]) for r in selected):
                raise ValueError('Difference requires two known configurations')
            # Conditions include all non-component, nonnumeric columns. Numeric
            # conditions not requested as metrics also remain fixed.
            delta = number(selected[1]['cells'][metric]) - number(selected[0]['cells'][metric])
            text = render(selected[0], refs[0]) + '；' + render(selected[1], refs[1])
            text += ('；后者减前者=' if chinese else '; right minus left=') + str(delta)
            text += ' percentage points' if '%' in metric else ' source-column units'
            # Two literal premises are added with stable IDs; ordinary inference
            # guards and the independent audit still decide support.
            premise_ids = []
            for i, (row, ref) in enumerate(zip(selected, refs)):
                fid = fact['id'] + '_endpoint_' + str(i)
                if fid in {f.get('id') for f in output['facts']}:
                    raise ValueError('Projection premise ID collision')
                premise_ids.append(fid)
                facts.append({'id': fid, 'kind': 'SOURCE_FACT', 'text': render(row, ref),
                              'evidence_ids': [row['evidence_id']], 'depends_on': [], 'row_refs': [ref]})
            projected = {'id': fact['id'], 'kind': 'INFERENCE', 'text': text,
                'evidence_ids': [selected[0]['evidence_id']], 'depends_on': premise_ids, 'row_refs': refs}
        else:
            row, ref = projected_rows[0], projected_refs[0]
            text = render(row, ref) + suffix
            literals = [v['metric'] for v in cell_value_types if v['kind'] == 'NON_NUMERIC_LITERAL']
            if literals:
                text += ('；非数值单元格原文：' if chinese else '; non-numeric source literals: ') + ', '.join(literals)
            projected = {'id': fact['id'], 'kind': 'SOURCE_FACT', 'text': text,
                'evidence_ids': [row['evidence_id']], 'depends_on': [], 'row_refs': [ref]}
        facts.append(projected)
        trace.append({'version': VERSION, 'raw_selection': deepcopy(fact), 'projected_fact': deepcopy(projected),
                      'cell_value_types': cell_value_types,
                      **filter_trace,
                      **({'rankings': rankings} if rankings else {})})
    _bind_calculation_premises(facts, trace)
    output['facts'] = facts
    return output, trace
