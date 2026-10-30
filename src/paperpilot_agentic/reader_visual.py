"""Explicit, source-scoped bridge to the existing persistent layout tool.

Targets must already be selected/authorized. This is not an automatic router.
"""
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

from .models import Budget, StepStatus
from .llm_runtime import LLMBudgetExhausted
from .numeric_scope import metric_references


def table_calculations(rows):
    """Exact column extrema over the transcribed rows, not cross-table rankings."""
    columns = {}
    for index, row in enumerate(rows):
        for header, raw in (row.get('cells') or {}).items():
            if not metric_references(str(header)):
                continue
            text = str(raw).strip()
            if not re.fullmatch(r'\d+(?:\.\d+)?\s*%?', text):
                continue
            columns.setdefault(header, []).append((index, Decimal(text.rstrip('%').strip())))
    result = []
    for header, cells in columns.items():
        # Missing values cannot establish the maximum of the complete table.
        if len(cells) != len(rows):
            continue
        low, high = min(v for _, v in cells), max(v for _, v in cells)
        result.append({'metric_header': header, 'min': str(low), 'max': str(high),
                       'max_row_indices': [i for i, v in cells if v == high],
                       'min_row_indices': [i for i, v in cells if v == low],
                       'max_minus_min': str(high - low),
                       'difference_unit': 'percentage_points' if '%' in header else 'source_column_units'})
    return result


def named_table_content(item, label):
    """Lossless named cells for the reader; do not interpret rows or choose a winner."""
    meta = (item.get('provenance') or {}).get('vision_metadata') or {}
    rows = meta.get('audited_table_rows')
    if not rows:
        return item['content']
    table = {'column_headers': meta.get('column_headers') or [], 'rows': rows,
         'computed_column_summaries': table_calculations(rows),
         'nearby_pdf_caption_text': (item.get('provenance') or {}).get('nearby_caption_text')}
    if meta.get('literal_source_annotations'):
        table['literal_source_annotations'] = meta['literal_source_annotations']
        table['nearby_pdf_caption_status'] = (
            'Unreviewed PDF text extraction; may contain glyph errors. '
            'The separately identified literal annotations record the reviewed crop transcription; '
            'answer claims still require independent audit.')
    return f'{label}\nNamed-cell transcription (each row is independent):\n' + json.dumps(table, ensure_ascii=False)


def deterministic_table_difference(question, evidence):
    """Answer a narrowly specified baseline-vs-full-row subtraction exactly."""
    if not (re.search(r'完整组合|full\s+combination', question, re.I) and
            re.search(r'基线|baseline', question, re.I) and
            re.search(r'多少.*百分点|高多少|difference', question, re.I)):
        return None
    metrics = {item['metric'] for item in metric_references(question)}
    if len(metrics) != 1:
        return None
    requested_metric = next(iter(metrics))
    from .numeric_scope import _metric_name, _structured_tables
    matches = []
    for row in evidence:
        for table in _structured_tables(row['content']):
            values = []
            for index, table_row in enumerate(table['rows']):
                checked = list(table_row.get('checked_columns') or [])
                for header, raw in (table_row.get('cells') or {}).items():
                    if _metric_name(str(header)) != requested_metric:
                        continue
                    text = str(raw).strip().rstrip('%').strip()
                    if re.fullmatch(r'\d+(?:\.\d+)?', text):
                        values.append((index, checked, Decimal(text), str(header)))
            if not values:
                continue
            baseline = [item for item in values if not item[1]]
            largest = max(len(item[1]) for item in values)
            full = [item for item in values if len(item[1]) == largest]
            if len(baseline) == 1 and len(full) == 1 and largest > 0:
                matches.append((row['evidence_id'], baseline[0], full[0]))
    if len(matches) != 1:
        return None
    evidence_id, baseline, full = matches[0]
    difference = full[2] - baseline[2]
    value = str(difference)
    if re.search(r'[\u4e00-\u9fff]', question):
        answer = (f'完整组合的 {requested_metric} 为 {full[2]}，基线为 {baseline[2]}，'
                  f'因此高 {value} 个百分点（{full[2]} − {baseline[2]} = {value}）。')
    else:
        answer = (f'The full combination is {full[2]} versus the {baseline[2]} baseline, '
                  f'a difference of {value} percentage points ({full[2]} − {baseline[2]} = {value}).')
    return {'answer': f'{answer} [{evidence_id}]', 'difference': value,
            'baseline': str(baseline[2]), 'full': str(full[2]),
            'metric': requested_metric, 'evidence_id': evidence_id}


def deterministic_ablation_summary(question, evidence):
    """Summarize two or more verified named-cell ablation tables exactly.

    The model has already transcribed and geometrically validated each table.
    This layer performs only row/column selection and a conservative comparison
    boundary; it does not infer paper-wide superiority.
    """
    if not (re.search(r'消融|ablation', question, re.I) and
            re.search(r'最佳|best', question, re.I) and
            re.search(r'更优|优于|superior|better', question, re.I)):
        return None
    from .numeric_scope import _structured_tables, dataset_matches
    items = []
    for evidence_row in evidence:
        label_match = re.search(r'(?im)^Table\s+([IVXLCDM]+|\d+)\s*$', evidence_row['content'])
        if not label_match:
            continue
        label = f'Table {label_match.group(1)}'
        owner_match = re.search(
            rf'(?P<owner>[A-Za-z][A-Za-z0-9_-]*(?:Mamba|Net|ViT))\s+{re.escape(label)}',
            question, re.I)
        owner = owner_match.group('owner') if owner_match else label
        for table in _structured_tables(evidence_row['content']):
            rows = table.get('rows') or []
            summaries = table.get('computed_column_summaries') or table_calculations(rows)
            if not rows or not summaries:
                continue
            max_rows = [index for summary in summaries for index in summary.get('max_row_indices', [])]
            if not max_rows:
                continue
            best_index = max(set(max_rows), key=lambda index: (max_rows.count(index), -index))
            if best_index >= len(rows):
                continue
            best_row = rows[best_index]
            settings = list(dict.fromkeys(str(column) for row in rows
                                          for column in (row.get('checked_columns') or [])))
            offsets = []
            for row in rows:
                raw = str((row.get('cells') or {}).get('offset') or row.get('row_label') or '').strip()
                if raw and raw.casefold() not in {'w/o', 'without', 'none', '0', '0px'} and raw not in offsets:
                    offsets.append(raw)
            metrics = [(str(summary['metric_header']), str(summary['max'])) for summary in summaries
                       if best_index in summary.get('max_row_indices', [])]
            caption = table.get('nearby_pdf_caption_text') or ''
            if isinstance(caption, dict):
                caption = caption.get('text') or ''
            datasets = list(dict.fromkeys(item['dataset'] for item in dataset_matches(str(caption))))
            items.append({'owner': owner, 'label': label, 'evidence_id': evidence_row['evidence_id'],
                          'settings': settings, 'offsets': offsets, 'metrics': metrics,
                          'datasets': datasets,
                          'best_checked': list(best_row.get('checked_columns') or []),
                          'best_offset': str((best_row.get('cells') or {}).get('offset') or '').strip()})
    if len(items) < 2:
        return None
    chinese = bool(re.search(r'[\u4e00-\u9fff]', question))
    lines = []
    for item in items:
        dataset = f"（{', '.join(item['datasets'])}）" if item['datasets'] else ''
        settings = '、'.join(item['settings']) if item['settings'] else '表内配置项'
        if item['offsets']:
            settings += '；非零偏移为 ' + '、'.join(item['offsets'])
        best = '+'.join(item['best_checked']) if item['best_checked'] else '基线配置'
        if item['best_offset'] and item['best_offset'].casefold() not in {'w/o', 'without', 'none'}:
            best += f"，offset={item['best_offset']}"
        metrics = '、'.join(f'{name}={value}' for name, value in item['metrics'])
        if chinese:
            lines.append(f"{item['owner']} {item['label']}{dataset}：消融设置包括{settings}；"
                         f"最佳配置为{best}，{metrics}。 [{item['evidence_id']}]")
        else:
            lines.append(f"{item['owner']} {item['label']}{dataset}: settings include {settings}; "
                         f"the best row is {best}, with {metrics}. [{item['evidence_id']}]")
    datasets = {dataset.casefold() for item in items for dataset in item['datasets']}
    metric_sets = {tuple(name.casefold() for name, _ in item['metrics']) for item in items}
    citations = ' '.join(f"[{item['evidence_id']}]" for item in items)
    if chinese:
        boundary = ('这些表使用不同的数据集或指标体系，只能说明各自组件在各自实验设置内的贡献，'
                    '不能据此证明某篇论文的方法整体更优。')
    else:
        boundary = ('The tables use different datasets or metric sets. They show component effects only '
                    'within each experiment and cannot prove that either paper is overall superior.')
    # Even if labels happen to match, an ablation-only comparison never proves
    # paper-wide superiority. Keep the observed difference for diagnostics.
    return {'answer': '\n\n'.join(lines + [f'{boundary} {citations}']),
            'tables': items, 'different_datasets': len(datasets) > 1,
            'different_metric_sets': len(metric_sets) > 1,
            'evidence_ids': [item['evidence_id'] for item in items]}


def validate_visual_targets(targets, scope):
    if not isinstance(targets, list) or not 1 <= len(targets) <= 4:
        raise ValueError('Expected one to four explicit visual targets')
    seen, result = set(), []
    for target in targets:
        if not isinstance(target, dict) or set(target) != {'paper_sha256', 'label'}:
            raise ValueError('Visual targets accept only paper_sha256 and label, never client paths')
        sha, label = target['paper_sha256'], target['label']
        if not isinstance(sha, str) or sha not in scope:
            raise ValueError('Visual target outside current paper scope')
        if not isinstance(label, str) or not re.fullmatch(r'Table (?:[IVXLCDM]+|[0-9]+)', label):
            raise ValueError('Use an explicit Table label')
        if (sha, label) in seen:
            raise ValueError('Duplicate visual target')
        seen.add((sha, label))
        if sum(s == sha for s, _ in seen) > 2:
            raise ValueError('At most two target tables per paper')
        result.append((sha, label))
    return result


def collect_visual_evidence(runtime, run_id, question, targets, scope, tool):
    selected = validate_visual_targets(targets, scope)
    resolved = []
    for sha, label in selected:
        paths = runtime._resolve_pdf_paths({sha})
        if len(paths) != 1:
            raise ValueError('Target must resolve to exactly one indexed PDF')
        path = Path(paths[0]).resolve()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != sha:
            raise ValueError('Target PDF no longer matches indexed source hash')
        resolved.append((sha, label, str(path)))
    result, skipped = [], []
    for sha, label, path in resolved:
        run = runtime.db.get_run(run_id)
        if run['status'] == 'CANCELLED':
            raise InterruptedError('Run cancelled')
        budget = Budget.from_dict(run['budget'])
        if run['used_steps'] >= budget.max_steps:
            raise LLMBudgetExhausted('Visual step budget exhausted')
        if (not budget.unlimited_llm_usage and budget.max_tokens - run['used_tokens'] <=
                budget.synthesis_reserve_tokens + budget.repair_reserve_tokens):
            raise LLMBudgetExhausted('No unreserved token budget for visual extraction')
        step = runtime.db.create_step(run_id, 'reader.visual', {'paper_sha256': sha, 'label': label})
        try:
            outcome = tool('paper.inspect_layout', lambda: runtime.tools.invoke(
                run_id=run_id, step_id=step, tool_name='paper.inspect_layout',
                # A single crop cannot answer the whole multi-paper question.
                # Keep this subtask local; Writer/Auditor retain the original question.
                arguments={'pdf_path': path, 'object_type': 'table',
                           'query': f'{label}. Transcribe the complete table with exact headers, '
                                    'all rows, indicator groups, offsets, and numeric values. '
                                    'Do not compare papers or interpret model superiority.',
                           'top_k': 1}))
            if outcome.result is None or not outcome.result.ok:
                raise RuntimeError('Visual tool did not complete successfully')
            if not outcome.result.evidence:
                skipped.append({'paper_sha256': sha, 'label': label, 'reason': 'no_matching_layout_evidence'})
            for index, item in enumerate(outcome.result.evidence):
                if not str(item.get('source_uri', '')).startswith(f'paper://{sha}/'):
                    raise ValueError('Visual tool returned another paper')
                provenance = dict(item.get('provenance') or {})
                if not provenance.get('vision_verified'):
                    skipped.append({'paper_sha256': sha, 'label': label, 'reason': 'unverified_visual_extraction'})
                    continue
                binding = provenance.get('layout_target_binding') or {}
                object_id = provenance.get('layout_object_id')
                if (binding.get('label') != ['table', label.split()[1]] or
                        binding.get('document_sha256') != sha or
                        binding.get('object_id') != object_id or not object_id or
                        binding.get('basis') != 'own_caption_or_heading' or
                        item.get('source_uri') != f'paper://{sha}/layout/{object_id}'):
                    skipped.append({'paper_sha256': sha, 'label': label,
                                    'reason': 'unbound_visual_target'})
                    continue
                result.append({**item, 'evidence_id': f'{outcome.tool_call_id}_{index}',
                    'content': ('[AI/human reviewed literal source asset; answer still requires audit]\n'
                                if provenance.get('reviewed_source') else
                                '[Vision-model transcription; requires source audit, not original PDF text]\n')
                    + named_table_content(item, label)})
            runtime.db.update_step(step, StepStatus.COMPLETED, output={'tool_call_id': outcome.tool_call_id})
        except Exception as exc:
            runtime.db.update_step(step, StepStatus.FAILED, error=type(exc).__name__)
            skipped.append({'paper_sha256': sha, 'label': label, 'reason': type(exc).__name__})
            runtime.db.update_run_metadata(run_id, {'reader_visual_skipped': skipped})
            raise
    runtime.db.update_run_metadata(run_id, {'reader_visual_skipped': skipped,
                                           'reader_visual_evidence_count': len(result)})
    return result
