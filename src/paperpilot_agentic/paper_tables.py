"""Conservative table structure checks and literal, condition-bearing reports."""
from decimal import Decimal, InvalidOperation
import re

from .paper_catalog import identity

DATASETS = ('LEVIR-CD', 'DSIFN-CD', 'WHU-CD', 'WHU', 'S2Looking', 'MSOSCD', 'California', 'PASCAL VOC 2012')


def table_structure(data):
    from .docling_adapter import project_table
    try:
        _, headers, paths, _, _ = project_table(data)
        return {'structure_status': 'GRID_VALIDATED_NOT_FACT_VERIFIED', 'column_headers': headers,
                'header_paths': paths, 'structure_errors': []}
    except (ValueError, KeyError, TypeError) as exc:
        return {'structure_status': 'INVALID_STRUCTURE', 'column_headers': [], 'header_paths': [],
                'structure_errors': [str(exc)]}


def literal_reports(unit):
    """Never split merged numeric cells or infer missing experiment conditions."""
    obj = unit.get('object') or {}
    if obj.get('structure_status') != 'GRID_VALIDATED_NOT_FACT_VERIFIED':
        return []
    cells = obj['cells']
    paths = obj['header_paths']
    reports = []
    for cell in cells:
        if cell['column_header'] or cell['column'] == 0 or cell['column_end'] - cell['column'] != 1:
            continue
        value = cell['text'].strip()
        try:
            if not Decimal(value).is_finite():
                continue
        except InvalidOperation:
            continue
        path = ' / '.join(paths[cell['column']])
        metrics = list(dict.fromkeys(m[0] for m in re.finditer(r'(?<!\w)(?:F1(?:-score)?|mIoU|mIOU|IoU|Precision|Recall|OA)(?!\w)', path, re.I)))
        if len(metrics) != 1:
            continue
        metric = metrics[0]
        labels = [c for c in cells if c['column'] == 0 and c['row'] <= cell['row'] < c['row_end'] and not c['column_header']]
        if len(labels) != 1:
            continue
        method = labels[0]['text'].strip()
        if not method or re.fullmatch(r'[\d.\s]+', method):
            continue
        datasets = [d for d in DATASETS if re.search(re.escape(d) + r'(?![\w-])', path, re.I)]
        # WHU is a prefix of WHU-CD: keep only longest literal identities.
        datasets = [d for d in datasets if not any(d != other and d in other for other in datasets)]
        if not datasets:
            caption = ' '.join(obj.get('caption', []))
            datasets = [d for d in DATASETS if re.search(re.escape(d) + r'(?![\w-])', caption, re.I)]
            datasets = [d for d in datasets if not any(d != other and d in other for other in datasets)]
        dataset = datasets[0] if len(datasets) == 1 else None
        combined = path + ' ' + ' '.join(obj.get('caption', []) + obj.get('footnotes', []))
        splits = list(dict.fromkeys(m[0] for m in re.finditer(r'(?<!\w)(?:Val|Validation|Test|Train)(?!\w)', combined, re.I)))
        split = splits[0] if len(splits) == 1 else None
        measure_unit = '%' if '%' in combined else None
        fields = {'method': method, 'dataset': dataset, 'split': split, 'metric': metric, 'unit': measure_unit}
        # Quotes are original snippets in the unit, not reconstructed normalized prose.
        quotes = {k: min((line for line in unit['text'].splitlines() if v in line), key=len, default='')
                  for k, v in fields.items() if v is not None}
        if any(v not in unit['text'] for v in fields.values() if v is not None):
            continue
        reports.append({'id': identity([unit['id'], cell['row'], cell['column'], value]), **fields,
            'value': value, 'source_cell': {'row': cell['row'], 'column': cell['column']},
            'header_path': paths[cell['column']],
            'method_cell': {'row': labels[0]['row'], 'column': labels[0]['column']},
            'context_quotes': quotes, 'pretraining': None, 'input_size': None, 'training_protocol': None})
    return reports


def validate_grid(obj):
    if obj.get('row_group') is not None:
        header_rows = {r for c in obj['cells'] if c.get('column_header')
                       for r in range(c['row'], c.get('row_end', c['row']+1))}
        retained = sorted(header_rows | set(obj['row_group']))
        remap = {r: i for i, r in enumerate(retained)}
    else:
        remap = None
    data = {'num_rows': obj.get('num_rows'), 'num_cols': obj.get('num_cols'), 'table_cells': [
        {'start_row_offset_idx': c['row'], 'end_row_offset_idx': c.get('row_end', c['row']+1),
         'start_col_offset_idx': c['column'], 'end_col_offset_idx': c.get('column_end', c['column']+1),
         'text': c['text'], 'column_header': c.get('column_header', False),
         'row_header': c.get('row_header', False), 'bbox': c.get('bbox')} for c in obj.get('cells', [])]}
    if remap is not None:
        for cell in data['table_cells']:
            if (type(obj.get('num_rows')) is not int or cell['start_row_offset_idx'] < 0
                    or cell['end_row_offset_idx'] > obj['num_rows']):
                raise ValueError('Table row group escaped original dimensions')
            rows = [r for r in remap if cell['start_row_offset_idx'] <= r < cell['end_row_offset_idx']]
            if not rows:
                raise ValueError('Cell escaped declared row group')
            cell['start_row_offset_idx'], cell['end_row_offset_idx'] = remap[min(rows)], remap[max(rows)]+1
        data['num_rows'] = len(remap)
    status = table_structure(data)
    if status['structure_errors']:
        raise ValueError('Reported numbers require a complete unambiguous table grid: ' + str(status['structure_errors']))
