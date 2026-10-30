"""Conservative provenance checks, not a semantic fact judge.

Only explicit named-dataset percentage assertions are parsed. Table captions
and local sentences supply scope anchors; no dataset/model names or benchmark
values are hard-coded. A second, deliberately necessary-condition layer checks
explicit table, metric, method and causal markers against the cited source
group. Ambiguous scope never establishes numerical support. None of these
checks proves full entailment or semantic truth.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
import re

VERSION = 'dataset-percentage-scope-claim-bindings-v11-outcome-causal-scope'
NAME = r'[A-Za-z][A-Za-z0-9_-]{1,63}'
BARE_NAME = r'(?=[A-Za-z0-9_-]*[A-Z0-9_-]\b)[A-Za-z][A-Za-z0-9_-]{1,63}'
NUMBER = re.compile(r'(?<![A-Za-z0-9_.+-])(?P<value>\d+(?:\.\d+)?)\s*(?:[%％]|percent\b|per\s+cent\b)', re.I)
DATASET = re.compile(
    rf'\bon\s+(?:the\s+)?(?P<en>{NAME})\s+(?:(?:(?:t\s*est|train(?:ing)?|validation)\s+)?'
    r'(?:data\s*set|set|benchmark)|(?:test|training|validation)(?=\s*[,;:.]|$))\b|'
    rf'在\s*(?P<zh>{NAME})\s*(?:数据集|测试集|验证集|训练集)?\s*上', re.I)
NAMED_DATASET_ZH = re.compile(rf'(?P<name>{NAME})\s*(?:数据集|测试集|验证集|训练集)')
BARE_DATASET = re.compile(rf'\bon\s+(?:the\s+)?(?P<en>{BARE_NAME})\b')
MULTI_DATASET = re.compile(
    rf'\bon\s+(?:the\s+)?(?P<names>{NAME}(?:\s*(?:,|and|&)\s*{NAME})+)\s+'
    r'(?:data\s*sets?|sets?|benchmarks?)\b')
HEADING = re.compile(r'^\s*TABLE\s+([IVXLCDM]+|\d+)\b', re.M)
BOUNDARY = re.compile(r'\n\s*\n|(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])')
TABLE_REFERENCE = re.compile(
    r'(?<![A-Za-z0-9])(?:table|表)\s*(?P<table>[IVXLCDM]+|\d+)(?![A-Za-z0-9])', re.I)
METRIC_REFERENCE = re.compile(
    r'(?ix)(?<![A-Za-z0-9])(?:'
    r'f\s*1\s*(?:overall|loc|dmg)|f\s*overall\s*1|f\s*overall|'
    r'overall\s+f\s*1|'
    r'overall\s+accuracy|m\s*iou|miou|accuracy|iou|oa|kc|sek|f1|'
    r'总体准确率|准确率|平均交并比|交并比'
    r')(?![A-Za-z0-9])')
MODEL_REFERENCE = re.compile(
    r'(?<![A-Za-z0-9])(?P<name>[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*)(?![A-Za-z0-9])')

# These are ordinary prose words, not a named method. They prevent the
# necessary-condition check from treating ``Method`` or a table caption word
# as an invented model name. The list is intentionally conservative: unknown
# CamelCase names remain visible to the check rather than being silently
# accepted.
_ENTITY_STOPWORDS = {
    'a', 'an', 'and', 'at', 'baseline', 'by', 'change', 'comparison', 'data',
    'dataset', 'datasets', 'different', 'detection', 'details', 'dl', 'for',
    'from', 'in', 'implementation', 'is', 'model', 'method', 'module', 'new',
    'of', 'on', 'or', 'our', 'ours', 'paper', 'performances', 'performance',
    'proposed', 'remote', 'results', 'result', 'sensing', 'set', 'space',
    'state', 'study', 'table', 'test', 'testing', 'the', 'these', 'this',
    'those', 'to', 'training', 'using', 'validation', 'where', 'with',
    'without', 'work', 'figure', 'fig', 'ieee', 'chapter', 'section', 'task',
    'tasks', 'network', 'networks', 'loss', 'mechanism', 'mechanisms',
    'accuracy', 'overall', 'f1', 'iou', 'miou', 'oa', 'kc', 'sek',
}
_TASK_OR_ROLE_NAMES = {'bcd', 'scd', 'bda', 'cnn', 'rnn', 'ssm', 'dl'}
_WEAK_METRICS = {'accuracy', 'overall_accuracy'}


def _normalise_token(value):
    return ''.join(ch.casefold() for ch in str(value) if ch.isalnum())


def _roman_to_int(value):
    text = str(value).upper()
    if text.isdigit():
        return int(text)
    if not text or any(ch not in 'IVXLCDM' for ch in text):
        return None
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    previous = 0
    for char in reversed(text):
        current = values[char]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total


def table_references(text):
    """Return normalized explicit table references (Roman and Arabic forms)."""
    result = []
    for match in TABLE_REFERENCE.finditer(text):
        number = _roman_to_int(match.group('table'))
        if number is not None:
            result.append({'table': number, 'text': match.group(0),
                           'start': match.start(), 'end': match.end()})
    return result


def _metric_name(value):
    normalized = _normalise_token(value)
    if normalized in {'f1overall', 'foverall1', 'foverall', 'overallf1'}:
        return 'f1overall'
    if normalized in {'f1loc', 'floc1', 'floc'}:
        return 'f1loc'
    if normalized in {'f1dmg', 'fdmg1', 'fdmg'}:
        return 'f1dmg'
    if normalized in {'f1'}:
        return 'f1'
    if normalized in {'miou'}:
        return 'miou'
    if normalized in {'iou'}:
        return 'iou'
    if normalized in {'oa'}:
        return 'oa'
    if normalized in {'kc'}:
        return 'kc'
    if normalized in {'sek'}:
        return 'sek'
    if normalized in {'overallaccuracy'}:
        return 'overall_accuracy'
    if normalized in {'accuracy', '准确率'}:
        return 'accuracy'
    if normalized in {'总体准确率'}:
        return 'overall_accuracy'
    if normalized in {'平均交并比'}:
        return 'miou'
    if normalized in {'交并比'}:
        return 'iou'
    return None


def metric_references(text):
    """Return canonical metric labels explicitly mentioned in ``text``."""
    result = []
    occupied = []
    for match in METRIC_REFERENCE.finditer(text):
        name = _metric_name(match.group(0))
        if name is None or any(match.start() < end and match.end() > start
                               for start, end in occupied):
            continue
        occupied.append((match.start(), match.end()))
        result.append({'metric': name, 'text': match.group(0),
                       'start': match.start(), 'end': match.end(),
                       'strength': 'weak' if name in _WEAK_METRICS else 'strong'})
    return result


def _method_references(text):
    """Extract likely named methods/modules without treating prose as a model."""
    datasets = {_normalise_token(item['dataset']) for item in dataset_matches(text)}
    metric_spans = metric_references(text)
    table_spans = table_references(text)
    result = []
    for match in MODEL_REFERENCE.finditer(text):
        token = match.group('name')
        key = _normalise_token(token)
        # Mathematical subscripts and PDF line-wrapped feature symbols are not
        # model identifiers. Do not normalize them into invented method names.
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9]*_[a-z](?:_?[a-z0-9])*', token):
            continue
        if re.fullmatch(r'[A-Z]{1,2}\d*', token):
            continue
        if not key or key in _ENTITY_STOPWORDS or key in datasets:
            continue
        if any(match.start() < item['end'] and match.end() > item['start']
               for item in table_spans):
            continue
        if key in _TASK_OR_ROLE_NAMES:
            continue
        has_inner_capital = any(char.isupper() for char in token[1:])
        has_lower = any(char.islower() for char in token)
        suffix_name = key.endswith(('mamba', 'net', 'former', 'transformer'))
        context = text[max(0, match.start() - 30):match.end() + 30].casefold()
        acronym = token.isupper() and len(token) >= 3 and bool(re.search(
            r'module|loss|mechanism|model|network|method|baseline|模块|损失|机制|模型|方法', context))
        if not ((has_inner_capital and has_lower) or suffix_name or acronym):
            continue
        if any(match.start() < item['end'] and match.end() > item['start']
               for item in metric_spans):
            continue
        result.append({'method': token, 'key': key, 'start': match.start(), 'end': match.end()})
    unique = {(item['key'], item['start']): item for item in result}
    return list(unique.values())


_CAUSAL_FAMILIES = {
    'attribution': re.compile(
        r'(?i)\b(?:attribut\w*|because\s+of|due\s+to)\b|归因|由于'),
    'benefit': re.compile(
        r'(?i)\b(?:benefit\w*\s+from|stem\w*\s+from|result\w*\s+from)\b|得益于|源于'),
    'cause': re.compile(
        r'(?i)\b(?:caus\w*|lead\w*\s+to|result\w*\s+in)\b|导致|促使'),
    'contribution': re.compile(
        r'(?i)\b(?:contribut\w*|effectiv\w*|demonstrat\w*|improv\w*|enhanc\w*)\b|贡献|提升|改善|增强'),
}


def _causal_families(text):
    return {name for name, pattern in _CAUSAL_FAMILIES.items() if pattern.search(text)}


def _empirical_outcome_claim(text):
    """Scope this lexical veto to outcome attribution, not all causal grammar.

    A problem statement translated from an English method paragraph need not
    repeat its causal cue family. Its actor/operation/purpose entailment remains
    the independent Auditor's job. Numbers and performance assertions retain
    the existing conservative outcome binding and table-condition guards.
    """
    return bool(NUMBER.search(text) or re.search(
        r'性能|准确率|精度|鲁棒|显著|得分|\b(?:performance|accuracy|robustness|'
        r'precision|recall|score|significan\w*)\b', text, re.I))


def _source_method_keys(text):
    return {_normalise_token(item['method']) for item in _method_references(text)}


def _source_metric_keys(text):
    return {item['metric'] for item in metric_references(text)}


def _metric_compatible(claim, source_metrics):
    if claim in source_metrics:
        return True
    # ``accuracy`` and ``overall accuracy`` are commonly shortened in a
    # generated answer. Treat the pair as an alias only for this weak metric;
    # do not merge F1 variants or other explicitly named measures.
    return claim in _WEAK_METRICS and bool(
        source_metrics.intersection(_WEAK_METRICS))


VALUE_LITERAL = re.compile(r'(?<![A-Za-z0-9_.])(?P<value>\d+(?:\.\d+)?)(?:\s*[%％])?')


def _metric_value_pairs(text):
    """Pair an explicit metric with a directly stated numeric literal.

    Do not use a broad proximity window: in a sentence listing several
    metrics, the last percentage must not be attached to every preceding label.
    """
    pairs = []
    for metric in metric_references(text):
        tail = text[metric['end']:min(len(text), metric['end'] + 70)]
        match = re.match(
            r'''\s*(?:[（(]\s*[%％]\s*[）)]\s*)?["')\]}]*
                (?:\s*(?:=|:|is\s+|of\s+|at\s+|score\s+of\s+|value\s+of\s+|reaches?\s+|为\s*|是\s*|达到\s*|得分为\s*)\s*|\s+)
                ["']?(?P<value>\d+(?:\.\d+)?)(?![\d.])''',
            tail, re.I | re.X)
        if match:
            pairs.append({'metric': metric['metric'], 'value': match.group('value'),
                          'metric_start': metric['start'],
                          'value_start': metric['end'] + match.start('value'),
                          'distance': metric['end'] + match.end() - metric['end']})
    return pairs


def _value_present(text, value):
    target = Decimal(str(value))
    return any(Decimal(number.group('value')) == target
               for number in VALUE_LITERAL.finditer(text))


def _cell_value_present(value, target):
    return _value_present(str(value), target)


def _structured_tables(text):
    """Decode lossless named-cell transcriptions when a visual bridge supplied one."""
    marker = 'Named-cell transcription'
    decoder = json.JSONDecoder()
    result = []
    offset = 0
    while True:
        marker_start = text.find(marker, offset)
        if marker_start < 0:
            break
        start = text.find('{', marker_start + len(marker))
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, TypeError):
            offset = start + 1
            continue
        if isinstance(value, dict) and isinstance(value.get('rows'), list):
            headers = value.get('column_headers') or []
            if isinstance(headers, list) and all(isinstance(header, str) for header in headers):
                preceding = text[:marker_start]
                table_matches = table_references(preceding)
                table = table_matches[-1]['table'] if table_matches else None
                result.append({'table': table, 'headers': headers, 'rows': value['rows'],
                               'computed_column_summaries': value.get('computed_column_summaries') or [],
                               'nearby_pdf_caption_text': value.get('nearby_pdf_caption_text'),
                               'start': marker_start, 'end': marker_start + end})
        offset = start + end
    return result


def _table_metric_cells(table):
    cells = {}
    for header in table['headers']:
        metric_items = metric_references(str(header))
        if not metric_items:
            continue
        metric = metric_items[0]['metric']
        values = []
        for row in table['rows']:
            if not isinstance(row, dict) or not isinstance(row.get('cells'), dict):
                continue
            for key, value in row['cells'].items():
                if _metric_name(str(key)) == metric:
                    values.append({'row_label': row.get('row_label'), 'value': value,
                                   'header': str(key)})
        if values:
            cells.setdefault(metric, []).extend(values)
    return cells


def _table_cell_checks(fact, rows, metric_values):
    """Check metric=value pairs against structured visual table columns.

    The check is deliberately inactive for plain prose/table text: without a
    verified named-cell transcription there is no safe row/column interpretation.
    """
    structures = []
    for row in rows:
        for table in _structured_tables(row['content']):
            datasets = {_key(item['dataset']) for item in dataset_matches(row['content'])}
            structures.append({**table, 'evidence_id': row['evidence_id'],
                               'metric_cells': _table_metric_cells(table),
                               'scope_datasets': sorted(datasets)})
    if not structures or not metric_values:
        return []
    requested_tables = {item['table'] for item in table_references(fact['text'])}
    if requested_tables:
        selected = [item for item in structures if item['table'] in requested_tables]
        if not selected:
            selected = structures
    else:
        selected = structures
    checks = []
    for pair in metric_values:
        cells = [cell for table in selected for cell in table['metric_cells'].get(pair['metric'], [])]
        if not cells:
            other_cells = [cell for table in selected for values in table['metric_cells'].values()
                           for cell in values]
            status = 'MISMATCH' if any(_cell_value_present(cell['value'], pair['value'])
                                       for cell in other_cells) else 'UNRESOLVED'
        elif any(_cell_value_present(cell['value'], pair['value']) for cell in cells):
            status = 'MATCH'
        else:
            other_cells = [cell for table in selected
                           for values in table['metric_cells'].values()
                           for cell in values if cell not in cells]
            status = ('MISMATCH' if any(_cell_value_present(cell['value'], pair['value'])
                                        for cell in other_cells) else 'UNRESOLVED')
        checks.append(_binding_check('table_cell', f"{pair['metric']}={pair['value']}", status,
                                     blocking=True, metric=pair['metric'], value=pair['value'],
                                     table_numbers=sorted({item['table'] for item in selected
                                                           if item['table'] is not None}),
                                     scope_datasets=sorted({dataset for item in selected
                                                           for dataset in item['scope_datasets']})))
    return checks


ROW_CUE = re.compile(
    r'(?i)baseline|基线|单独(?:加入|引入|启用)|同时(?:加入|引入|启用)|'
    r'加入|引入|启用|without|w/o|共同|同时|组合|combination|both')


def _row_component_names(text, start, end, *, cut_at_forward_punctuation=True, known_components=()):
    if cut_at_forward_punctuation:
        punctuation = [position for position in
                       (text.find(mark, start, end) for mark in ',，;；。') if position >= 0]
        if punctuation:
            end = min(end, min(punctuation))
    metric_spans = metric_references(text)
    names = set()
    for match in MODEL_REFERENCE.finditer(text, start, end):
        if match.end() < len(text) and re.match(r'[A-Za-z0-9_-]', text[match.end()]):
            continue
        token = match.group('name')
        key = _normalise_token(token)
        prefix = text[max(0, match.start() - 24):match.start()]
        if re.search(r'(?:无|不|未)(?:使用|加入|引入|启用)?\s*$|'
                     r'(?:without|w/o|do not (?:use|enable))\s*$', prefix, re.I):
            continue
        if (key in _ENTITY_STOPWORDS or key in _TASK_OR_ROLE_NAMES or
                _roman_to_int(token) is not None or
                any(match.start() < item['end'] and match.end() > item['start']
                    for item in metric_spans)):
            continue
        if key in known_components or (token.isupper() and len(token) >= 2):
            names.add(key)
        elif any(char.isupper() for char in token[1:]) and any(char.islower() for char in token):
            names.add(key)
    return names


def _row_references(text, known_components=()):
    """Find explicit ablation-row cues and their named component combination."""
    result = []
    cues = list(ROW_CUE.finditer(text))
    has_specific = any(cue.group(0).casefold() not in {'组合', 'combination'} for cue in cues)
    for cue in cues:
        cue_name = cue.group(0).casefold()
        if (any(mark in cue_name for mark in ('加入', '引入', '启用')) and
                re.search(r'(?:不|未|没有)\s*$', text[max(0, cue.start()-4):cue.start()])):
            continue
        if cue_name in {'组合', 'combination'} and has_specific:
            continue
        # A prose mention such as ``row labeled 'w/o'`` describes a source
        # label, not the configuration asserted by the sentence.
        if cue_name == 'w/o' and re.search(
                r"row\s+labeled?\s*['\"]?\s*$", text[max(0, cue.start() - 28):cue.start()], re.I):
            continue
        if cue_name == 'w/o' and re.search(r'[\w /()%.-]+=\s*$', text[max(0, cue.start()-60):cue.start()]):
            # A literal condition value such as offset=w/o is not a claim
            # that every component is absent. Owned row refs check the actual
            # configuration independently; prose baseline cues still apply.
            continue
        if cue_name in {'baseline', '基线', 'without', 'w/o'}:
            start, end = cue.start(), min(len(text), cue.end() + 100)
            names = set()
        elif any(mark in cue_name for mark in ('加入', '引入', '启用')):
            start, end = cue.end(), min(len(text), cue.end() + 42)
            names = _row_component_names(text, start, end, known_components=known_components)
            if cue_name in {'加入', '引入', '启用'}:
                clause_start = max([text.rfind(mark, max(0, cue.start() - 45), cue.start())
                                    for mark in ',，;；。'] + [-1]) + 1
                names |= _row_component_names(text, clause_start, cue.start(), known_components=known_components)
        elif cue_name in {'组合', 'combination'}:
            boundary = max([text.rfind(mark, max(0, cue.start() - 55), cue.start())
                            for mark in ',，;；。'] + [-1])
            start, end = max(boundary + 1, cue.start() - 45), cue.start()
            names = _row_component_names(text, start, end, known_components=known_components)
        else:
            start, end = max(0, cue.start() - 45), min(len(text), cue.end() + 30)
            names = _row_component_names(text, start, end, cut_at_forward_punctuation=False,
                                         known_components=known_components)
        if not names and cue_name not in {'baseline', '基线', 'without', 'w/o'}:
            continue
        before_cue = text[max(0, cue.start() - 30):cue.start()].casefold()
        negative = (cue_name in {'baseline', '基线', 'without', 'w/o'} or
                    bool(re.search(r'(?:无|without|w/o)(?:\s+[a-z0-9]+){0,2}\s*$', before_cue)))
        combo = frozenset() if negative else frozenset(names)
        result.append({'combo': combo, 'cue': cue.group(0), 'start': cue.start(),
                       'end': cue.end(), 'window_start': start, 'window_end': end})
    unique = {(item['combo'], item['start']): item for item in result}
    return list(unique.values())


def _row_value_references(text, metric, known_components=(), *, single_metric=True):
    """Attach a metric value to an explicit row cue.

    Bare numbers are deliberately excluded. They are commonly table numbers,
    pixel offsets or list indices and previously made ``20 px`` look like an
    F1 value. A candidate must be paired with the metric or carry a percent.
    """
    candidates = []
    for pair in _metric_value_pairs(text):
        if pair['metric'] == metric:
            candidates.append({'value': pair['value'], 'start': pair['value_start']})
    if single_metric:
        candidates.extend({'value': match.group('value'), 'start': match.start()}
                          for match in NUMBER.finditer(text))
    candidates = list({(item['value'], item['start']): item for item in candidates}.values())
    result = []
    rows = sorted(_row_references(text, known_components), key=lambda item: item['start'])
    for index, row in enumerate(rows):
        next_cue = rows[index + 1]['start'] if index + 1 < len(rows) else len(text)
        after = [item for item in candidates
                 if row['end'] <= item['start'] < next_cue and item['start'] - row['end'] <= 140]
        if not row['combo']:
            after = [item for item in after
                     if not re.search(r'[)）]', text[row['end']:item['start']])]
        local = after or [item for item in candidates if abs(item['start'] - row['start']) <= 140]
        if not row['combo'] and not after:
            # Baseline/without is directional: a prior or out-of-parenthesis
            # number cannot establish that row's metric value.
            continue
        if not local:
            continue
        if not single_metric:
            # Every explicitly labeled metric in this configuration clause
            # belongs to the row. Never distribute bare percentages across
            # several metrics or borrow values from another row cue.
            result.extend({**row, 'value': item['value']} for item in after)
            continue
        nearest = min(local, key=lambda item: abs(item['start'] - row['end']))
        if row['combo'] and len(after) >= 2:
            ordered = sorted(after, key=lambda item: item['start'])
            first, second = ordered[:2]
            lead = text[row['end']:first['start']]
            transition = text[first['start']:second['start']]
            if (re.search(r'从|\bfrom\b', lead, re.I) and
                    re.search(r'变为|变到|提升至|提高到|下降至|到|\bto\b', transition, re.I)):
                nearest = second
        result.append({**row, 'value': nearest['value']})
    unique = {(item['combo'], item['value']): item for item in result}
    return list(unique.values())


def _table_row_cells(table, metric):
    values = []
    for row in table['rows']:
        if not isinstance(row, dict) or not isinstance(row.get('cells'), dict):
            continue
        combo = frozenset(_normalise_token(str(item)) for item in row.get('checked_columns') or [])
        for key, value in row['cells'].items():
            if _metric_name(str(key)) == metric:
                values.append({'combo': combo, 'value': value,
                               'row_label': row.get('row_label')})
    return values


def _table_row_checks(fact, rows):
    """Check explicit ablation-row combinations against named-cell rows."""
    fact_metrics = {item['metric'] for item in metric_references(fact['text'])}
    if not fact_metrics:
        return []
    structures = []
    for row in rows:
        for table in _structured_tables(row['content']):
            structures.append({**table, 'scope_datasets': sorted(
                {_key(item['dataset']) for item in dataset_matches(row['content'])})})
    if not structures:
        return []
    requested_tables = {item['table'] for item in table_references(fact['text'])}
    selected = ([item for item in structures if item['table'] in requested_tables]
                if requested_tables else structures)
    if not selected:
        selected = structures
    known_components = {_normalise_token(str(component)) for table in selected
                        for row in table['rows'] if isinstance(row, dict)
                        for component in row.get('checked_columns') or []}
    checks = []
    for metric in sorted(fact_metrics):
        cells = [{**cell, 'scope_datasets': table['scope_datasets']}
                 for table in selected for cell in _table_row_cells(table, metric)]
        row_values = _row_value_references(fact['text'], metric, known_components,
                                           single_metric=len(fact_metrics) == 1)
        for item in row_values:
            matching = [cell for cell in cells if cell['combo'] == item['combo']
                        and _cell_value_present(cell['value'], item['value'])]
            if matching:
                status = 'MATCH'
            elif any(_cell_value_present(cell['value'], item['value']) for cell in cells):
                status = 'MISMATCH'
            else:
                status = 'UNRESOLVED'
            datasets = sorted({dataset for cell in matching for dataset in cell['scope_datasets']})
            checks.append(_binding_check('table_row', f"{sorted(item['combo'])} {metric}={item['value']}",
                                         status, blocking=True, metric=metric, value=item['value'],
                                         expected_checked_columns=sorted(item['combo']),
                                         **({'scope_datasets': datasets} if datasets else {})))
    return checks


def _binding_check(kind, claim, status, *, blocking=True, **extra):
    return {'kind': kind, 'claim': claim, 'status': status, 'blocking': blocking, **extra}


def check_bindings(fact, evidence):
    """Check explicit table/metric/method/causal anchors in cited evidence.

    This is intentionally a necessary-condition check. A missing method name in
    an otherwise generic result snippet is reported but is non-blocking because
    many paper extractors omit the title; an explicit conflicting name, table or
    metric is blocking. Explicit table/metric/causal assertions without any
    corresponding source anchor are blocking and become ``UNCERTAIN``.
    """
    rows = [row for row in evidence if row['evidence_id'] in fact['evidence_ids']]
    source_text = '\n'.join(row['content'] for row in rows)
    checks = []
    source_tables = {item['table'] for item in table_references(source_text)}
    for item in table_references(fact['text']):
        if item['table'] in source_tables:
            status, blocking = 'MATCH', True
        elif source_tables:
            status, blocking = 'MISMATCH', True
        else:
            status, blocking = 'UNRESOLVED', True
        checks.append(_binding_check('table', item['text'], status, blocking=blocking,
                                     normalized_table=item['table']))

    source_metrics = _source_metric_keys(source_text)
    fact_metrics = metric_references(fact['text'])
    fact_metric_values = _metric_value_pairs(fact['text'])
    # Structured row/column binding belongs to cited SOURCE_FACT premises.
    # A cross-table inference can legitimately mention values from two tables;
    # forcing all values through one selected table creates a false mismatch.
    if fact.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT':
        checks.extend(_table_cell_checks(fact, rows, fact_metric_values))
        checks.extend(_table_row_checks(fact, rows))
    for item in fact_metrics:
        metric = item['metric']
        pair = next((candidate for candidate in fact_metric_values
                     if candidate['metric'] == metric and
                     candidate['metric_start'] == item['start']), None)
        compatible = _metric_compatible(metric, source_metrics)
        if compatible and (pair is None or _value_present(source_text, pair['value'])):
            status, blocking = 'MATCH', True
        elif compatible and pair is not None:
            # A metric label may occur in one cited passage while the claimed
            # value is absent. Do not let a nearby but different number pass;
            # classify the missing value as uncertainty rather than a proven
            # contradiction (the source may use a different table row).
            status, blocking = 'UNRESOLVED', True
        elif source_metrics:
            status, blocking = 'MISMATCH', True
        else:
            # Generic accuracy is often expressed only by a table caption or a
            # percentage column. Preserve compatibility while still recording
            # the missing explicit metric anchor for diagnostics.
            status, blocking = 'UNRESOLVED', metric not in _WEAK_METRICS
        checks.append(_binding_check('metric', item['text'], status, blocking=blocking,
                                     metric=metric, strength=item['strength'],
                                     value=pair['value'] if pair is not None else None))

    source_methods = _source_method_keys(source_text)
    titles = [(m.start(), m.end()) for m in re.finditer(r'《[^》]+》', fact['text'])]
    for item in _method_references(fact['text']):
        if any(start < item['start'] and item['end'] < end for start, end in titles):
            # A document title identifies its paper, not an algorithm in this
            # row. Source ownership and semantic citation audit still apply.
            checks.append(_binding_check('paper_title', item['method'], 'UNRESOLVED', blocking=False))
            continue
        key = item['key']
        if key in source_methods:
            status, blocking = 'MATCH', True
        elif source_methods:
            # Absence is not a contradictory relation. For numerical assertions
            # require further verification; for prose this heuristic is advisory.
            status, blocking = 'UNRESOLVED', bool(NUMBER.search(fact['text']))
        else:
            # A short table-value snippet may not repeat the model name. This
            # is useful evidence of a coverage gap, but not a false rejection.
            status, blocking = 'UNRESOLVED', False
        checks.append(_binding_check('method', item['method'], status, blocking=blocking))

    if fact.get('kind', 'SOURCE_FACT') == 'SOURCE_FACT':
        fact_causal = _causal_families(fact['text'])
        source_causal = _causal_families(source_text)
        for family in sorted(fact_causal - source_causal):
            checks.append(_binding_check('causal', family, 'UNRESOLVED',
                                         blocking=_empirical_outcome_claim(fact['text']),
                                         source_families=sorted(source_causal)))
        for family in sorted(fact_causal & source_causal):
            checks.append(_binding_check('causal', family, 'MATCH', blocking=True,
                                         source_families=sorted(source_causal)))
    return {'fact_id': fact['id'], 'checks': checks,
            'coverage': 'explicit_bindings_necessary_condition_only',
            'recognized_binding_count': len(checks)}


def _dataset(match):
    return match.group('en') or match.group('zh')


def _key(text):
    return text.casefold()


def dataset_matches(text):
    """Return explicit dataset anchors, including `A and B datasets`.

    The normal pattern intentionally accepts one name. The multi-name form is
    expanded here so an aggregate claim cannot silently inherit only the first
    dataset mentioned in its caption.
    """
    result = []
    for match in DATASET.finditer(text):
        result.append({'dataset': _dataset(match), 'start': match.start(), 'end': match.end()})
    for match in NAMED_DATASET_ZH.finditer(text):
        result.append({'dataset': match['name'], 'start': match.start(), 'end': match.end()})
    for match in BARE_DATASET.finditer(text):
        result.append({'dataset': match.group('en'), 'start': match.start(), 'end': match.end()})
    for match in MULTI_DATASET.finditer(text):
        names = match.group('names')
        for name in re.finditer(NAME, names):
            result.append({'dataset': name.group(0),
                           'start': match.start('names') + name.start(),
                           'end': match.start('names') + name.end()})
    unique = {(item['dataset'].casefold(), item['start']): item for item in result}
    return list(unique.values())


def assertions(text):
    """Dataset must precede the value in the same local claim clause.

    A bare `on X` is accepted only when X has an explicit dataset-like capital
    or numeric token; ordinary lowercase words are not guessed as datasets.
    Names are never translated and attribution never crosses semicolons.
    Unsupported grammar is exposed as uncovered.
    """
    result = []
    for clause in re.finditer(r'[^;；。\n]+', text):
        names = dataset_matches(clause[0])
        for number in NUMBER.finditer(clause[0]):
            before = [m for m in names if m['end'] <= number.start()]
            if not before:
                # Only an explicit reverse attribution can bind a later name;
                # arbitrary nearby dataset mentions must not acquire the value.
                reverse = re.match(r'\s*(?:对应|属于)\s*', clause[0][number.end():])
                after = [m for m in names if m['start'] >= number.end()]
                name = None
                if reverse:
                    name = next((m for m in after
                                 if m['start'] == number.end() + reverse.end()), None)
                if name is None:
                    # Common Chinese answer order: "96.68% 是模型在 xBD 数据集上取得".
                    # Require both the copula and a dataset relation before the
                    # explicit dataset name; mere later co-occurrence is unsafe.
                    for candidate in after:
                        bridge = clause[0][number.end():candidate['start']]
                        if (len(bridge) <= 100 and re.match(r'^\s*是', bridge) and
                                re.search(r'(?:在|来自|属于|对应)\s*$', bridge)):
                            name = candidate
                            break
                if name is None:
                    continue
            else:
                name = before[-1]
            # Very distant mentions cannot safely scope a following number.
            if number.start() - name['end'] > 160:
                continue
            result.append({'dataset': name['dataset'], 'value': number['value'], 'unit': '%',
                           'claim_start': clause.start() + name['start'],
                           'value_start': clause.start() + number.start(),
                           'value_end': clause.start() + number.end()})
    return result


def source_occurrences(row, value):
    text = row['content']
    headings = list(HEADING.finditer(text))
    result = []
    for number in NUMBER.finditer(text):
        if Decimal(number['value']) != Decimal(value):
            continue
        previous = [m for m in headings if m.start() < number.start()]
        heading = previous[-1] if previous else None
        next_heading = next((m.start() for m in headings if m.start() > number.start()), len(text))
        # Restrict scope to the current caption-led block. In a captionless
        # passage only the current sentence can establish the dataset.
        boundaries = [0] + [m.end() for m in BOUNDARY.finditer(text)] + [len(text)]
        start = max(p for p in boundaries if p <= number.start())
        end = min(p for p in boundaries if p >= number.end())
        sentence = text[start:end]
        anchors = [{**m, 'start': start + m['start'], 'end': start + m['end'],
                    'kind': 'same_sentence'} for m in dataset_matches(sentence)]
        table = None
        if not anchors and heading and number.start() - heading.start() <= 1800:
            table = heading[1]
            # Caption has to supply one explicit dataset before its first full
            # stop. Never inherit an arbitrary earlier dataset from the paper.
            caption_end = min(next_heading, text.find('.', heading.end()) + 1 or len(text))
            caption_end = min(caption_end, heading.end() + 450)
            caption = text[heading.end():caption_end]
            anchors = [{'dataset': m['dataset'], 'start': heading.end() + m['start'],
                        'end': heading.end() + m['end'], 'kind': 'table_caption'}
                       for m in dataset_matches(caption)]
            # If intervening prose mentions another dataset, scope is ambiguous.
            for m in dataset_matches(text[caption_end:number.start()]):
                anchors.append({'dataset': m['dataset'], 'start': caption_end + m['start'],
                                'end': caption_end + m['end'], 'kind': 'intervening_scope'})
        scopes = {_key(a['dataset']) for a in anchors}
        result.append({'evidence_id': row['evidence_id'], 'value_start': number.start(),
                       'value_end': number.end(), 'value_text': number[0], 'table': table,
                       'anchors': anchors, 'scope': next(iter(scopes)) if len(scopes) == 1 else None,
                       'scope_status': 'explicit' if len(scopes) == 1 else 'ambiguous' if scopes else 'missing'})
    return result


def deterministic_numeric_attribution(question, evidence):
    """Resolve one requested percentage through explicit caption and respectively order."""
    requested = list(NUMBER.finditer(question))
    if (len(requested) != 1 or
            not re.search(r'数据集|dataset', question, re.I) or
            not re.search(r'指标|metric', question, re.I) or
            not re.search(r'哪张表|which\s+table|table', question, re.I)):
        return None
    value = requested[0]['value']
    candidates = []
    for row in evidence:
        text = row['content']
        for occurrence in source_occurrences(row, value):
            if occurrence['scope_status'] != 'explicit' or not occurrence['table']:
                continue
            position = occurrence['value_start']
            boundaries = [0] + [match.end() for match in BOUNDARY.finditer(text)] + [len(text)]
            segment_index = max(index for index, start in enumerate(boundaries[:-1])
                                if start <= position)
            sentence_start = boundaries[segment_index]
            sentence_end = boundaries[segment_index + 1]
            sentence = text[sentence_start:sentence_end]
            numbers = list(NUMBER.finditer(sentence))
            local_position = position - sentence_start
            index = next((i for i, item in enumerate(numbers)
                          if item.start() == local_position), None)
            metric = None
            direct_pairs = _metric_value_pairs(sentence)
            for pair in direct_pairs:
                if Decimal(pair['value']) == Decimal(value):
                    metric = pair['metric']
                    break
            if metric is None and index is not None and re.search(r'respectively|分别', sentence, re.I):
                previous_end = sentence_start
                previous_start = boundaries[max(0, segment_index - 1)]
                previous_pairs = _metric_value_pairs(text[previous_start:previous_end])
                if len(previous_pairs) == len(numbers) and index < len(previous_pairs):
                    metric = previous_pairs[index]['metric']
            if metric is None:
                continue
            dataset = next((anchor['dataset'] for anchor in occurrence['anchors']
                            if _key(anchor['dataset']) == occurrence['scope']), occurrence['scope'])
            candidates.append({'evidence_id': row['evidence_id'], 'dataset': dataset,
                               'metric': metric, 'table': occurrence['table'], 'value': value})
    unique = {(item['dataset'].casefold(), item['metric'], str(item['table'])): item for item in candidates}
    if len(unique) != 1:
        return None
    result = next(iter(unique.values()))
    labels = {'overall_accuracy': '总体准确率（Overall Accuracy）', 'accuracy': '准确率（Accuracy）',
              'miou': '平均交并比（mIoU）', 'f1overall': 'F1overall'}
    metric = labels.get(result['metric'], result['metric'])
    table = f"Table {result['table']}"
    if re.search(r'[\u4e00-\u9fff]', question):
        answer = (f"{value}% 对应 {result['dataset']} 数据集的{metric}，位于 {table}。"
                  '正文使用 respectively 按前句指标顺序对应这组数值。')
    else:
        answer = (f"{value}% is the {metric} result on {result['dataset']} in {table}. "
                  'The source binds the values by the metric order followed by “respectively.”')
    return {**result, 'answer': f"{answer} [{result['evidence_id']}]"}


def check_fact(fact, evidence):
    claims = assertions(fact['text'])
    rows = [row for row in evidence if row['evidence_id'] in fact['evidence_ids']]
    checks = []
    for claim in claims:
        matches = [m for row in rows for m in source_occurrences(row, claim['value'])]
        scopes = {m['scope'] for m in matches if m['scope']}
        unknown = any(m['scope'] is None for m in matches)
        if not matches or unknown or len(scopes) != 1:
            status = 'UNRESOLVED'
        elif _key(claim['dataset']) in scopes:
            status = 'SCOPE_MATCH_ONLY'
        else:
            status = 'SCOPE_MISMATCH'
        checks.append({**claim, 'status': status, 'occurrences': matches})
    bindings = check_bindings(fact, evidence)
    # A verified named-cell table may carry a numeric value without a percent
    # sign in each cell. Let its explicit table caption establish scope when
    # the table cell and the requested dataset agree; plain prose still stays
    # unresolved under the stricter percentage parser above.
    for claim in checks:
        if claim['status'] != 'UNRESOLVED':
            continue
        cell_matches = [item for item in bindings['checks']
                        if item['kind'] in {'table_cell', 'table_row'} and item['status'] == 'MATCH'
                        and Decimal(str(item['value'])) == Decimal(str(claim['value']))]
        scopes = {dataset for item in cell_matches for dataset in item.get('scope_datasets', [])}
        if not matches and len(scopes) == 1:
            claim['status'] = ('SCOPE_MATCH_ONLY' if _key(claim['dataset']) in scopes else 'SCOPE_MISMATCH')
    return {'fact_id': fact['id'], 'checks': checks,
            'binding_checks': bindings['checks'],
            'unparsed_percentage_count': len(list(NUMBER.finditer(fact['text']))) - len(claims),
            'recognized_binding_count': bindings['recognized_binding_count'],
            'coverage': 'recognized_scope_and_explicit_bindings_only_not_full_entailment'}


def apply_scope_guard(draft, audit, evidence, *, mode='enforce'):
    if mode not in {'off', 'shadow', 'enforce'}:
        raise ValueError('Unknown numerical scope mode')
    effective = deepcopy(audit)
    report = {'version': VERSION, 'mode': mode, 'facts': [], 'changed_fact_ids': []}
    if mode == 'off':
        return effective, report
    by_id = {f['id']: f for f in draft['facts']}
    for assessment in effective['assessments']:
        fact = by_id[assessment['fact_id']]
        result = check_fact(fact, evidence)
        report['facts'].append(result)
        statuses = {c['status'] for c in result['checks']}
        binding_checks = result.get('binding_checks', [])
        blocking_binding_checks = [item for item in binding_checks if item.get('blocking')]
        binding_statuses = {item['status'] for item in blocking_binding_checks}
        if mode != 'enforce' or assessment['verdict'] != 'SUPPORTED':
            continue
        scope_mismatch = 'SCOPE_MISMATCH' in statuses
        scope_unresolved = 'UNRESOLVED' in statuses
        binding_mismatch = 'MISMATCH' in binding_statuses
        binding_unresolved = 'UNRESOLVED' in binding_statuses
        if not (scope_mismatch or scope_unresolved or binding_mismatch or binding_unresolved):
            continue
        # The model's verdict remains available in the raw report. This guard
        # never upgrades a rejection and never fabricates a corrected answer.
        assessment['model_verdict'] = assessment['verdict']
        assessment['verdict'] = 'UNSUPPORTED' if scope_mismatch or binding_mismatch else 'UNCERTAIN'
        if scope_mismatch:
            assessment['numerical_scope_status'] = 'mismatch'
        elif scope_unresolved:
            assessment['numerical_scope_status'] = 'unresolved'
        if binding_mismatch:
            assessment['claim_binding_status'] = 'mismatch'
        elif binding_unresolved:
            assessment['claim_binding_status'] = 'unresolved'
        report['changed_fact_ids'].append(fact['id'])
    if report['changed_fact_ids']:
        if effective['coverage'] == 'COMPLETE':
            effective['coverage'] = 'PARTIAL'
        chinese = any(re.search(r'[\u4e00-\u9fff]', by_id[fid]['text']) for fid in report['changed_fact_ids'])
        if any(item.get('claim_binding_status') for item in effective['assessments']):
            limitation = ('部分结论的表格、指标、方法或因果绑定与所引原文不符或尚不能确认，未作为可靠结论展示。'
                          if chinese else
                          'Some table, metric, method or causal bindings conflict with or are unresolved in the cited source and are withheld.')
        else:
            limitation = ('部分数值结论的数据集归属与原文不符或尚不能确认，未作为可靠结论展示。' if chinese else
                          'Some numerical claims have conflicting or unresolved dataset attribution and are withheld.')
        effective['missing_aspects'] = list(dict.fromkeys(effective.get('missing_aspects', []) + [limitation]))
    return effective, report
