"""Build a comparison from jointly audited records, never from summary prose."""
from copy import deepcopy
import hashlib
import json
import re
import unicodedata

VERSION = 'paper-comparison-v2-single-value'
WRITER_SCHEMA = ('{"facts":[{"id":"f1","kind":"SOURCE_FACT","text":"本文方法：方法名",'
    '"comparison_field":"method","comparison_value":"方法名",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]}],'
    '"direct_answer_id":"f1","limitations":[],"comparison_records":'
    '[{"record_id":"r1","fields":{"method":"f1"},"missing_fields":'
    '["module","task","dataset","split","protocol","metric","result"]}]}')
FIELDS = {
    'method': '本文方法', 'module': '方法模块', 'task': '本文实验任务',
    'dataset': '本文实验数据集', 'split': '数据划分', 'protocol': '训练与评估条件',
    'metric': '指标口径', 'result': '本文报告结果',
}
CONDITIONS = ('task', 'dataset', 'split', 'protocol', 'metric')
WRITER_POLICY = '''
The mandatory output contract is comparison-v1, NOT the ordinary flat answer.
Always return comparison_records, including [] if no coherent record is supported.
Extract this paper's methods and its OWN experimental records, not prior-work
dataset definitions. Return comparison_records (at most 4):
[{"record_id":"r1","fields":{"method":"f1","task":"f2","dataset":"f3"},
  "missing_fields":["split","protocol","metric","result","module"]}].
Allowed fields and exact text labels are supplied in comparison_fields.
Every fact must be a SOURCE_FACT with comparison_field and comparison_value;
text must be the exact field label followed by Chinese colon and comparison_value.
The value must be a short source-supported statement, not a guessed canonical
name. Fields reference fact IDs, never a second unreviewed value. Each fact must
be referenced; every absent field must be in missing_fields. Keep the 12-fact cap;
share genuinely identical facts across records. Each record represents ONE method
variant and ONE experimental setting, not a blend of different tables, datasets,
training stages or related-work results. Preserve dataset version, official/custom
split, supervision, training/evaluation settings and metric aggregation/unit.
Do not infer unspecified conditions. If the current passages do not support a
coherent record, return no records and explain the specific gap in limitations.
Do not rank methods or infer SOTA from these records. Do not hide missing records.
'''


def writer_instruction(base):
    """Replace the flat schema example, preserving the source/language policies."""
    prefix, rest = base.split('Return JSON: ', 1)
    _, policies = rest.split('The designated direct answer', 1)
    return prefix + 'Return JSON: ' + WRITER_SCHEMA + '. The designated direct answer' + policies + '\n' + WRITER_POLICY
AUDIT_POLICY = '''
Also independently check comparison_records AS JOINT EXPERIMENTAL RECORDS.
Return record_assessments: [{"record_id":"r1","verdict":"SUPPORTED|UNCERTAIN|UNSUPPORTED",
"reason":"brief original-source justification"}], one per record.
A row is SUPPORTED only when the originals establish that ALL its populated fields
belong to the SAME paper method/variant, dataset, split, stage and metric scope.
Individually true facts from different experiments do not establish a valid row.
Check own experiments versus prior work; training versus evaluation; binary versus
semantic change detection; macro/micro/per-class versus overall metrics.
Missing fields remain missing, not inferred. The field labels assert experimental
roles, so a background mention cannot support an own-use label. The result and
metric must describe the same quantity. Do not approve a row merely because its
fact IDs/citations exist. A row verdict does not replace individual fact verdicts.
SUPPORTED assesses joint attribution of ONLY POPULATED fields, not completeness.
An omitted field is not evidence of misattribution: do not reject a partial record
merely because split, dataset, task or another field is absent. Reject populated
fields with wrong roles or incompatible experiments, not merely because a field is absent.
For method-only records, verify this paper's own method; do not infer experiments.
Completeness and eligibility for cross-paper comparison are separate program checks.
'''

FIELD_POLICY = '''
Verify column meaning, not only whether individual sentences appear in sources.
Dataset must identify this experiment's dataset, not unnamed images or a list of
all benchmarks in the paper. Split means training/validation/test allocation or
an explicitly documented full-data protocol, NOT crop size/overlap/preprocessing.
Metrics and results must describe the row's task; do not mix module segmentation
quality with change-detection accuracy. Unknown/not-reported conditions are gaps,
not supported populated cells. Multiple datasets/settings need separate rows.
True statements assigned to wrong columns do NOT establish joint support.
'''

AUDIT_COVERAGE_POLICY = '''
Comparison output has TWO INDEPENDENT decisions:
1. record_assessments judge the factual roles and joint attribution of the fields
   actually present in comparison_records. Missing columns are not assertions.
2. coverage/missing_aspects judge unanswered requirements and comparison_coverage.
Do NOT transfer coverage failures into record verdicts. For example a sourced
method/task pair may be SUPPORTED while coverage is PARTIAL and split is missing.
Conversely individually supported fields from incompatible experiments must be
UNSUPPORTED even when all columns are present. Do not impute missing conditions.
Return record_assessments for each record_id, with reason and then verdict.
SUPPORTED does not authorize rankings or certify a complete experimental protocol.
'''


def comparison_audit_view(draft):
    """Keep attribution assertions separate from losslessly retained coverage gaps."""
    rows = draft['comparison_records']
    return {
        'comparison_records': [{'record_id': r['record_id'], 'fields': deepcopy(r['fields'])} for r in rows],
        'comparison_coverage': [{'record_id': r['record_id'],
            'missing_fields': sorted(set(FIELDS) - set(r['fields'])),
            'format_gaps': list(r.get('format_gaps', []))} for r in rows],
        'comparison_fields': FIELDS,
    }


def field_policy_gap(field, text, row, facts):
    """Reject clear type errors; do not extract guessed replacement values."""
    if re.search(r'未明确|未报告|未说明|无法确定|未命名|待核对|未知|\b(?:unknown|unspecified|unreported)\b|not (?:specified|reported|named)', text, re.I):
        return '值包含缺证说明，不作为已知条件'
    if field == 'split' and re.search(r'patch|crop|overlap|裁剪|重叠', text, re.I):
        partitions = sum(bool(re.search(pattern, text, re.I)) for pattern in (
            r'训练集|training(?: set)?|\btrain\b', r'验证集|validation|\bval\b', r'测试集|testing|\btest\b'))
        if partitions < 2 and not re.search(r'held.out|cross.validation|fold|交叉验证|完整数据.*评估|全量数据.*评估|full.data.*evaluat', text, re.I):
            return '仅含裁剪/重叠设置，不是训练验证测试划分'
    if field == 'dataset' and (re.search(r'共\s*\d+\s*个.*数据集|\bdatasets\s+(?:include|including)', text, re.I)
            or re.search(r'数据集(?:包括|包含)', text) and '、' in text):
        return '包含多个数据集汇总，尚未定位单个实验设置'
    task = facts.get(row.get('fields', {}).get('task'), {}).get('text', '')
    if (field == 'metric' and re.search(r'变化检测|change detection', task, re.I)
            and re.search(r'超像素分割|superpixel.*(?:quality|segmentation)|achievable segmentation accuracy', text, re.I)):
        return '混入另一任务的模块分割指标，需要拆分记录'
    return None

SINGLE_VALUE_SCHEMA = ('{"facts":[{"id":"f1","kind":"SOURCE_FACT","text":"本文方法名",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]},'
    '{"id":"f2","kind":"SOURCE_FACT","text":"本文实验任务",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]},'
    '{"id":"f3","kind":"SOURCE_FACT","text":"该实验数据集名",'
    '"evidence_ids":["exact supplied ID"],"depends_on":[]}],"direct_answer_id":"f1",'
    '"limitations":[],"comparison_records":[{"record_id":"r1",'
    '"fields":{"method":"f1","task":"f2","dataset":"f3"}}]}')
SINGLE_VALUE_POLICY = '''
Use ONLY the comparison-v2 schema shown above. Each cell references one fact ID.
Every fields value is an EXACT ID in this response's facts, NOT cell prose or an
evidence ID. The example placeholders are not facts: replace them from originals.
The fact text itself is the short cell value (at most 400 characters), with exact
supplied evidence IDs. Do NOT generate comparison_value or a second copy of text.
Allowed field names are in comparison_fields. Return comparison_records, even
when empty. Each row describes ONE method/experiment setting established by the
originals, not mixed tasks, variants, tables or prior work. Shared facts may be
referenced by multiple rows only under the same field. Missing fields are omitted
from fields; the program derives gaps. Never invent a condition to fill a cell.
Keep the 12-fact cap and at most four rows. For multiple settings share genuinely
common facts, preserving dataset versions, splits, supervision and metric units.
Only SOURCE_FACT cells are allowed. Do not infer rankings or SOTA. Put unresolved
requirements in limitations, not into uncited facts. Do not copy placeholder IDs.
Partial records are allowed: omit unavailable fields, never insert unknown values.
Do not mix multiple datasets or settings to fill a row. Cropping is preprocessing,
not a train/validation/test allocation. Module quality is not task accuracy.
'''


def single_value_instruction(base):
    prefix, rest = base.split('Return JSON: ', 1)
    _, policies = rest.split('The designated direct answer', 1)
    return prefix + 'Return JSON: ' + SINGLE_VALUE_SCHEMA + '. The designated direct answer' + policies + '\n' + SINGLE_VALUE_POLICY


def prepare_single_value_records(value):
    """Quarantine malformed bindings, never guess IDs or reinterpret cell text."""
    value = deepcopy(value)
    value['comparison_contract'] = 'single-value-v2'
    rows = value.get('comparison_records')
    if not isinstance(rows, list):
        return value  # strict validator reports the missing contract
    facts = {f['id']: f for f in value['facts']}
    assignments = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get('fields'), dict):
            for field, fid in row['fields'].items():
                if field in FIELDS and isinstance(fid, str) and fid in facts:
                    assignments.setdefault(fid, set()).add(field)
    retained, gaps = [], []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('fields'), dict):
            retained.append(row)
            continue
        valid, rejected = {}, []
        for field, fid in row['fields'].items():
            if (field not in FIELDS or not isinstance(fid, str) or fid not in facts
                    or len(assignments.get(fid, set())) != 1
                    or facts[fid].get('kind', 'SOURCE_FACT') != 'SOURCE_FACT'
                    or len(facts[fid]['text']) > 400):
                rejected.append(FIELDS.get(field, '未识别字段'))
            else:
                policy_gap = field_policy_gap(field, facts[fid]['text'], row, facts)
                if policy_gap:
                    rejected.append(FIELDS[field] + '（' + policy_gap + '）')
                else:
                    valid[field] = fid
        notes = ['输出绑定无效，已隔离而未猜测补齐：' + '、'.join(rejected)] if rejected else []
        gaps.extend(notes)
        if valid:
            retained.append({**row, 'fields': valid, 'missing_fields': sorted(set(FIELDS) - set(valid)),
                             'format_gaps': notes})
    value['comparison_records'] = retained
    value['comparison_format_gaps'] = list(dict.fromkeys(gaps))
    return value


def validate_records(draft):
    rows = draft.get('comparison_records')
    if not isinstance(rows, list) or len(rows) > 4:
        raise ValueError('Comparison requires a bounded comparison_records list')
    facts = {f['id']: f for f in draft['facts']}
    seen, used = set(), set()
    single_value = draft.get('comparison_contract') == 'single-value-v2'
    fact_fields = {}
    for row in rows:
        rid = row.get('record_id') if isinstance(row, dict) else None
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError('Comparison record IDs must be unique')
        seen.add(rid)
        fields, missing = row.get('fields'), row.get('missing_fields')
        if (not isinstance(fields, dict) or not fields or set(fields) - set(FIELDS)
                or not isinstance(missing, list) or any(not isinstance(x, str) for x in missing)
                or len(missing) != len(set(missing)) or set(missing) != set(FIELDS) - set(fields)):
            raise ValueError('Comparison must account for every populated and missing field')
        for field, fid in fields.items():
            if not isinstance(fid, str) or fid not in facts:
                raise ValueError('Comparison fields must reference current fact IDs')
            fact = facts[fid]
            if single_value:
                if (fact.get('kind', 'SOURCE_FACT') != 'SOURCE_FACT' or len(fact['text']) > 400
                        or fid in fact_fields and fact_fields[fid] != field):
                    raise ValueError('A comparison fact must be a short source fact bound to one field')
                fact_fields[fid] = field
                used.add(fid)
                continue
            value = fact.get('comparison_value')
            if (fact.get('kind', 'SOURCE_FACT') != 'SOURCE_FACT'
                    or fact.get('comparison_field') != field
                    or not isinstance(value, str) or not value.strip() or len(value) > 400
                    or fact['text'] != FIELDS[field] + '：' + value):
                raise ValueError('Comparison cell must be identical to its source-audited fact value')
            used.add(fid)
    if not single_value and set(facts) != used:
        raise ValueError('All comparison facts must belong to a record')


def validate_record_audit(audit, draft):
    rows = audit.get('record_assessments')
    expected = {r['record_id'] for r in draft['comparison_records']}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(r, dict) for r in rows)
            or {r.get('record_id') for r in rows} != expected):
        raise ValueError('Audit must assess each comparison record exactly once')
    for row in rows:
        if (row.get('verdict') not in {'SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN'}
                or not isinstance(row.get('reason'), str) or not row['reason'].strip()):
            raise ValueError('Joint comparison record requires an evidence-based verdict')


def reviewed_records(report, sha, title):
    """Only publish cells and joint bindings that survived the effective audit."""
    if report.get('status') != 'REVIEW_REQUIRED':
        return []
    draft, audit = report['draft'], report['audit']
    validate_records(draft)
    validate_record_audit(audit, draft)
    facts = {f['id']: f for f in draft['facts']}
    assessments = {r['fact_id']: r for r in audit['assessments']}
    joints = {r['record_id']: r for r in audit['record_assessments']}
    records = []
    for row in draft['comparison_records']:
        cells = {}
        joint_ok = joints[row['record_id']]['verdict'] == 'SUPPORTED'
        for field, fid in row['fields'].items():
            fact, assessment = facts[fid], assessments[fid]
            if joint_ok and assessment['verdict'] == 'SUPPORTED' and assessment.get('relevance') != 'TANGENTIAL':
                cells[field] = {'value': fact['text'] if draft.get('comparison_contract') == 'single-value-v2' else fact['comparison_value'], 'text': fact['text'],
                                'evidence_ids': fact['evidence_ids'],
                                'span_ids': assessment.get('span_ids', [])}
        identity = [sha, {k: v['value'] for k, v in cells.items()}]
        records.append({'record_id': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24],
                        'paper_sha256': sha, 'title': title, 'cells': cells,
                        'status': 'REVIEWED' if joint_ok and len(cells) == len(row['fields']) and not row.get('format_gaps') else 'PARTIAL' if cells else 'WITHHELD',
                        'missing_fields': sorted(set(FIELDS) - set(cells)),
                        'format_gaps': row.get('format_gaps', []),
                        'joint_verdict': joints[row['record_id']]['verdict']})
    return records


def normalized(value):
    # Exact normalized equality only; no acronym expansion or semantic merging.
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def assemble_comparison(records, papers):
    records = deepcopy(records)
    unique = {}
    for row in records:
        if not row['cells']:
            continue
        key = (row['paper_sha256'], tuple((k, normalized(v['value'])) for k, v in sorted(row['cells'].items())))
        if key not in unique:
            unique[key] = row
        else:
            for field, cell in row['cells'].items():
                old = unique[key]['cells'][field]
                for attr in ('evidence_ids', 'span_ids'):
                    old[attr] = list(dict.fromkeys(old[attr] + cell[attr]))
    rows = list(unique.values())
    bindings = {}
    for row in rows:
        cells = row['cells']
        if all(k in cells for k in ('method', *CONDITIONS, 'result')):
            key = (row['paper_sha256'], *(normalized(cells[k]['value']) for k in ('method', *CONDITIONS)))
            bindings.setdefault(key, []).append(row)
    conflicts = []
    for bound in bindings.values():
        if len({normalized(r['cells']['result']['value']) for r in bound}) > 1:
            ids = [r['record_id'] for r in bound]
            conflicts.append({'record_ids': ids, 'reason': '同一已记录条件下存在不同结果，需要回查原文。'})
            for row in bound:
                row['status'] = 'CONFLICT'
    buckets = {}
    for row in rows:
        cells = row['cells']
        if row['status'] == 'REVIEWED' and all(k in cells for k in ('method', *CONDITIONS, 'result')):
            key = tuple(normalized(cells[k]['value']) for k in CONDITIONS)
            buckets.setdefault(key, []).append(row)
    groups = []
    for bucket in buckets.values():
        if len({r['paper_sha256'] for r in bucket}) > 1:
            groups.append({'record_ids': [r['record_id'] for r in bucket],
                           'conditions': {k: bucket[0]['cells'][k]['value'] for k in CONDITIONS},
                           'status': 'MATCHED_RECORDED_CONDITIONS'})
    coverage = [{'paper_sha256': p['paper_sha256'], 'title': p['title'],
                 'record_count': sum(r['paper_sha256'] == p['paper_sha256'] for r in rows),
                 'gaps': list(dict.fromkeys(p.get('gaps', []) + [g for r in rows if r['paper_sha256'] == p['paper_sha256'] for g in r.get('format_gaps', [])] + [
                     FIELDS[k] + '待核对' for k in FIELDS if not any(
                         r['paper_sha256'] == p['paper_sha256'] and k in r['cells'] for r in rows)]))}
                for p in papers]
    return {'version': VERSION, 'records': rows, 'condition_groups': groups,
            'conflicts': conflicts, 'coverage': coverage,
            'withheld_record_count': sum(not r['cells'] for r in records),
            'semantic_correctness': 'MODEL_REVIEWED_NOT_INDEPENDENT_GROUND_TRUTH'}


def comparison_markdown(comparison):
    def escape(text):
        return str(text).replace('|', '\\|').replace('\n', ' ')
    lines = ['## 逐篇证据比较表', '|论文|' + '|'.join(FIELDS.values()) + '|',
             '|---|' + '|'.join('---' for _ in FIELDS) + '|']
    for row in comparison['records']:
        cells = []
        for field in FIELDS:
            cell = row['cells'].get(field)
            text = (escape(cell['value']) + ' ' + ' '.join(f'[{e}]' for e in cell['evidence_ids'])) if cell else '待核对'
            if row['status'] == 'CONFLICT' and field == 'result':
                text = '存在冲突：' + text
            cells.append(text)
        lines.append('|' + escape(row['title']) + '|' + '|'.join(cells) + '|')
    if not comparison['records']:
        lines.append('尚未获得通过字段与联合归属审核的记录。')
    lines += ['', '## 条件对照与缺口',
              f"当前有 {len(comparison['condition_groups'])} 组跨论文记录的任务、数据集、划分、条件与指标文本一致。"
              '这只建立已记录条件的对照，不证明所有条件等价，也不自动推断 SOTA。']
    for group in comparison['condition_groups']:
        lines.append('- 可对照的已记录条件：' + '；'.join(FIELDS[k] + '：' + escape(v) for k, v in group['conditions'].items()))
    for row in comparison['coverage']:
        if row['gaps']:
            lines.append('- ' + escape(row['title']) + '：' + '；'.join(escape(x) for x in row['gaps']))
    if comparison['conflicts']:
        lines.append(f"- {len(comparison['conflicts'])} 组结果存在冲突，已从条件对照组排除。")
    return '\n'.join(lines)
