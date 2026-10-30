"""Comparison records cross the same draft/audit interface as ordinary answers."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit, evidence_spans, validate_draft
from paperpilot_agentic.paper_comparison import (
    FIELDS, reviewed_records, assemble_comparison, comparison_markdown, validate_records,
)
from test_answer_audit import Model


def candidate(values=None, joint='SUPPORTED'):
    values = values or {'method': '模型甲', 'module': '双时相编码', 'task': '二值变化检测',
        'dataset': 'DeltaSet v1', 'split': '官方验证集', 'protocol': '监督学习，同一固定训练协议',
        'metric': '变化类像素 IoU (%)', 'result': '84.1%'}
    facts = [{'id': k, 'text': FIELDS[k] + '：' + v,
              'comparison_field': k, 'comparison_value': v, 'evidence_ids': ['e1']}
             for k, v in values.items()]
    draft = {'facts': facts, 'direct_answer_id': next(iter(values)), 'limitations': [],
             'comparison_records': [{'record_id': 'r1', 'fields': {k: k for k in values},
                                     'missing_fields': [k for k in FIELDS if k not in values]}]}
    evidence = [{'evidence_id': 'e1', 'content': '\n'.join(f['text'] for f in facts)}]
    sid = evidence_spans(evidence)[0]['span_id']
    audit = {'assessments': [{'fact_id': k, 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                              'span_ids': [sid]} for k in values],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'record_assessments': [{'record_id': 'r1', 'verdict': joint, 'reason': 'Joint fixture review'}]}
    return draft, evidence, audit


def record(sha, **changes):
    draft, _, audit = candidate()
    for field, value in changes.items():
        fact = next(f for f in draft['facts'] if f['id'] == field)
        fact.update(comparison_value=value, text=FIELDS[field] + '：' + value)
    return reviewed_records({'status': 'REVIEW_REQUIRED', 'draft': draft, 'audit': audit}, sha, sha)[0]


def test_true_fields_cannot_publish_a_row_with_unverified_joint_attribution():
    draft, evidence, audit = candidate(joint='UNCERTAIN')
    result = answer_then_audit('整理实验比较表', evidence, Model([draft, audit]), draft_contract='comparison-v1')
    assert result['status'] == 'REVIEW_REQUIRED'
    rows = reviewed_records(result, 'paper-a', 'Paper A')
    assert rows[0]['status'] == 'WITHHELD'
    assert not rows[0]['cells']


def test_unreviewed_cell_value_and_unknown_fact_binding_are_rejected():
    draft, _, _ = candidate()
    changed = deepcopy(draft)
    changed['facts'][0]['comparison_value'] = '另一个模型'
    with pytest.raises(ValueError, match='identical'):
        validate_records(changed)
    changed = deepcopy(draft)
    changed['comparison_records'][0]['fields']['method'] = 'invented'
    with pytest.raises(ValueError, match='current fact'):
        validate_records(changed)


def test_missing_joint_review_is_not_replaced_by_individual_fact_passes():
    draft, evidence, audit = candidate()
    del audit['record_assessments']
    result = answer_then_audit('整理实验比较表', evidence, Model([audit]), frozen_draft=draft,
                               draft_contract='comparison-v1', allow_repair=False)
    assert result['status'] == 'FAILED'
    assert not result.get('answer')


def test_condition_groups_require_all_explicit_conditions_and_never_expand_names():
    left, right = record('a'), record('b', method='模型乙', result='85.2%')
    comparison = assemble_comparison([left, right], [])
    assert len(comparison['condition_groups']) == 1
    del right['cells']['split']
    assert not assemble_comparison([left, right], [])['condition_groups']
    right = record('b', task='语义变化检测')
    assert not assemble_comparison([left, right], [])['condition_groups']
    right = record('b', dataset='DeltaSet v2')
    assert not assemble_comparison([left, right], [])['condition_groups']


def test_duplicates_merge_provenance_but_conflicting_results_are_not_ranked():
    left = record('a')
    duplicate = deepcopy(left)
    duplicate['cells']['result']['evidence_ids'] = ['e2']
    merged = assemble_comparison([left, duplicate], [])
    assert len(merged['records']) == 1
    assert merged['records'][0]['cells']['result']['evidence_ids'] == ['e1', 'e2']
    conflict = record('a', result='79.0%')
    other_paper = record('b', method='模型乙')
    comparison = assemble_comparison([left, conflict, other_paper], [])
    assert len(comparison['conflicts']) == 1
    assert not comparison['condition_groups']
    assert '存在冲突' in comparison_markdown(comparison)


def test_selected_paper_with_no_record_remains_in_coverage():
    comparison = assemble_comparison([record('a')], [
        {'paper_sha256': 'a', 'title': 'A'}, {'paper_sha256': 'b', 'title': 'B', 'gaps': ['没有正文']}])
    missing = comparison['coverage'][1]
    assert missing['record_count'] == 0
    assert '没有正文' in missing['gaps']
    assert len(missing['gaps']) == len(FIELDS) + 1


def test_writer_and_repair_use_one_explicit_comparison_schema():
    draft, evidence, audit = candidate()
    legacy = deepcopy(draft)
    del legacy['comparison_records']
    class Capture(Model):
        def __init__(self):
            super().__init__([legacy, draft, audit])
            self.prompts = []
        def complete(self, messages, **kwargs):
            self.prompts.append(deepcopy(messages))
            return super().complete(messages, **kwargs)
    model = Capture()
    result = answer_then_audit('整理实验比较表', evidence, model, draft_contract='comparison-v1')
    assert result['status'] == 'REVIEW_REQUIRED'
    schema = model.prompts[0][0]['content'].split('Return JSON: ', 1)[1].split('. The designated', 1)[0]
    assert 'comparison_records' in json.loads(schema)
    assert 'comparison_records' in model.prompts[1][-1]['content']
    assert 'comparison_field/comparison_value' in model.prompts[1][-1]['content']


def test_single_value_cells_use_only_the_exact_audited_fact_and_derive_gaps():
    draft, evidence, audit = candidate({'method': '模型甲'})
    draft['facts'][0]['comparison_value'] = '未经审核的诱导值'
    draft['comparison_records'][0]['missing_fields'] = []
    result = answer_then_audit('整理实验比较表', evidence, Model([draft, audit]), draft_contract='comparison-v2')
    assert result['status'] == 'REVIEW_REQUIRED'
    row = reviewed_records(result, 'a', 'A')[0]
    assert row['cells']['method']['value'] == draft['facts'][0]['text']
    assert '未经审核' not in comparison_markdown(assemble_comparison([row], []))
    assert set(row['missing_fields']) == set(FIELDS) - {'method'}


def test_single_value_fact_cannot_be_bound_to_two_different_fields():
    draft, evidence, audit = candidate({'method': '模型甲'})
    draft['comparison_records'][0]['fields']['dataset'] = 'method'
    prepared = validate_draft(draft, evidence, comparison='single-value-v2')
    assert not prepared['comparison_records']
    assert prepared['comparison_format_gaps']


def test_invalid_binding_is_quarantined_without_guessing_an_id_or_promoting_text():
    draft, evidence, audit = candidate({'method': '模型甲'})
    draft['comparison_records'][0]['fields']['dataset'] = '未经审核的数据集值'
    result = answer_then_audit('整理实验比较表', evidence, Model([draft, audit]), draft_contract='comparison-v2')
    assert result['status'] == 'REVIEW_REQUIRED'
    row = reviewed_records(result, 'a', 'A')[0]
    assert row['status'] == 'PARTIAL'
    assert set(row['cells']) == {'method'}
    assert 'dataset' in row['missing_fields']
    assert row['format_gaps']
    assert '未经审核' not in comparison_markdown(assemble_comparison([row], []))


def test_scientific_names_may_keep_original_language_only_in_structured_extraction():
    draft, evidence, _ = candidate({'method': 'ChangeMamba'})
    draft['facts'][0]['text'] = 'ChangeMamba'
    assert validate_draft(draft, evidence, question='整理方法', comparison='single-value-v2')
    with pytest.raises(ValueError, match='language'):
        validate_draft(draft, evidence, question='解释方法')


@pytest.mark.parametrize('field,value', [
    ('dataset', '未明确数据集名称，仅使用双时相 PolSAR 图像'),
    ('dataset', '共5个真实数据集，包括两个城市的图像'),
    ('split', '重叠裁剪 patch 256×256，overlap=0.29'),
    ('metric', '变化检测 F1 及超像素分割 BR/ASA'),
])
def test_source_supported_but_wrong_column_values_remain_explicit_gaps(field, value):
    draft, evidence, audit = candidate({'method': '模型甲', 'task': '二值变化检测', field: value})
    original = deepcopy(draft)
    result = answer_then_audit('整理比较表', evidence, Model([draft, audit]), draft_contract='comparison-v2')
    row = reviewed_records(result, 'a', 'A')[0]
    assert field not in row['cells']
    assert field in row['missing_fields']
    assert row['status'] == 'PARTIAL'
    assert row['format_gaps']
    assert result['draft']['facts'] == original['facts']
    assert draft == original  # never rewrite source facts or invent replacement values


def test_explicit_partition_is_not_rejected_merely_for_also_mentioning_crops():
    draft, evidence, audit = candidate({'method': '模型甲', 'dataset': 'SYSU-CD',
        'split': '训练集 12000 对，验证集 4000 对，测试集 4000 对；裁剪 patch 256×256'})
    result = answer_then_audit('整理比较表', evidence, Model([draft, audit]), draft_contract='comparison-v2')
    row = reviewed_records(result, 'a', 'A')[0]
    assert set(row['cells']) == {'method', 'dataset', 'split'}
    assert not row['format_gaps']


def test_comparison_prompts_distinguish_incomplete_records_from_false_attribution():
    from paperpilot_agentic.paper_comparison import (
        AUDIT_POLICY, SINGLE_VALUE_SCHEMA, SINGLE_VALUE_POLICY,
    )
    example = json.loads(SINGLE_VALUE_SCHEMA)
    assert {'method', 'task', 'dataset'} <= set(example['comparison_records'][0]['fields'])
    assert 'not evidence of misattribution' in AUDIT_POLICY
    assert 'not merely because a field is absent' in AUDIT_POLICY
    assert 'NOT cell prose' in SINGLE_VALUE_POLICY


def test_jointly_supported_partial_facts_do_not_become_a_complete_comparison():
    draft, evidence, audit = candidate({'method': '模型甲', 'task': '二值变化检测'})
    report = answer_then_audit('逐篇整理', evidence, Model([draft, audit]), draft_contract='comparison-v2')
    rows = reviewed_records(report, 'a', 'A')
    assert len(rows) == 1
    assert set(rows[0]['cells']) == {'method', 'task'}
    assert set(rows[0]['missing_fields']) == set(FIELDS) - {'method', 'task'}
    comparison = assemble_comparison(rows, [{'paper_sha256': 'a', 'title': 'A'}])
    assert not comparison['condition_groups']
    assert comparison['coverage'][0]['gaps']


def test_auditor_separates_populated_bindings_from_coverage_in_its_interface():
    draft, evidence, audit = candidate({'method': '模型甲', 'task': '二值变化检测'})
    class Capture(Model):
        def __init__(self):
            super().__init__([draft, audit])
            self.prompts = []
        def complete(self, messages, **kwargs):
            self.prompts.append(deepcopy(messages))
            return super().complete(messages, **kwargs)
    model = Capture()
    report = answer_then_audit('比较所有实验条件', evidence, model, draft_contract='comparison-v2')
    payload = json.loads(model.prompts[1][-1]['content'])
    assert set(payload['comparison_records'][0]) == {'record_id', 'fields'}
    assert set(payload['comparison_coverage'][0]['missing_fields']) == set(FIELDS) - {'method', 'task'}
    assert 'Do NOT transfer coverage failures' in model.prompts[1][0]['content']
    assert report['draft']['comparison_records'][0]['missing_fields']
