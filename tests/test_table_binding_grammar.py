"""Table condition/value regressions from QA failures, with unrelated values."""
import json

import pytest

from paperpilot_agentic.numeric_scope import apply_scope_guard, check_bindings


def evidence():
    table = {'column_headers': ['Alpha', 'BETA', 'OA(%)', 'F1overall(%)'], 'rows': [
        {'row_label': 'baseline', 'checked_columns': [],
         'cells': {'OA(%)': '90.1', 'F1overall(%)': '70.2'}},
        {'row_label': 'alpha', 'checked_columns': ['Alpha'],
         'cells': {'OA(%)': '91.3', 'F1overall(%)': '71.4'}},
        {'row_label': 'beta', 'checked_columns': ['BETA'],
         'cells': {'OA(%)': '92.5', 'F1overall(%)': '72.6'}},
        {'row_label': 'both', 'checked_columns': ['Alpha', 'BETA'],
         'cells': {'OA(%)': '93.7', 'F1overall(%)': '73.8'}}]}
    return [{'evidence_id': 't', 'content':
             'Table IV\nRESULTS ON THE RED TEST SET.\nNamed-cell transcription:\n'
             + json.dumps(table)}]


def fact(text):
    return {'id': 'f', 'kind': 'SOURCE_FACT', 'text': text, 'evidence_ids': ['t']}


@pytest.mark.parametrize('label', ['F1overall (%)', 'F1overall（%）', 'F1overall'])
def test_percent_unit_and_copula_free_baseline_values_have_cell_receipts(label):
    text = f'Table IV 在 Red 上仅启用 BETA模块，{label}为72.6%，基线（OA 90.1%，F1overall 70.2%）。'
    checks = check_bindings(fact(text), evidence())['checks']
    cells = [(c['metric'], c['value'], c['status']) for c in checks if c['kind'] == 'table_cell']
    assert cells == [('f1overall', '72.6', 'MATCH'), ('oa', '90.1', 'MATCH'),
                     ('f1overall', '70.2', 'MATCH')]


@pytest.mark.parametrize('component,metric,value', [
    ('Alpha', 'OA', '91.3'), ('Alpha', 'F1overall', '71.4'),
    ('BETA', 'OA', '92.5'), ('BETA', 'F1overall', '72.6')])
def test_component_followed_by_chinese_and_titlecase_names_keep_row_identity(component, metric, value):
    checks = check_bindings(fact(f'Table IV 启用{component}模块，{metric}为{value}%。'), evidence())['checks']
    rows = [c for c in checks if c['kind'] == 'table_row']
    assert len(rows) == 1
    assert rows[0]['expected_checked_columns'] == [component.casefold()]
    assert rows[0]['status'] == 'MATCH'


@pytest.mark.parametrize('oa,f1,expected', [
    ('91.3', '71.4', 'MATCH'), ('92.5', '72.6', 'MISMATCH'),
    ('71.4', '91.3', 'UNRESOLVED')])
def test_multiple_metrics_and_baseline_are_bound_to_their_own_rows(oa, f1, expected):
    text = (f'Table IV 在 Red 上仅启用 Alpha模块，OA为{oa}%，F1overall为{f1}%，'
            '基线（OA 90.1%，F1overall 70.2%）。')
    checks = check_bindings(fact(text), evidence())['checks']
    rows = [c for c in checks if c['kind'] == 'table_row']
    assert len(rows) == 4
    assert all(c['status'] == expected for c in rows if c['expected_checked_columns'])
    assert all(c['status'] == 'MATCH' for c in rows if not c['expected_checked_columns'])


@pytest.mark.parametrize('text,expected', [
    ('Table IV 在 Red 上仅启用 BETA模块，F1overall (%)为72.6%，基线F1overall 70.2%。', 'SUPPORTED'),
    ('Table IV 在 Blue 上仅启用 BETA模块，F1overall (%)为72.6%。', 'UNSUPPORTED'),
    ('Table IV 在 Red 上同时启用 Alpha 和 BETA模块，F1overall (%)为72.6%。', 'UNSUPPORTED'),
    ('Table IV 在 Red 上仅启用 BETA模块，F1overall (%)为92.5%。', 'UNSUPPORTED'),
    ('Table IV 在 Red 上仅启用 BETA模块，F1overall (%)为72.6%，表明改善了特征对齐。', 'UNCERTAIN'),
])
def test_scope_guard_keeps_valid_facts_but_rejects_wrong_dataset_row_metric_and_causal_addition(text, expected):
    draft = {'facts': [fact(text)], 'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = apply_scope_guard(draft, audit, evidence())
    assert effective['assessments'][0]['verdict'] == expected
    assert audit['assessments'][0]['verdict'] == 'SUPPORTED'


def test_untyped_percentages_are_not_assigned_to_every_metric():
    text = 'Table IV 启用 Alpha模块，OA和F1overall分别为91.3%和71.4%。'
    checks = check_bindings(fact(text), evidence())['checks']
    # Unsupported grammar is a coverage gap, not fabricated row receipts.
    assert not [c for c in checks if c['kind'] == 'table_row']
