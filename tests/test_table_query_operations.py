import copy
import pytest
from paperpilot_agentic.table_projection import project_table_facts


def catalog():
    return [{'evidence_id': 't', 'row_id': f'r{i}', 'artifact_sha256': 'artifact',
        'paper_sha256': 'paper', 'table': 'Table IX', 'row_label': model,
        'checked_columns': [], 'component_columns': [], 'metric_headers': ['F1 (%)'],
        'cells': {'Encoder': model, 'dataset': 'Synthetic', 'F1 (%)': f1,
                  'GFLOPs': flops, 'Param (M)': params}}
        for i, (model, f1, flops, params) in enumerate([
            ('Model-A', '71.20', '93', '30'), ('Model-B', '70.30', '73', '20'),
            ('Model-C', '69.10', '101', '25')])]


def selection(op):
    return {'facts': [{'id': 'f', 'kind': 'TABLE_FACT', 'operation': op,
        'row_refs': [{'evidence_id': 't', 'row_id': r['row_id'], 'metrics': ['F1 (%)']}
                     for r in catalog()]}]}


def test_descriptive_difference_allows_configuration_and_resource_measurement_changes():
    draft = selection('difference')
    draft['facts'][0]['row_refs'] = draft['facts'][0]['row_refs'][:2]
    result, _ = project_table_facts(draft, catalog())
    assert '-0.90 percentage points' in result['facts'][-1]['text']
    changed = catalog()
    changed[1]['cells']['batch_size'] = '8'
    changed[0]['cells']['batch_size'] = '4'
    with pytest.raises(ValueError, match='Other source conditions'):
        project_table_facts(draft, changed)


def test_filter_ranks_complete_table_and_can_have_one_eligible_row():
    draft = selection('filter_max')
    draft['facts'][0]['where'] = {'metric': 'GFLOPs', 'operator': '<=', 'value': '80'}
    result, trace = project_table_facts(draft, catalog())
    assert 'Model-B' in result['facts'][0]['text'] and '70.30' in result['facts'][0]['text']
    assert trace[0]['eligible_row_ids'] == ['r1']
    assert trace[0]['filter']['value'] == '80'
    bad = copy.deepcopy(draft)
    bad['facts'][0]['row_refs'].pop()
    with pytest.raises(ValueError, match='complete source table'):
        project_table_facts(bad, catalog())
    bad = copy.deepcopy(draft)
    bad['facts'][0]['where']['metric'] = 'invented'
    with pytest.raises(ValueError):
        project_table_facts(bad, catalog())


def test_explicit_method_rows_do_not_require_ablation_checkboxes():
    rows = catalog()[:2]
    for row in rows:
        row['checked_columns'] = None
        row['row_label'] = ''
        row['cells']['Methods'] = row['cells'].pop('Encoder')
    draft = selection('difference')
    draft['facts'][0]['row_refs'] = draft['facts'][0]['row_refs'][:2]
    result, _ = project_table_facts(draft, rows)
    assert '-0.90' in result['facts'][-1]['text']
    rows[1]['cells']['dataset'] = 'Different'
    with pytest.raises(ValueError):
        project_table_facts(draft, rows)
