import copy
import pytest
from paperpilot_agentic.table_projection import project_table_facts


def rows():
    return [{'evidence_id': 't', 'row_id': 'r' + str(i), 'table': 'Table IV',
             'paper_sha256': 'p', 'artifact_sha256': 'a', 'row_label': label,
             'checked_columns': modules, 'component_columns': ['Alpha', 'Beta'],
             'cells': {'dataset': 'Delta', 'F1 (%)': value}}
            for i, (label, modules, value) in enumerate([
                ('', [], '61.20'), ('', ['Alpha'], '62.30'), ('', ['Beta'], '63.40')])]


def draft():
    return {'facts': [{'id': 'f1', 'kind': 'TABLE_FACT', 'operation': 'value',
        'row_refs': [{'evidence_id': 't', 'row_id': 'r1', 'metrics': ['F1 (%)']}]}],
        'direct_answer_id': 'f1', 'limitations': []}


def test_project_values_configuration_and_unit_not_model_prose():
    result, trace = project_table_facts(draft(), rows(), chinese=True)
    fact = result['facts'][0]
    assert fact['kind'] == 'SOURCE_FACT'
    assert '62.30' in fact['text'] and 'Alpha' in fact['text'] and 'Delta' in fact['text']
    assert fact['evidence_ids'] == ['t'] and trace[0]['raw_selection']['kind'] == 'TABLE_FACT'


@pytest.mark.parametrize('mutation', ['forged_row', 'wrong_metric', 'extra_number', 'cross_paper'])
def test_bad_or_unowned_selection_rejected(mutation):
    value = draft()
    f = value['facts'][0]
    if mutation == 'forged_row':
        f['row_refs'][0]['row_id'] = 'fake'
    if mutation == 'wrong_metric':
        f['row_refs'][0]['metrics'] = ['IoU']
    if mutation == 'extra_number':
        f['text'] = 'F1 999.9'
    if mutation == 'cross_paper':
        f['row_refs'][0]['evidence_id'] = 'other'
    with pytest.raises(ValueError):
        project_table_facts(value, rows(), chinese=True)


def test_unbound_numbers_cannot_bypass_by_omitting_metric_name():
    value = draft()
    value['facts'][0] = {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Table IV baseline 78.5%, full 81.8%.',
                         'evidence_ids': ['t'], 'depends_on': []}
    with pytest.raises(ValueError):
        project_table_facts(value, rows(), chinese=True)


def test_body_number_without_table_catalog_keeps_normal_audit_path():
    value = draft()
    value['facts'][0] = {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Training uses 20 epochs.',
                         'evidence_ids': ['body'], 'depends_on': []}
    result, trace = project_table_facts(value, rows(), chinese=False)
    assert result == value and trace == []


def test_difference_exact_and_other_conditions_reject_cross_dataset():
    value = draft()
    value['facts'][0]['operation'] = 'difference'
    value['facts'][0]['row_refs'].insert(0, {'evidence_id':'t','row_id':'r0','metrics':['F1 (%)']})
    projected, _ = project_table_facts(value, rows(), chinese=True)
    assert '1.10 percentage points' in projected['facts'][-1]['text']
    changed = rows()
    changed[1]['cells']['dataset'] = 'Other'
    with pytest.raises(ValueError):
        project_table_facts(value, changed, chinese=True)


def test_local_extrema_scope_and_no_global_or_causal_claim():
    value = draft()
    value['facts'][0]['operation'] = 'max'
    value['facts'][0]['row_refs'] = [{'evidence_id':'t','row_id':r['row_id'],'metrics':['F1 (%)']} for r in rows()]
    projected, _ = project_table_facts(value, rows(), chinese=True)
    assert '63.40' in projected['facts'][0]['text'] and '所选表内配置' in projected['facts'][0]['text']
    assert '因果' not in projected['facts'][0]['text']


@pytest.mark.parametrize('operation', ['value', 'max', 'min', 'difference'])
def test_numeric_conditions_survive_projection_without_adding_unrequested_metrics(operation):
    catalog = rows()
    for row in catalog:
        row['metric_headers'] = ['F1 (%)', 'IoU (%)']
        row['cells'].update({'batch_size': '4', 'crop_size': '256', 'IoU (%)': '50.00'})
    selection = draft()
    selection['facts'][0]['operation'] = operation
    if operation != 'value':
        selection['facts'][0]['row_refs'] = [
            {'evidence_id': 't', 'row_id': row['row_id'], 'metrics': ['F1 (%)']}
            for row in catalog[:2]]
    before = copy.deepcopy(catalog)
    result, _ = project_table_facts(selection, catalog, chinese=True)
    for fact in result['facts']:
        assert 'batch_size=4' in fact['text']
        assert 'crop_size=256' in fact['text']
        assert 'IoU (%)=' not in fact['text']
    assert catalog == before


def test_numeric_condition_changes_remain_incomparable():
    catalog = rows()
    for row in catalog:
        row['cells']['batch_size'] = '4'
    catalog[1]['cells']['batch_size'] = '8'
    selection = draft()
    selection['facts'][0]['operation'] = 'difference'
    selection['facts'][0]['row_refs'].insert(0,
        {'evidence_id': 't', 'row_id': 'r0', 'metrics': ['F1 (%)']})
    with pytest.raises(ValueError, match='Other source conditions'):
        project_table_facts(selection, catalog, chinese=True)
