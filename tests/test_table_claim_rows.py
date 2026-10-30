import copy

import pytest

from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion
from paperpilot_agentic.table_claims import row_catalog, validate_row_refs, apply_row_guard


def table():
    artifact = ParsedArtifact(kind='table', source=SourceRegion(
        paper_sha256='p', object_id='o', label='Table IV', page=2,
        bbox=[0, 0, 10, 10], crop_sha256='crop'), parser='fixture', parser_version='2',
        config_sha256='cfg', parse_status='PARSED', validation_status='VALIDATED',
        column_headers=['Alpha', 'Beta', 'offset', 'F1'], table_rows=[
            {'checked_columns': [], 'cells': {'offset': 'w/o', 'F1': '61.2'}},
            {'checked_columns': ['Alpha'], 'cells': {'offset': 'w/o', 'F1': '62.3'}},
            {'checked_columns': ['Beta'], 'cells': {'offset': 'w/o', 'F1': '63.4'}},
            {'checked_columns': ['Alpha', 'Beta'], 'cells': {'offset': 'w/o', 'F1': '64.5'}},
            {'checked_columns': ['Alpha', 'Beta'], 'cells': {'offset': '5px', 'F1': '64.1'}},
        ])
    return {'evidence_id': 't', 'source_uri': 'paper://p/layout/o',
            'content': 'Table IV reports F1 on DeltaSet.',
            'provenance': {'structured_extraction': artifact.model_dump()}}


def ref(row):
    return {k: row[k] for k in ('evidence_id', 'row_id')} | {'metrics': ['F1']}


def test_row_identity_binds_crop_parser_config_and_exact_source():
    source = table()
    rows = row_catalog([source])
    assert len(rows) == 5 and len({r['row_id'] for r in rows}) == 5
    changed = copy.deepcopy(source)
    changed['provenance']['structured_extraction']['parser_version'] = '3'
    assert row_catalog([changed])[0]['row_id'] != rows[0]['row_id']
    changed['source_uri'] = 'paper://neighbor/layout/o'
    assert row_catalog([changed]) == []


@pytest.mark.parametrize('mutation', ['row', 'metric', 'citation', 'two_rows'])
def test_model_cannot_invent_rows_or_mix_configs_in_one_source_fact(mutation):
    source = table()
    rows = row_catalog([source])
    fact = {'id': 'f', 'kind': 'SOURCE_FACT', 'text': 'Alpha F1 is 62.3.',
            'evidence_ids': ['t'], 'row_refs': [ref(rows[1])]}
    if mutation == 'row':
        fact['row_refs'][0]['row_id'] = 'invented'
    if mutation == 'metric':
        fact['row_refs'][0]['metrics'] = ['IoU']
    if mutation == 'citation':
        fact['evidence_ids'] = ['another']
    if mutation == 'two_rows':
        fact['row_refs'].append(ref(rows[0]))
    with pytest.raises(ValueError):
        validate_row_refs({'facts': [fact]}, rows)


@pytest.mark.parametrize('indices,text,expected', [
    ([3, 0], '移除 Alpha 时 F1 从 64.5% 降到 61.2%。', 'MISMATCH'),
    ([3, 2], '移除 Alpha 时 F1 从 64.5% 降到 63.4%。', 'MATCH'),
    ([1, 0], 'Removing Alpha changes F1 from 62.3% to 61.2%.', 'MATCH'),
    ([3, 4], '移除 Alpha 时 F1 从 64.5% 降到 64.1%。', 'MISMATCH'),
    ([3, 0], '移除Alpha模块时 F1 从 64.5% 降到 61.2%。', 'MISMATCH'),
])
def test_single_factor_requires_other_components_and_conditions_to_stay_fixed(indices, text, expected):
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'kind': 'INFERENCE', 'text': text,
                      'evidence_ids': ['t'], 'row_refs': [ref(rows[i]) for i in indices]}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, diagnostic = apply_row_guard(draft, audit, rows)
    assert diagnostic['facts'][0]['status'] == expected
    assert actual['assessments'][0]['verdict'] == ('UNSUPPORTED' if expected == 'MISMATCH' else 'SUPPORTED')


def test_legacy_no_refs_cannot_hide_a_two_component_removal():
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'text': '移除 Alpha 时 F1 从 64.5% 降到 61.2%。',
                       'evidence_ids': ['t']}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, diagnostic = apply_row_guard(draft, audit, rows)
    assert actual['assessments'][0]['verdict'] == 'UNSUPPORTED'
    assert diagnostic['facts'][0]['changed_components'] == ['alpha', 'beta']


def test_unknown_endpoints_remain_unknown_and_shadow_never_changes_model_verdict():
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'text': '移除 Alpha 改善性能。', 'evidence_ids': ['t']}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, diagnostic = apply_row_guard(draft, audit, rows)
    assert actual['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert diagnostic['facts'][0]['status'] == 'UNRESOLVED'
    assert apply_row_guard(draft, audit, rows, mode='shadow')[0] == audit


@pytest.mark.parametrize('value,verdict', [('62.3', 'SUPPORTED'), ('61.2', 'UNSUPPORTED')])
def test_enabled_single_configuration_is_not_a_two_endpoint_comparison(value, verdict):
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT',
                       'text': f'仅启用 Alpha 时 F1 为 {value}%。',
                       'evidence_ids': ['t'], 'row_refs': [ref(rows[1])]}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, _ = apply_row_guard(draft, audit, rows)
    assert actual['assessments'][0]['verdict'] == verdict


@pytest.mark.parametrize('text', ['仅启用 Alpha 时 F1 为 63.4%。',
                                 '仅启用 Alpha 时 F1 为 64.5%。'])
def test_single_configuration_cannot_borrow_another_configuration_value(text):
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': text, 'evidence_ids': ['t']}]}
    actual, _ = apply_row_guard(draft, {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
                                      'coverage': 'COMPLETE'}, rows)
    assert actual['assessments'][0]['verdict'] == 'UNSUPPORTED'


def test_one_configuration_does_not_establish_a_causal_improvement():
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT',
                       'text': '启用 Alpha 使 F1 提升至 62.3%。',
                       'evidence_ids': ['t'], 'row_refs': [ref(rows[1])]}]}
    actual, _ = apply_row_guard(draft, {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
                                      'coverage': 'COMPLETE'}, rows)
    assert actual['assessments'][0]['verdict'] == 'UNCERTAIN'


@pytest.mark.parametrize('text', [
    '完整组合（同时启用 Alpha 和 Beta）在 F1 上优于基线及各单一机制配置。',
    'The full configuration (simultaneously enabling Alpha and Beta) is better than the baseline and either single configuration on F1.',
])
@pytest.mark.parametrize('verdict', ['SUPPORTED', 'UNSUPPORTED'])
def test_descriptive_group_configuration_does_not_claim_a_single_factor_effect(text, verdict):
    rows = row_catalog([table()])[:4]
    premises = [{'id': f'p{i}', 'kind': 'SOURCE_FACT', 'text': f'F1={row["cells"]["F1"]}',
                 'evidence_ids': ['t'], 'row_refs': [ref(row)]} for i, row in enumerate(rows)]
    draft = {'facts': premises + [{'id': 'i', 'kind': 'INFERENCE', 'text': text,
                                  'evidence_ids': ['t'], 'depends_on': [f['id'] for f in premises]}]}
    audit = {'assessments': [{'fact_id': f['id'], 'verdict': verdict if f['id'] == 'i' else 'SUPPORTED'}
                             for f in draft['facts']], 'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, report = apply_row_guard(draft, audit, rows)
    assert actual == audit  # No host approval: keep the independent verdict.
    assert 'i' not in report['changed_fact_ids']


@pytest.mark.parametrize('text', [
    '同时启用 Alpha 和 Beta 使 F1 提升，全部改善由 Alpha 导致。',
    '同时启用 Alpha 时 F1 从 61.2% 提升到 64.5%。',
    'Simultaneously enabling Alpha and Beta causes an improvement in F1 due only to Alpha.',
    '仅启用 Alpha 时 F1 从 61.2% 提升到 64.5%。',
])
def test_group_words_do_not_bypass_causal_or_single_module_checks(text):
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'i', 'kind': 'INFERENCE', 'text': text,
                       'evidence_ids': ['t'], 'row_refs': [ref(rows[0]), ref(rows[3])]}]}
    actual, report = apply_row_guard(draft, {'assessments': [{'fact_id': 'i', 'verdict': 'SUPPORTED'}],
                                          'coverage': 'COMPLETE'}, rows)
    assert actual['assessments'][0]['verdict'] == 'UNSUPPORTED'
    assert report['facts'][0]['changed_components'] == ['alpha', 'beta']


@pytest.mark.parametrize('name', ['ChangeNet', 'LeadingModel', '变化检测模型'])
def test_method_name_is_not_a_comparative_verb(name):
    rows = row_catalog([table()])
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT',
                       'text': f'{name} 仅启用 Alpha 时 F1 为 62.3%。',
                       'evidence_ids': ['t'], 'row_refs': [ref(rows[1])]}]}
    actual, _ = apply_row_guard(draft, {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED'}],
                                      'coverage': 'COMPLETE'}, rows)
    assert actual['assessments'][0]['verdict'] == 'SUPPORTED'


@pytest.mark.parametrize('change', ['numeric_condition', 'dependency_metric'])
def test_numeric_conditions_and_dependency_metrics_cannot_be_silently_mixed(change):
    source = table()
    for i, row in enumerate(source['provenance']['structured_extraction']['table_rows']):
        row['cells'].update({'lr': '0.002' if change == 'numeric_condition' and i == 1 else '0.001',
                             'IoU': '50.0'})
    rows = row_catalog([source])
    source_refs = [ref(rows[1]), ref(rows[0])]
    if change == 'dependency_metric':
        source_refs[1]['metrics'] = ['IoU']
    draft = {'facts': [{'id': 'a', 'text': 'Alpha row', 'evidence_ids': ['t'], 'row_refs': [source_refs[0]]},
                       {'id': 'b', 'text': 'Baseline row', 'evidence_ids': ['t'], 'row_refs': [source_refs[1]]},
                       {'id': 'i', 'kind': 'INFERENCE', 'text': '移除 Alpha 时 F1 从 62.3% 降到 61.2%。',
                        'evidence_ids': ['t'], 'depends_on': ['a', 'b']}]}
    audit = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED'} for f in draft['facts']],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = apply_row_guard(draft, audit, rows)
    assert effective['assessments'][2]['verdict'] == 'UNSUPPORTED'
