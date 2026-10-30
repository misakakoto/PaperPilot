"""Synthetic counterexamples; scope matches do not prove semantic accuracy."""
from copy import deepcopy

import pytest

from paperpilot_agentic.numeric_scope import apply_scope_guard, assertions, check_bindings


def inputs(dataset='Red', value='87.25', source=None):
    text = source or 'TABLE IV\nRESULTS ON THE RED TEST SET.\nA baseline reaches 80%. '
    if source is None:
        text += f'The proposed method improves these results to {value} and 65%, respectively.'
        # Include the percent sign inside the caller-supplied numerical assertion.
        text = text.replace(f'to {value} and', f'to {value}% and')
    draft = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                       'text': f'Method 在 {dataset} 上报告的准确率为 {value}%', 'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                             'span_ids': ['s0'], 'reason': 'Original model assessment'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    return draft, audit, [{'evidence_id': 'e1', 'content': text}]


@pytest.mark.parametrize('dataset,expected', [('Red', 'SUPPORTED'), ('Blue', 'UNSUPPORTED')])
def test_caption_scopes_cross_sentence_percentage(dataset, expected):
    draft, audit, evidence = inputs(dataset)
    effective, report = apply_scope_guard(draft, audit, evidence)
    assert effective['assessments'][0]['verdict'] == expected
    assert report['facts'][0]['checks'][0]['occurrences'][0]['table'] == 'IV'
    assert audit['assessments'][0]['verdict'] == 'SUPPORTED'


@pytest.mark.parametrize('name,value', [('Alpha-2', '73.2'), ('Beta_2025', '91.03'),
                                     ('FieldSet', '14'), ('red', '87.250')])
def test_no_frozen_names_or_values_needed(name, value):
    source = f'TABLE 7\nRESULTS ON THE {name} TEST SET.\nThe method scores {value}%.'
    draft, audit, rows = inputs(name, value, source)
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'SUPPORTED'
    draft['facts'][0]['text'] = draft['facts'][0]['text'].replace(name, 'WrongSet')
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNSUPPORTED'


def test_pdf_wrapped_test_caption_keeps_scope():
    draft, audit, rows = inputs('Blue', source='TABLE IV\nRESULTS ON THE RED T EST SET.\nAccuracy is 87.25%.')
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNSUPPORTED'


@pytest.mark.parametrize('source', [
    'The model reaches 87.25%.',
    'On Red and Blue datasets, the model reaches 87.25%.',
    'TABLE IV\nRESULTS ON THE RED TEST SET.\nOn Blue dataset another experiment runs. Accuracy is 87.25%.',
    'TABLE IV\nRESULTS ON THE RED TEST SET.\nAccuracy is 87.25%.\nTABLE V\nRESULTS ON THE BLUE TEST SET.\nAccuracy is 87.25%.',
    'TABLE IV\nRESULTS ON THE RED TEST SET.\nAccuracy is 70%.',
])
def test_ambiguous_or_absent_binding_is_uncertain_not_approved(source):
    draft, audit, rows = inputs(source=source)
    effective, _ = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert effective['coverage'] == 'PARTIAL'


def test_other_paper_and_uncited_evidence_cannot_supply_binding():
    draft, audit, rows = inputs(source='A model reaches 87.25%.')
    rows.append({'evidence_id': 'not_cited', 'content': 'On Red dataset accuracy is 87.25%.'})
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_no_scope_inheritance_between_passages():
    draft, audit, rows = inputs(source='TABLE IV\nRESULTS ON THE RED TEST SET.')
    draft['facts'][0]['evidence_ids'].append('e2')
    rows.append({'evidence_id': 'e2', 'content': 'The proposed method reaches 87.25%.'})
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_new_table_without_caption_does_not_inherit_previous_scope():
    draft, audit, rows = inputs(source='TABLE IV\nRESULTS ON THE RED TEST SET.\nAccuracy is 70%.\nTABLE V\nAccuracy is 87.25%.')
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_same_sentence_scope_can_stand_without_table():
    draft, audit, rows = inputs(source='On Red dataset, the overall accuracy is 87.25%.')
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'SUPPORTED'


@pytest.mark.parametrize('verdict', ['UNSUPPORTED', 'UNCERTAIN'])
def test_numeric_scope_cannot_override_a_model_rejection(verdict):
    draft, audit, rows = inputs()
    audit['assessments'][0]['verdict'] = verdict
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == verdict


@pytest.mark.parametrize('mode', ['off', 'shadow'])
def test_ablations_do_not_modify_model_verdict(mode):
    draft, audit, rows = inputs('Blue')
    before = deepcopy((draft, audit, rows))
    effective, _ = apply_scope_guard(draft, audit, rows, mode=mode)
    assert effective == audit
    assert (draft, audit, rows) == before


def test_unrecognized_grammar_exposes_coverage_gap():
    draft, audit, rows = inputs()
    draft['facts'][0]['text'] = 'Accuracy was 87.25%.'
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective == audit
    assert report['facts'][0]['unparsed_percentage_count'] == 1
    assert report['facts'][0]['checks'] == []


def test_semicolon_does_not_carry_dataset_to_another_claim():
    assert assertions('On Red dataset the method improves; a different method scores 90%.') == []


def test_english_explicit_dataset_assertion():
    values = assertions('On Alpha dataset, accuracy reaches 91.25%.')
    assert [(r['dataset'], r['value']) for r in values] == [('Alpha', '91.25')]


def binding_inputs(fact_text, source, kind='SOURCE_FACT'):
    draft = {'facts': [{'id': 'f1', 'kind': kind, 'text': fact_text,
                        'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                              'relevance': 'DIRECT', 'span_ids': ['s0'],
                              'reason': 'Original model assessment'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    return draft, audit, [{'evidence_id': 'e1', 'content': source}]


def test_explicit_table_metric_and_method_bindings_match():
    draft, audit, rows = binding_inputs(
        'FlowMamba 在 Red 数据集上 Table IV 的 F1=91.0%',
        'TABLE IV\nRESULTS ON THE RED TEST SET.\nFlowMamba reports F1=91.0%.')
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    assert all(item['status'] == 'MATCH' for item in report['facts'][0]['binding_checks'])


def test_conflicting_table_metric_and_method_are_withheld():
    draft, audit, rows = binding_inputs(
        'FakeMamba 在 Red 数据集上 Table V 的 F1=91.0%',
        'TABLE IV\nRESULTS ON THE RED TEST SET.\nFlowMamba reports mIoU=91.0%.')
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'UNSUPPORTED'
    statuses = {(item['kind'], item['status']) for item in report['facts'][0]['binding_checks']}
    assert ('table', 'MISMATCH') in statuses
    assert ('metric', 'MISMATCH') in statuses
    assert ('method', 'UNRESOLVED') in statuses


def test_explicit_causal_claim_without_causal_source_is_uncertain():
    draft, audit, rows = binding_inputs(
        'FlowMamba 的 MFA 模块导致性能提升并贡献了鲁棒性。',
        'FlowMamba uses an MFA module and reports an accuracy score.')
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'UNCERTAIN'
    causal = [item for item in report['facts'][0]['binding_checks'] if item['kind'] == 'causal']
    assert causal and all(item['status'] == 'UNRESOLVED' for item in causal)


def test_audited_problem_motivation_is_not_an_experimental_causal_effect():
    draft, audit, rows = binding_inputs(
        'DeltaNet 通过估计光流对齐双时相特征，解决因空间错位导致的配准问题。',
        'DeltaNet estimates optical flow and warps bitemporal features to address spatial misalignment.')
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    causal = [item for item in report['facts'][0]['binding_checks'] if item['kind'] == 'causal']
    assert causal and not any(item['blocking'] for item in causal)
    assert all(item['status'] == 'UNRESOLVED' for item in causal)  # Still not a lexical proof.
    audit['assessments'][0]['verdict'] = 'UNSUPPORTED'
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNSUPPORTED'


def test_causal_cue_in_source_can_support_necessary_binding():
    draft, audit, rows = binding_inputs(
        'FlowMamba 的 MFA 模块得益于特征对齐并提升鲁棒性，体现出重要贡献。',
        'FlowMamba benefits from the MFA module and improves robustness; '
        'the contribution demonstrates its effectiveness.')
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    assert all(item['status'] == 'MATCH' for item in report['facts'][0]['binding_checks'])


def test_check_bindings_reports_missing_method_context_without_false_rejection():
    draft, audit, rows = binding_inputs(
        'FlowMamba 在 Red 数据集上报告准确率为 91.0%',
        'TABLE IV\nRESULTS ON THE RED TEST SET.\nThe method reaches 91.0%.')
    result = check_bindings(draft['facts'][0], rows)
    method = [item for item in result['checks'] if item['kind'] == 'method']
    assert method and method[0]['status'] == 'UNRESOLVED' and not method[0]['blocking']
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'SUPPORTED'


def test_quoted_document_title_is_not_a_table_method_but_wrong_unquoted_method_is_blocked():
    source = 'TABLE IV\nF1=62.3%. AlphaNet is the row method.'
    draft, audit, rows = binding_inputs('《StudyNet》的 Table IV 报告 F1=62.3%。', source)
    result, diagnostic = apply_scope_guard(draft, audit, rows)
    assert result['assessments'][0]['verdict'] == 'SUPPORTED'
    assert any(c['kind'] == 'paper_title' and not c['blocking']
               for c in diagnostic['facts'][0]['binding_checks'])
    draft, audit, rows = binding_inputs('WrongNet 的 Table IV 报告 F1=62.3%。', source)
    assert apply_scope_guard(draft, audit, rows)[0]['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_structured_table_cell_binding_checks_metric_column():
    source = ('Table IV\nRESULTS ON THE RED TEST SET.\n'
              'Named-cell transcription (each row is independent):\n'
              '{"column_headers":["Method","F1","IoU"],"rows":['
              '{"row_label":"FlowMamba","cells":{"Method":"FlowMamba",'
              '"F1":"91.0","IoU":"80.0"}}]}')
    draft, audit, rows = binding_inputs(
        'FlowMamba 在 Red 数据集上 Table IV 的 F1=91.0%', source)
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    cells = [item for item in report['facts'][0]['binding_checks'] if item['kind'] == 'table_cell']
    assert cells and cells[0]['status'] == 'MATCH'


def test_structured_table_cell_binding_withholds_value_in_wrong_metric_column():
    source = ('Table IV\nRESULTS ON THE RED TEST SET.\n'
              'Named-cell transcription (each row is independent):\n'
              '{"column_headers":["Method","F1","IoU"],"rows":['
              '{"row_label":"FlowMamba","cells":{"Method":"FlowMamba",'
              '"F1":"80.0","IoU":"91.0"}}]}')
    draft, audit, rows = binding_inputs(
        'FlowMamba 在 Red 数据集上 Table IV 的 F1=91.0%', source)
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'UNSUPPORTED'
    cells = [item for item in report['facts'][0]['binding_checks'] if item['kind'] == 'table_cell']
    assert cells and cells[0]['status'] == 'MISMATCH'


def test_structured_table_row_binding_checks_baseline_configuration():
    prefix = ('Table IV\nRESULTS ON THE RED TEST SET.\n'
              'Named-cell transcription (each row is independent):\n')
    source = prefix + ('{"column_headers":["MFA","F1"],"rows":['
              '{"row_label":"w/o","checked_columns":[],"cells":{"F1":"91.0"}},'
              '{"row_label":"mfa","checked_columns":["MFA"],"cells":{"F1":"80.0"}}]}')
    draft, audit, rows = binding_inputs(
        'FlowMamba 在 Red 数据集上 Table IV 的 baseline F1=91.0%', source)
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED'
    row_checks = [item for item in report['facts'][0]['binding_checks'] if item['kind'] == 'table_row']
    assert row_checks and row_checks[0]['status'] == 'MATCH'


def test_structured_table_row_binding_withholds_value_from_wrong_configuration():
    prefix = ('Table IV\nRESULTS ON THE RED TEST SET.\n'
              'Named-cell transcription (each row is independent):\n')
    source = prefix + ('{"column_headers":["MFA","F1"],"rows":['
              '{"row_label":"w/o","checked_columns":[],"cells":{"F1":"80.0"}},'
              '{"row_label":"mfa","checked_columns":["MFA"],"cells":{"F1":"91.0"}}]}')
    draft, audit, rows = binding_inputs(
        'FlowMamba 在 Red 数据集上 Table IV 的 baseline F1=91.0%', source)
    effective, report = apply_scope_guard(draft, audit, rows)
    assert effective['assessments'][0]['verdict'] == 'UNSUPPORTED'
    row_checks = [item for item in report['facts'][0]['binding_checks'] if item['kind'] == 'table_row']
    assert row_checks and row_checks[0]['status'] == 'MISMATCH'
