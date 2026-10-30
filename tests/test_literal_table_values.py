"""Reading a missing-value token is distinct from computing with that token."""
from copy import deepcopy

import pytest

from paperpilot_agentic.table_projection import project_table_facts
from paperpilot_agentic.table_claims import validate_row_refs, apply_row_guard
from test_table_fact_projection import draft, rows


@pytest.mark.parametrize('token', ['†', '—', 'N/A', 'not observed'])
def test_value_preserves_literal_without_numeric_coercion_or_support_verdict(token):
    catalog = rows()
    catalog[1]['cells']['F1 (%)'] = token
    original = deepcopy(catalog)
    result, trace = project_table_facts(draft(), catalog, chinese=True)
    assert catalog == original
    assert 'F1 (%)=' + token in result['facts'][0]['text']
    assert '非数值' in result['facts'][0]['text']
    assert 'F1 (%)=0' not in result['facts'][0]['text']
    assert trace[0]['cell_value_types'][0]['kind'] == 'NON_NUMERIC_LITERAL'
    assert trace[0]['cell_value_types'][0]['source_text'] == token
    assert 'verdict' not in result['facts'][0]
    validate_row_refs(result, catalog)
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    checked, _ = apply_row_guard(result, audit, catalog)
    assert checked['assessments'][0]['verdict'] == 'SUPPORTED'


def test_real_numeric_zero_remains_a_numeric_value():
    catalog = rows()
    catalog[1]['cells']['F1 (%)'] = '0.00'
    result, trace = project_table_facts(draft(), catalog, chinese=True)
    assert 'F1 (%)=0.00' in result['facts'][0]['text']
    assert trace[0]['cell_value_types'][0]['kind'] == 'NUMERIC'


@pytest.mark.parametrize('operation', ['difference', 'max', 'min', 'filter_max', 'filter_min'])
def test_calculations_still_reject_non_numeric_cells(operation):
    catalog = rows()
    catalog[1]['cells']['F1 (%)'] = '†'
    selection = draft()
    fact = selection['facts'][0]
    fact['operation'] = operation
    fact['row_refs'] = [{'evidence_id': 't', 'row_id': r['row_id'], 'metrics': ['F1 (%)']}
                        for r in catalog[:2]]
    if operation.startswith('filter_'):
        fact['where'] = {'metric': 'F1 (%)', 'operator': '>=', 'value': '0'}
    with pytest.raises(ValueError, match='numeric'):
        project_table_facts(selection, catalog, chinese=True)


@pytest.mark.parametrize('value', ['', ' ', None])
def test_empty_or_missing_cell_is_not_invented_as_a_literal(value):
    catalog = rows()
    catalog[1]['cells']['F1 (%)'] = value
    with pytest.raises(ValueError):
        project_table_facts(draft(), catalog, chinese=True)


def test_wrong_literal_binding_rejected_and_prior_audit_rejection_never_upgraded():
    catalog = rows()
    catalog[1]['cells']['F1 (%)'] = '†'
    result, _ = project_table_facts(draft(), catalog, chinese=True)
    rejected = {'assessments': [{'fact_id': 'f1', 'verdict': 'UNSUPPORTED', 'relevance': 'DIRECT'}],
                'coverage': 'PARTIAL', 'missing_aspects': []}
    checked, _ = apply_row_guard(result, rejected, catalog)
    assert checked['assessments'][0]['verdict'] == 'UNSUPPORTED'
    wrong = deepcopy(result)
    wrong['facts'][0]['text'] = 'An unrelated mark † in a different column.'
    with pytest.raises(ValueError):
        validate_row_refs(wrong, catalog)
