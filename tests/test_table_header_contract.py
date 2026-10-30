import copy

import pytest

from paperpilot_agentic.table_header_contract import header_candidates, normalize_header_tiers


HEADERS = {'column_headers': ['Options', 'A', 'B', 'Score'],
           'indicator_columns': ['A', 'B'], 'group_headers': ['Options'], 'uncertain_fields': []}
ROWS = {'column_headers': ['Options', 'Score'], 'indicator_columns': ['A', 'B'],
        'table_rows': [{'cells': {'A': '', 'B': '✓', 'Score': '81.11'},
                        'checked_columns': ['B'], 'row_label': ''}]}


def test_retained_leaf_order_repairs_only_header_shape():
    candidates = header_candidates(HEADERS)
    assert candidates['column_headers'] == ['A', 'B', 'Score']
    original = copy.deepcopy(ROWS)
    result, changes = normalize_header_tiers(ROWS, candidates)
    assert result['column_headers'] == ['A', 'B', 'Score']
    assert result['table_rows'] == ROWS['table_rows']
    assert ROWS == original
    assert changes[0]['operation'] == 'header_tiers_to_retained_leaf_order'


@pytest.mark.parametrize('cells', [
    {'A': '', 'B': '✓'},
    {'A': '', 'B': '✓', 'Score': '81.11', 'unknown': ''},
    {'A': '', 'B': '✓', 'Score': '81.11', 'Options': 'populated'},
])
def test_missing_unknown_or_populated_parent_cells_are_not_repaired(cells):
    value = {**ROWS, 'table_rows': [{'cells': cells}]}
    assert normalize_header_tiers(value, header_candidates(HEADERS))[1] == []


def test_conflicting_header_order_and_indicator_roles_are_not_repaired():
    for override in ({'column_headers': ['Score', 'A']}, {'indicator_columns': ['Options']}):
        assert normalize_header_tiers({**ROWS, **override}, header_candidates(HEADERS))[1] == []


def test_uncertain_or_unknown_header_roles_do_not_allow_a_second_read():
    for override in ({'uncertain_fields': ['A is unreadable']}, {'indicator_columns': ['unknown']}):
        with pytest.raises(ValueError):
            header_candidates({**HEADERS, **override})


def test_separate_checkmark_schema_does_not_require_duplicate_indicator_cells():
    value = {**ROWS, 'table_rows': [{'cells': {'Score': '81.11'}, 'checked_columns': ['B']}]}
    result, changes = normalize_header_tiers(value, header_candidates(HEADERS))
    assert changes
    assert result['column_headers'] == ['A', 'B', 'Score']
    assert result['table_rows'] == value['table_rows']
