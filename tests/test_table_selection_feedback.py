"""Missing metric selections must be actionable, never guessed by the host."""
from copy import deepcopy

import pytest

from paperpilot_agentic.table_projection import project_table_facts
from test_table_fact_projection import draft, rows


def test_missing_metrics_names_the_required_field_without_projecting_values():
    selection = draft()
    del selection['facts'][0]['row_refs'][0]['metrics']
    original = deepcopy(selection)
    with pytest.raises(ValueError, match=r'row_refs\[0\]\.metrics.*F1 \(%\)'):
        project_table_facts(selection, rows(), chinese=True)
    assert selection == original


def test_forged_row_does_not_gain_a_column_catalog():
    selection = draft()
    selection['facts'][0]['row_refs'][0] = {'evidence_id': 'other', 'row_id': 'r1'}
    with pytest.raises(ValueError, match='Unowned source row') as error:
        project_table_facts(selection, rows(), chinese=True)
    assert 'F1' not in str(error.value)
