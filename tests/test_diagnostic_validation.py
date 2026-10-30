from copy import deepcopy

import pytest

from paperpilot_agentic.diagnostic_validation import require_observed_refs


def test_missing_computed_field_reports_observed_endpoints_without_rewriting_claim():
    catalog = {'paired/0/baseline/fp': 8, 'paired/0/candidate/fp': 3,
               'paired/0/delta/f1': -0.1}
    claim = ['paired/0/delta/fp']
    before = deepcopy((catalog, claim))
    with pytest.raises(ValueError) as caught:
        require_observed_refs(claim, catalog, label='next_experiment.support_refs')
    text = str(caught.value)
    assert 'paired/0/delta/fp' in text
    assert 'paired/0/baseline/fp' in text and 'paired/0/candidate/fp' in text
    assert 'Claims were not repaired' in text
    assert (catalog, claim) == before
    require_observed_refs(['paired/0/candidate/fp'], catalog)


def test_empty_references_identify_the_exact_field_and_unobserved_data_stays_unknown():
    with pytest.raises(ValueError, match='next_experiment.counter_refs requires'):
        require_observed_refs([], {}, label='next_experiment.counter_refs')
    with pytest.raises(ValueError, match='unknown observed fact IDs'):
        require_observed_refs(['unobserved/confidence'], {'paired/0/delta/f1': 0.1})
