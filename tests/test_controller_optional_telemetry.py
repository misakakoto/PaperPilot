import pytest

from paperpilot_agentic.intelligence import EvidenceDrivenController
from paperpilot_agentic.llm_runtime import StructuredOutputSemanticError


@pytest.mark.parametrize('gain', [None, {'value': .8}, 'high because evidence is missing', 'NaN', True])
def test_bad_optional_gain_does_not_destroy_safe_novel_query(gain):
    decision = EvidenceDrivenController._validate_decision(
        {'action': 'REFORMULATE', 'queries': ['new targeted query'], 'expected_information_gain': gain},
        ['previous query'])
    assert decision['queries'] == ['new targeted query']
    assert decision['expected_information_gain'] == 0
    assert decision['telemetry_warning']


@pytest.mark.parametrize('value', [
    {'action': 'EXECUTE', 'queries': ['new targeted query']},
    {'action': 'SEARCH', 'queries': ['previous query']},
    {'action': 'SEARCH', 'queries': []},
])
def test_action_and_novelty_guards_still_reject_invalid_decisions(value):
    with pytest.raises(StructuredOutputSemanticError):
        EvidenceDrivenController._validate_decision(value, ['previous query'])
