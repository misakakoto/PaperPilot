"""Execution failures are not scientific counter-evidence."""
import pytest

from paperpilot_agentic.sandbox import _evaluate_criteria
from paperpilot_agentic.builtin_tools import _experiment_evaluate_handler
from paperpilot_agentic.models import ExperimentProtocol


@pytest.mark.parametrize('metrics', [{}, {'score': None}, {'score': float('nan')}, {'score': float('inf')}])
def test_unusable_metric_is_inconclusive(metrics):
    outcomes, accepted = _evaluate_criteria(metrics, [{'metric': 'score', 'operator': '>=', 'value': .8}])
    assert accepted is None
    assert outcomes[0]['error']


@pytest.mark.parametrize('score,accepted', [(.9, True), (.7, False)])
def test_measured_result_can_support_or_fail_criterion(score, accepted):
    _, actual = _evaluate_criteria({'score': score}, [{'metric': 'score', 'operator': '>=', 'value': .8}])
    assert actual is accepted


@pytest.mark.parametrize('accepted,criteria,polarity', [
    (None, [], 'NEUTRAL'),
    (False, [{'error': 'KeyError: score', 'passed': False}], 'NEUTRAL'),
    (False, [{'error': None, 'passed': False}], 'CONTRADICTS'),
    (True, [{'error': None, 'passed': True}], 'SUPPORTS'),
])
def test_evaluation_preserves_unknown_instead_of_fabricating_refutation(accepted, criteria, polarity):
    protocol = ExperimentProtocol(name='test', command=['python', 'study.py'], source_dir='source', output_dir='out')
    result = {'protocol_hash': protocol.fingerprint, 'status': 'COMPLETED',
              'source_unchanged': True, 'network_isolated': True,
              'accepted': accepted, 'criteria': criteria}
    evaluated = _experiment_evaluate_handler(None, {'protocol': protocol.to_dict(), 'result': result})
    assert evaluated.output['polarity'] == polarity
    assert evaluated.evidence[0]['polarity'] == polarity
    if polarity == 'NEUTRAL':
        assert evaluated.output['accepted'] is None
