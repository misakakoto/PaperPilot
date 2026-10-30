from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from paperpilot_agentic.builtin_tools import _experiment_plan_handler
from paperpilot_agentic.models import ExperimentProtocol, stable_hash


def setup_plan(tmp_path, seeds):
    source = tmp_path/'source'
    source.mkdir(exist_ok=True)
    return SimpleNamespace(workspace=tmp_path, run_id='r1'), {
        'source_dir': str(source), 'command': ['python', 'study.py'],
        'spec': {'hypothesis': 'A controlled change improves the metric.',
                 'baseline': 'fixed baseline', 'variants': ['one changed factor'],
                 'dataset': 'public fixture', 'data_split': {'train': 'train', 'test': 'heldout'},
                 'controlled_variables': ['training steps'], 'metrics': ['score'], 'seeds': seeds,
                 'acceptance_criteria': [{'metric': 'score', 'operator': '>=', 'value': .8}]}}


def test_full_design_is_preregistered_hashed_and_roundtripped(tmp_path):
    context, args = setup_plan(tmp_path, [7])
    before = deepcopy(args)
    result = _experiment_plan_handler(context, args)
    assert result.output['ready_for_execution']
    protocol = ExperimentProtocol.from_dict(result.output['protocol'])
    assert protocol.study_design['data_split'] == args['spec']['data_split']
    assert protocol.study_design['baseline'] == 'fixed baseline'
    saved = json.loads(open(result.output['preregistration_path'], encoding='utf-8').read())
    assert saved == protocol.to_dict()
    old_hash = protocol.fingerprint
    protocol.study_design['baseline'] = 'different baseline'
    assert protocol.fingerprint != old_hash
    assert args == before


def test_multiseed_plan_is_not_reduced_to_one_execution(tmp_path):
    context, args = setup_plan(tmp_path, [7, 11, 19])
    result = _experiment_plan_handler(context, args)
    assert result.output['ready_for_execution'] is False
    assert result.output['spec']['seeds'] == [7, 11, 19]
    assert 'protocol' not in result.output


@pytest.mark.parametrize('seeds', [[True], ['7'], [7, 7], None])
def test_invalid_seed_spec_fails_without_silent_coercion(tmp_path, seeds):
    context, args = setup_plan(tmp_path, seeds)
    with pytest.raises(ValueError, match='Seeds'):
        _experiment_plan_handler(context, args)


def test_legacy_empty_design_preserves_fingerprint():
    protocol = ExperimentProtocol('legacy', ['python', 'study.py'], 'source', 'output')
    legacy = protocol.to_dict()
    assert 'study_design' not in legacy
    assert ExperimentProtocol.from_dict(legacy).fingerprint == stable_hash(legacy)
