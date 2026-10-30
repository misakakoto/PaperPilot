from dataclasses import replace
import json
from pathlib import Path

import pytest

from paperpilot_agentic.models import ExperimentProtocol, SandboxPolicy
from paperpilot_agentic.sandbox import ExperimentResult, _hash_tree
from paperpilot_agentic.seed_sweep import run_seed_sweep, validate_seed_sweep
from paperpilot_agentic.builtin_tools import _experiment_evaluate_handler, _experiment_plan_handler
from test_experiment_design_binding import setup_plan


@pytest.fixture
def protocol(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    return ExperimentProtocol(
        'replication', ['python', 'study.py', '--seed', '__PAPERPILOT_SEED__'],
        str(source), str(tmp_path / 'output'), seed_sweep=[7, 11, 19],
        acceptance_criteria=[{'metric': 'score', 'operator': '>=', 'value': .8}],
        policy=SandboxPolicy(timeout_seconds=30),
    )


def fake_result(child, **kwargs):
    source_hash = _hash_tree(Path(child.source_dir))
    result = ExperimentResult(
        'fake', 'COMPLETED', 'docker', 0, False, '', '', 0,
        child.fingerprint, '', '', '', '', metrics={'seed': child.seed, 'score': .9},
        source_hash_before=source_hash, source_hash_after=source_hash, network_isolated=True)
    return replace(result, **kwargs)


def test_each_seed_bound_preregistered_and_evaluated(protocol):
    calls = []
    def run(child):
        calls.append(child)
        assert child.command[-1] == str(child.seed)
        assert not child.seed_sweep
        return fake_result(child)
    result = run_seed_sweep(protocol, run)
    assert [p.seed for p in calls] == [7, 11, 19]
    assert len({p.fingerprint for p in calls}) == 3
    assert len({p.output_dir for p in calls}) == 3
    assert result.accepted is True
    assert [row['seed'] for row in result.criteria] == [7, 11, 19]
    evaluated = _experiment_evaluate_handler(None, {'protocol': protocol.to_dict(), 'result': result.to_dict()})
    assert evaluated.output['polarity'] == 'SUPPORTS'
    # Persisted terminal results, including failures, never trigger hidden reruns.
    assert run_seed_sweep(protocol, run).to_dict() == result.to_dict()
    assert len(calls) == 3


@pytest.mark.parametrize('failure', ['seed', 'missing', 'exception', 'source', 'protocol', 'network'])
def test_bad_seed_run_is_not_scientific_refutation(protocol, failure):
    calls = []
    def run(child):
        calls.append(child.seed)
        if child.seed != 11:
            return fake_result(child)
        if failure == 'exception':
            raise RuntimeError('container unavailable')
        overrides = {
            'seed': {'metrics': {'seed': 7, 'score': .9}},
            'missing': {'metrics': {'seed': 11}},
            'source': {'source_hash_before': 'wrong'},
            'protocol': {'protocol_hash': 'wrong'},
            'network': {'network_isolated': False},
        }
        return fake_result(child, **overrides[failure])
    result = run_seed_sweep(protocol, run)
    assert calls == [7, 11, 19]
    assert result.accepted is None
    assert len(result.seed_runs) == 3
    evaluated = _experiment_evaluate_handler(None, {'protocol': protocol.to_dict(), 'result': result.to_dict()})
    assert evaluated.output['polarity'] == 'NEUTRAL'


def test_valid_counterexample_does_not_disappear_in_an_average(protocol):
    result = run_seed_sweep(protocol, lambda p: fake_result(
        p, metrics={'seed': p.seed, 'score': .7 if p.seed == 11 else .99}))
    assert result.accepted is False
    evaluated = _experiment_evaluate_handler(None, {'protocol': protocol.to_dict(), 'result': result.to_dict()})
    assert evaluated.output['polarity'] == 'CONTRADICTS'


def test_interrupted_attempt_is_not_blindly_reexecuted(protocol):
    def interrupt(child):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_seed_sweep(protocol, interrupt)
    with pytest.raises(RuntimeError, match='automatic rerun forbidden'):
        run_seed_sweep(protocol, lambda p: pytest.fail('must not execute'))
    journal = next(Path(protocol.output_dir).rglob('seed_runs.json'))
    assert [r['status'] for r in json.loads(journal.read_text())['seed_runs']] == ['RUNNING', 'NOT_RUN', 'NOT_RUN']


@pytest.mark.parametrize('changes', [
    {'seed_sweep': [True]}, {'seed_sweep': [7, 7]},
    {'command': ['python', 'study.py', '--seed', '7']},
    {'policy': SandboxPolicy(timeout_seconds=600)},
    {'study_design': {'seeds': [7]}}, {'metric_files': ['a.json', 'b.json']},
])
def test_invalid_execution_contract_rejected_before_running(protocol, changes):
    with pytest.raises(ValueError):
        validate_seed_sweep(replace(protocol, **changes))


def test_plan_binds_full_seed_set_only_with_approved_placeholder(tmp_path):
    context, args = setup_plan(tmp_path, [7, 11, 19])
    args['command'] += ['--seed', '__PAPERPILOT_SEED__']
    result = _experiment_plan_handler(context, args)
    assert result.output['ready_for_execution'] is True
    protocol = ExperimentProtocol.from_dict(result.output['protocol'])
    assert protocol.seed_sweep == [7, 11, 19]
    assert protocol.seed is None
    assert protocol.policy.timeout_seconds == 160
    assert 'not_a_significance_test' in result.output['execution_scope']
