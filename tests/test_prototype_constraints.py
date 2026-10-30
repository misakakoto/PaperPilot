import pytest

from paperpilot_agentic.prototype_worker import make_validation_contract, evaluate_constraints


def test_contract_normalizes_defaults_and_explicit_input_cases():
    assert make_validation_contract('snunet_ecam')['input_cases'] == [[1,32,32],[2,32,48],[1,48,64]]
    value = make_validation_contract('unbound', {'input_cases': [[2,65,97]], 'max_parameters': 100})
    assert value == {'kind': 'unbound', 'input_cases': [[2,65,97]], 'max_parameters': 100}


@pytest.mark.parametrize('constraints', [
    {'input_cases': []}, {'input_cases': [[True,64,64]]}, {'input_cases': [[1,1024,1024]]},
    {'input_cases': [[1,64,64],[1,64,64]]}, {'max_parameters': True}, {'max_parameters': 0},
    {'max_peak_rss_kib': 0}, {'max_verifier_seconds': float('nan')}, {'max_verifier_seconds': 181},
    {'command': 'arbitrary'}, {'kind': 'other'}, {'max_parameters':10**400},
])
def test_invalid_or_unsafe_constraints_fail_before_generation(constraints):
    with pytest.raises(ValueError):
        make_validation_contract('unbound', constraints)


def test_native_snunet_rejects_unsupported_odd_dimensions():
    with pytest.raises(ValueError, match='16'):
        make_validation_contract('snunet_ecam', {'input_cases': [[1,65,97]]})


def test_limits_use_measured_values_and_cannot_pass_missing_measurement():
    contract = make_validation_contract('unbound', {'max_parameters': 7, 'max_peak_rss_kib': 100,
                                                   'max_verifier_seconds': 10})
    assert evaluate_constraints(contract, {'parameters': 7, 'peak_rss_kib': 100, 'seconds': 10}) == []
    failures = evaluate_constraints(contract, {'parameters': 8, 'peak_rss_kib': None, 'seconds': 11})
    assert {f['metric'] for f in failures} == {'parameters','peak_rss_kib','seconds'}
    assert next(f for f in failures if f['metric']=='peak_rss_kib')['kind'] == 'MEASUREMENT_UNAVAILABLE'


def test_task_and_approval_bind_exact_constraints(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    runtime, harness, _ = waiting(tmp_path, monkeypatch)
    constraints = {'input_cases': [[1,48,80]], 'max_parameters': 10}
    rid = harness.start('build', 'constraints-fixture', 'parent', ['source'], constraints=constraints)['run']['run_id']
    harness.advance(rid)
    state = harness._load(rid)
    contract = state['validation_contract']
    assert contract['input_cases'] == [[1,48,80]] and contract['max_parameters'] == 10
    assert state['pending']['binding']['validation_contract'] == contract
    with pytest.raises(ValueError, match='identity'):
        harness.start('build', 'constraints-fixture', 'parent', ['source'], constraints={'max_parameters': 11})
    state['validation_contract'] = {**contract, 'max_parameters': 999}
    with pytest.raises(ValueError, match='contract'):
        harness._guard(rid, state)


@pytest.mark.parametrize(('metrics','kind'), [({'passed': True, 'parameters': 7},'DESIGN_CONSTRAINT'),
                                           ({'passed': True},'ENVIRONMENT')])
def test_host_rechecks_limits_even_if_worker_declares_pass(tmp_path, monkeypatch, metrics, kind):
    from types import SimpleNamespace
    from test_prototype_workspace import waiting
    runtime, harness, _ = waiting(tmp_path, monkeypatch)
    rid = harness.start('build', 'host-limit-fixture', 'parent', ['source'],
                        constraints={'max_parameters':6})['run']['run_id']
    harness.advance(rid)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    result = SimpleNamespace(status='COMPLETED', exit_code=0, timed_out=False, metrics=metrics,
                             to_dict=lambda: {'metrics':metrics})
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', lambda *a:result)
    harness._training(rid, state)
    assert state['validation_feedback']['result']['passed'] is False
    assert state['validation_feedback']['result']['failure_kind'] == kind
    assert state['work']['repairs'] == 1
