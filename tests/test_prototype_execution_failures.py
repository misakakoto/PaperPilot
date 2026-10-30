from types import SimpleNamespace

import pytest

from test_prototype_workspace import waiting


@pytest.mark.parametrize(('exit_code', 'expected'), [(125, 'ENVIRONMENT'), (127, 'ENVIRONMENT'),
                                                    (137, 'EXECUTION_INTERRUPTED')])
def test_container_failure_without_worker_receipt_stops_code_repair(tmp_path, monkeypatch, exit_code, expected):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    result = SimpleNamespace(status='FAILED', exit_code=exit_code, timed_out=False, metrics={}, backend='docker',
                             to_dict=lambda: {'exit_code': exit_code, 'stderr_path': 'retained-fixture.log'})
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', lambda *a: result)
    assert harness._training(rid, state) is False
    assert state['phase'] == 'PARTIAL'
    assert state['last_observation']['failure_kind'] == expected
    assert state['validation_feedback']['result']['execution']['stderr_path'] == 'retained-fixture.log'
    assert state['work']['tasks']['code']['status'] == 'DONE'
