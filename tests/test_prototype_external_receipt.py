"""Host receipt recovery fixtures; no candidate code or Docker is executed."""
import json
import os
from pathlib import Path

import pytest

from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.sandbox import ExperimentResult, _hash_tree
from test_prototype_workspace import waiting


def write_receipt(protocol):
    folder = Path(protocol.output_dir)/('experiment_'+protocol.fingerprint.split(':')[1][:24])
    folder.mkdir(parents=True, exist_ok=True)
    prereg = folder/'preregistration.json'
    prereg.write_text(json.dumps({'experiment_id': folder.name, 'protocol_hash': protocol.fingerprint,
                                 'protocol': protocol.to_dict()}), encoding='utf-8')
    for name in ('stdout.log', 'stderr.log'):
        (folder/name).write_text('', encoding='utf-8')
    source_hash = _hash_tree(Path(protocol.source_dir),
                            ignore={'.git', '.paperpilot_agentic', '__pycache__', '.pytest_cache'})
    result = ExperimentResult(experiment_id=folder.name, status='COMPLETED', backend='docker',
        exit_code=0, timed_out=False, started_at='2026-10-24T00:00:00+00:00',
        completed_at='2026-10-24T00:00:01+00:00', duration_seconds=1,
        protocol_hash=protocol.fingerprint, preregistration_path=str(prereg.resolve()),
        result_path=str((folder/'result.json').resolve()), stdout_path=str((folder/'stdout.log').resolve()),
        stderr_path=str((folder/'stderr.log').resolve()), source_hash_before=source_hash,
        source_hash_after=source_hash, source_unchanged=True, network_isolated=True,
        metrics={'passed': False, 'failure_kind': 'IMPLEMENTATION', 'error': 'fixture shape mismatch'})
    metrics_file = folder/'metrics'/'metrics.json'
    metrics_file.parent.mkdir()
    metrics_file.write_text(json.dumps(result.metrics), encoding='utf-8')
    result.artifacts.append({'kind': 'experiment_metrics', 'path': str(metrics_file.resolve()),
                             'sha256': _hash_tree(metrics_file.parent),
                             'metadata': {'source_name': 'metrics.json'}})
    (folder/'result.json').write_text(json.dumps(result.to_dict()), encoding='utf-8')
    return result


@pytest.mark.parametrize('real_container', [False, pytest.param(True, marks=pytest.mark.skipif(
    os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1', reason='Explicit container opt-in required'))])
def test_external_completed_receipt_recovers_without_second_launch(tmp_path, monkeypatch, real_container):
    from paperpilot_agentic.sandbox import SandboxExperimentRunner
    original_runner = SandboxExperimentRunner.run
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    launches = []

    def execute(_runner, protocol):
        launches.append(protocol)
        return original_runner(_runner, protocol) if real_container else write_receipt(protocol)

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', execute)
    original = runtime.db.update_tool_call

    def die(call_id, status, **kwargs):
        if str(getattr(status, 'value', status)) == 'COMPLETED':
            raise KeyboardInterrupt('Death after host result file, before database receipt')
        return original(call_id, status, **kwargs)

    monkeypatch.setattr(runtime.db, 'update_tool_call', die)
    with pytest.raises(KeyboardInterrupt):
        harness._training(rid, state)
    before = runtime.db.get_run(rid)
    assert runtime.db.get_tool_call(state['pending']['call'])['status'] == 'RUNNING'
    monkeypatch.setattr(runtime.db, 'update_tool_call', original)
    recovered = harness._load(rid)
    assert PrototypeHarness(runtime)._training(rid, recovered) is True
    assert len(launches) == 1
    assert recovered['last_observation']['failure_kind'] == 'IMPLEMENTATION'
    assert recovered['last_observation']['receipt_recovery']['new_executions'] == 0
    assert recovered['last_observation']['execution']['metrics']['passed'] is False
    assert runtime.db.get_tool_call(state['pending']['call'])['attempt'] == 1
    after = runtime.db.get_run(rid)
    assert after['used_llm_calls'] == before['used_llm_calls']
    assert after['used_steps'] == before['used_steps']


@pytest.mark.parametrize('damage', ['none', 'no_intent', 'protocol', 'source', 'result', 'missing',
                                   'metrics', 'metric_hash', 'pre_registration', 'path', 'backend',
                                   'timeout', 'network', 'failed'])
def test_receipt_recovery_requires_matching_complete_host_evidence(tmp_path, damage):
    from paperpilot_agentic.models import ExperimentProtocol, SandboxPolicy
    from paperpilot_agentic.prototype_receipts import execution_intent, recover_receipt
    source = tmp_path/'source'
    source.mkdir()
    (source/'model.py').write_text('maintained fixture', encoding='utf-8')
    protocol = ExperimentProtocol(name='fixture', command=['python', 'verify.py'],
        source_dir=str(source), output_dir=str(tmp_path/'output'), metric_files=['metrics.json'],
        policy=SandboxPolicy(backend='docker', network='none', allow_local_fallback=False))
    intent = execution_intent(protocol, 'owned')
    result = write_receipt(protocol)
    if damage == 'none':
        assert recover_receipt(protocol, 'owned', intent).metrics['passed'] is False
        return
    if damage == 'no_intent':
        intent = None
    elif damage == 'protocol':
        intent['protocol_hash'] = 'changed'
    elif damage == 'source':
        (source/'model.py').write_text('changed', encoding='utf-8')
    elif damage == 'result':
        Path(result.result_path).write_text('{', encoding='utf-8')
    elif damage == 'missing':
        Path(result.stderr_path).unlink()
    elif damage == 'metrics':
        (Path(result.result_path).parent/'metrics/metrics.json').write_text('{}', encoding='utf-8')
    elif damage == 'pre_registration':
        Path(result.preregistration_path).write_text('{}', encoding='utf-8')
    else:
        data = result.to_dict()
        if damage == 'metric_hash':
            data['artifacts'][0]['sha256'] = 'changed'
        elif damage == 'path':
            data['stdout_path'] = str(tmp_path/'foreign.log')
        else:
            key, value = {'backend': ('backend', 'local'), 'timeout': ('timed_out', True),
                          'network': ('network_isolated', False), 'failed': ('status', 'FAILED')}[damage]
            data[key] = value
        Path(result.result_path).write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((ValueError, OSError, TypeError)):
        recover_receipt(protocol, 'owned', intent)
