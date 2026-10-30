"""Cancellation at the adapter boundary; no generated candidate is executed."""
from types import SimpleNamespace

from test_prototype_workspace import waiting


def test_cancel_during_execution_retains_receipt_and_never_calls_model_again(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    call_id = state['pending']['call']
    code_id = state['pending']['binding']['code']
    runtime.approve(state['pending']['approval'])
    before = runtime.db.get_run(rid)
    launches = []

    def cancel_in_runner(_runner, protocol):
        launches.append(protocol.fingerprint)
        runtime.cancel(rid, reason='Fixture user cancellation during execution')
        assert _runner.cancel_requested()
        return SimpleNamespace(status='FAILED', backend='docker', exit_code=137, timed_out=False,
            metrics={}, to_dict=lambda: {'status': 'FAILED', 'exit_code': 137,
                                        'stderr_path': 'retained-cancellation.log'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', cancel_in_runner)
    result = harness.advance(rid)
    after = runtime.db.get_run(rid)
    assert result['run']['status'] == 'CANCELLED'
    assert harness._load(rid)['phase'] == 'CANCELLED'
    assert after['used_llm_calls'] == before['used_llm_calls']
    assert after['used_tokens'] == before['used_tokens']
    assert after['used_steps'] == before['used_steps']
    receipt = runtime.db.get_tool_call(call_id)
    assert receipt['status'] == 'COMPLETED' and receipt['attempt'] == 1
    assert receipt['result']['output']['passed'] is False
    assert receipt['result']['output']['execution']['stderr_path'] == 'retained-cancellation.log'
    assert harness._load(rid)['work']['tasks']['code']['output'] == code_id
    assert harness._load(rid)['work']['tasks']['check']['status'] != 'DONE'
    assert result['final_answer'] is None
    harness.advance(rid)
    assert len(launches) == 1
    assert runtime.db.get_run(rid)['used_llm_calls'] == before['used_llm_calls']
