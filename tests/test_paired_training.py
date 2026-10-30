from pathlib import Path

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic import paired_training as training
from paperpilot_agentic.training_state import TrainingBusy


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    runtime = ResearchAgentRuntime(db_path=tmp_path/'runtime.sqlite3', workspace=tmp_path/'work')
    parent = runtime.enqueue(question='Draft', metadata={'conversation_id': 'c'})['run']['run_id']
    card = {'binding_id': 'fixed', 'design_run_id': parent, 'protocol': {'version': 2}, 'requires_approval': True}
    monkeypatch.setattr(training, 'describe_training', lambda *_: card.copy())
    return runtime, card


def test_duplicate_and_refresh_require_new_v2_approval_without_execution(prepared, monkeypatch):
    runtime, card = prepared
    def unexpected(*_):
        pytest.fail('Unapproved worker executed')
    monkeypatch.setattr(training, 'execute_training', unexpected)
    rid = training.enqueue_training(runtime, 'fixed')['run']['run_id']
    assert training.enqueue_training(runtime, 'fixed')['run']['run_id'] == rid
    assert runtime.resume(rid)['run']['status'] == 'WAITING_APPROVAL'
    assert len(runtime.db.list_tool_calls(rid)) == len(runtime.db.list_approvals(rid)) == 1
    assert training.campaign(runtime).snapshot()['launches'] == []


def test_completed_response_reused_without_new_gpu_work(prepared, monkeypatch):
    runtime, _ = prepared
    state = training.enqueue_training(runtime, 'fixed')
    rid = state['run']['run_id']
    runtime.approve(state['pending_approvals'][0]['approval_id'])
    trials = {f'{seed}-{variant}': {'status': 'VERIFIED', 'initial_state_sha256': str(seed),
        'metrics': dict.fromkeys(('precision', 'recall', 'f1', 'iou'), .5)}
        for seed in (7, 11, 19) for variant in ('baseline', 'candidate')}
    calls = []
    def execute(*_):
        calls.append(True)
        folder = runtime.workspace/'method-training'/'runs'/rid
        folder.mkdir(parents=True)
        return {'trials': trials, 'checks': {'fixture': 'No actual training measured'}}
    monkeypatch.setattr(training, 'execute_training', execute)
    assert runtime.resume(rid)['run']['status'] == 'COMPLETED'
    assert runtime.resume(rid)['run']['status'] == 'COMPLETED'
    assert len(calls) == 1
    assert len(runtime.db.list_artifacts(rid)) == 1
    assert runtime.db.get_run(rid)['used_llm_calls'] == 0


def test_unknown_worker_blocks_without_marking_success(prepared, monkeypatch):
    runtime, _ = prepared
    state = training.enqueue_training(runtime, 'fixed')
    rid = state['run']['run_id']
    runtime.approve(state['pending_approvals'][0]['approval_id'])
    training.campaign(runtime).reserve('unknown-launch', 100)
    result = runtime.resume(rid)
    assert result['run']['status'] != 'COMPLETED'
    assert any(e['event_type'] == 'training.resume_blocked' for e in runtime.db.list_events(run_id=rid))
    assert training.campaign(runtime).snapshot()['reserved_seconds'] == 100


def test_changed_approval_rejected_before_launch(prepared):
    runtime, card = prepared
    state = training.enqueue_training(runtime, 'fixed')
    runtime.approve(state['pending_approvals'][0]['approval_id'])
    card['protocol'] = {'version': 1}
    with pytest.raises(ValueError, match='approval'):
        runtime.resume(state['run']['run_id'])


def test_pause_then_resume_retains_step_and_cumulative_budget(prepared, monkeypatch):
    runtime, _ = prepared
    state = training.enqueue_training(runtime, 'fixed')
    rid = state['run']['run_id']
    runtime.approve(state['pending_approvals'][0]['approval_id'])
    step = runtime.db.get_run(rid)['current_step_id']
    ledger = training.campaign(runtime)
    token, _ = ledger.reserve('preflight-fixture', 100)
    ledger.finish(token, 'PAUSED', 12)
    folder = runtime.workspace/'method-training'/'runs'/rid
    folder.mkdir(parents=True)
    calls = []
    def execute(*_):
        calls.append(True)
        if len(calls) == 1:
            return {'trials': {'7-baseline': {'status': 'PAUSED', 'step': 25}}, 'checks': {}}
        return {'trials': {f'{seed}-{variant}': {'status': 'VERIFIED', 'initial_state_sha256': str(seed),
            'metrics': dict.fromkeys(('precision', 'recall', 'f1', 'iou'), .5)}
            for seed in (7, 11, 19) for variant in ('baseline', 'candidate')}, 'checks': {}}
    monkeypatch.setattr(training, 'execute_training', execute)
    assert runtime.resume(rid)['run']['status'] == 'PENDING'
    assert runtime.db.get_run(rid)['current_step_id'] == step
    assert runtime.resume(rid)['run']['status'] == 'COMPLETED'
    assert ledger.snapshot()['charged_seconds'] == 12


def test_saved_report_without_matching_launch_is_rejected(tmp_path):
    from paperpilot_agentic.training_state import GpuBudget, atomic_json
    ledger = GpuBudget(tmp_path/'budget.sqlite3')
    atomic_json(tmp_path/'receipt-fake.json', {'status': 'COMPLETED', 'token': 'not-reserved'})
    with pytest.raises(ValueError, match='matching'):
        training._saved_report(Path(tmp_path), 'frozen', 'formal', 7, 'baseline', ledger)


def test_two_simultaneous_gpu_launches_cannot_be_scheduled(tmp_path):
    from paperpilot_agentic.training_state import GpuBudget
    ledger = GpuBudget(tmp_path/'budget.sqlite3')
    ledger.reserve('first', 100)
    with pytest.raises(TrainingBusy):
        ledger.reserve('second', 100)


def test_worker_failure_keeps_all_seeds_in_partial_receipt(prepared, monkeypatch):
    runtime, _ = prepared
    state = training.enqueue_training(runtime, 'fixed')
    rid = state['run']['run_id']
    runtime.approve(state['pending_approvals'][0]['approval_id'])
    def fail(*_):
        raise ValueError('Injected worker failure')
    monkeypatch.setattr(training, 'execute_training', fail)
    result = runtime.resume(rid)
    assert result['run']['status'] == 'FAILED'
    summary = result['run']['metadata']['method_comparison']
    assert summary['status'] == 'PARTIAL'
    assert [r['seed'] for r in summary['trials']] == [7, 11, 19]
    assert all(r['baseline']['status'] == r['candidate']['status'] == 'NOT_RUN' for r in summary['trials'])
    assert 'Injected worker failure' in summary['error']
    assert len(runtime.db.list_artifacts(rid)) == 1


def test_old_smoke_handler_cannot_expand_to_long_training(monkeypatch, tmp_path):
    from paperpilot_agentic import paired_methods
    monkeypatch.setattr(paired_methods, 'load_comparison', lambda *_: (tmp_path, {'version': 2}))
    with pytest.raises(ValueError, match='dedicated approval'):
        paired_methods.run_comparison(None, {'binding_id': 'new-v2-binding'})


def test_completed_receipt_does_not_mean_process_has_exited(tmp_path):
    from paperpilot_agentic.training_state import GpuBudget
    ledger = GpuBudget(tmp_path/'budget.sqlite3', 80)
    token, _ = ledger.reserve('formal', 30)
    with ledger.connection() as db:
        db.execute("UPDATE launches SET pid=123,identity='owned' WHERE token=?", (token,))
    ledger.finish(token, 'COMPLETED', 12)
    with pytest.raises(TrainingBusy, match='confirmed exited'):
        training.confirm_exits(ledger, lambda _: ('RUNNING', 'owned'))
    with pytest.raises(TrainingBusy):
        training.confirm_exits(ledger, lambda _: ('UNKNOWN', None))
    training.confirm_exits(ledger, lambda _: ('DEAD', None))
    # Later reuse by an unrelated, inaccessible process must not block forever.
    training.confirm_exits(ledger, lambda _: ('UNKNOWN', None))
    assert ledger.snapshot()['charged_seconds'] == 12


def test_real_cpu_process_exit_window_in_isolated_database(tmp_path):
    """CPU-only child deliberately lingers after settlement; no experiment GPU used."""
    import json
    import subprocess
    import sys
    from paperpilot_agentic import training_state
    code = '''import importlib.util,json,sys,time
spec=importlib.util.spec_from_file_location('fixture',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
b=m.GpuBudget(sys.argv[2],80)
token,_=b.reserve('CPU_PROCESS_LIFETIME_ONLY',30)
b.attach(token);b.finish(token,'COMPLETED',.01)
print(json.dumps({'token':token}),flush=True)
time.sleep(6)
'''
    process = subprocess.Popen([sys.executable, '-I', '-c', code, training_state.__file__, str(tmp_path/'cpu.sqlite3')],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        token = json.loads(process.stdout.readline())['token']
        assert process.poll() is None
        ledger = training_state.GpuBudget(tmp_path/'cpu.sqlite3', 80)
        assert ledger.snapshot()['launches'][0]['status'] == 'COMPLETED'
        with pytest.raises(TrainingBusy, match='confirmed exited'):
            training.confirm_exits(ledger)
        process.wait(timeout=15)
        training.confirm_exits(ledger)
        with ledger.connection() as db:
            assert db.execute('SELECT token FROM process_exits').fetchone()[0] == token
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_cancel_before_dispatch_does_not_reserve_or_launch(prepared, tmp_path):
    runtime, _ = prepared
    rid = training.enqueue_training(runtime, 'fixed')['run']['run_id']
    runtime.cancel(rid)
    ledger = training.campaign(runtime)
    with pytest.raises(TrainingBusy, match='Cancellation'):
        training.run_job(runtime, rid, tmp_path, {}, tmp_path/'worker', ledger, stage='formal')
    assert runtime.db.get_run(rid)['status'] == 'CANCELLED'
    assert ledger.snapshot()['launches'] == []
