"""Persistent, approved v2 training coordinator behind the existing comparison API."""
import json
import os
from pathlib import Path
import subprocess
import time

from .models import Budget, ResearchOutcome, RunStatus, SandboxPolicy, StepStatus, ToolCallStatus
from .sandbox import _sanitized_environment, _terminate_process
from .training_metrics import paired_summary, verify_trial
from .training_profiles import describe_training, load_training
from .training_state import GpuBudget, TrainingBusy, atomic_json, file_sha, process_identity


def campaign(runtime):
    return GpuBudget(runtime.workspace/'method-training'/'gpu-budget.sqlite3')


def enqueue_training(runtime, binding):
    card = describe_training(runtime, binding)
    ledger = campaign(runtime)
    with ledger.connection() as db:
        row = db.execute('SELECT run_id FROM runs WHERE binding=?', (binding,)).fetchone()
    if row:
        return runtime.status(row['run_id'])
    parent = runtime.db.get_run(card['design_run_id'])
    queued = runtime.enqueue(question='执行冻结的三种子长训练成对对照，保留负结果并独立复算指标。',
        metadata={'engine': 'method_training', 'binding_id': binding, 'method_card': card,
                  'parent_run_id': parent['run_id'], 'conversation_id': parent['metadata']['conversation_id'],
                  'resolved_paper_sha256s': parent['metadata'].get('resolved_paper_sha256s', [])},
        budget=Budget(max_steps=1, max_tool_calls=1, max_llm_calls=0, max_seconds=28800,
                      synthesis_reserve_calls=0, repair_reserve_calls=0))
    run_id = queued['run']['run_id']
    owner = ledger.claim_run(binding, run_id)
    if owner != run_id:
        runtime.db.update_run_status(run_id, RunStatus.CANCELLED, error='Duplicate submission; original run reused')
        return runtime.status(owner)
    step_id = runtime.db.create_step(run_id, 'method.compare', {'binding_id': binding})
    call_id, _ = runtime.db.create_tool_call(run_id=run_id, step_id=step_id, tool_name='method.compare',
        args={'binding_id': binding}, risk_level='HIGH', idempotency_key='training:'+binding, max_attempts=1)
    runtime.db.create_approval(run_id, call_id, card)
    runtime.db.update_tool_call(call_id, ToolCallStatus.WAITING_APPROVAL)
    runtime.db.update_run_status(run_id, RunStatus.WAITING_APPROVAL, current_step_id=step_id)
    return runtime.status(run_id)


def _saved_report(output, binding, stage, seed, variant, ledger):
    snapshot = ledger.snapshot()
    tokens = {r['token']: r for r in snapshot['launches']}
    for path in sorted(output.glob('receipt-*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        report = json.loads(path.read_bytes())
        token = tokens.get(report.get('token'))
        if token is None or token['stage'] != str(output):
            raise ValueError('Worker report has no matching GPU launch')
        if report.get('status') == 'FAILED':
            raise ValueError('Previous worker failed; inspect and freeze another protocol: '+report.get('error', ''))
        if report.get('status') == 'COMPLETED':
            expected = {'binding_id': binding, 'protocol_sha256': report['identity']['protocol_sha256'],
                        'stage': stage, 'seed': seed, 'variant': variant}
            if report.get('binding_id') != binding or report.get('identity') != expected:
                raise ValueError('Saved worker report belongs to another trial')
            return report, path
    return None, None


def confirm_exits(ledger, resolver=process_identity):
    """A settled receipt can precede OS exit. Persist exit confirmation separately.

Once confirmed, later PID reuse or permission changes cannot undo the recorded
exit. This host-side table is compatible with the immutable worker's ledger.
"""
    with ledger.connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS process_exits(token TEXT PRIMARY KEY,confirmed_at REAL NOT NULL)')
        rows = [dict(r) for r in db.execute('SELECT l.* FROM launches l LEFT JOIN process_exits e ON l.token=e.token '
                                           'WHERE l.pid IS NOT NULL AND e.token IS NULL')]
    for row in rows:
        status, identity = resolver(row['pid'])
        if status == 'UNKNOWN' or status == 'RUNNING' and identity == row['identity']:
            raise TrainingBusy('Previous owned process has not been confirmed exited; do not start another')
        with ledger.connection() as db:
            db.execute('INSERT OR IGNORE INTO process_exits VALUES(?,?)', (row['token'], time.time()))


def run_job(runtime, run_id, root, manifest, output, ledger, *, stage, seed=7, variant='baseline', pause_at=None):
    """One owned subprocess, bounded by a durable reservation; no transparent retry."""
    if runtime.db.get_run(run_id)['metadata'].get('cancel_requested'):
        runtime.db.update_run_status(run_id, RunStatus.CANCELLED)
        raise TrainingBusy('Cancellation requested; no new GPU worker started')
    ledger.recover()
    confirm_exits(ledger)
    binding = runtime.db.get_run(run_id)['metadata']['binding_id']
    output.mkdir(parents=True, exist_ok=True)
    report, path = _saved_report(output, binding, stage, seed, variant, ledger)
    if report:
        return report, path
    if (output/'stop').exists():
        (output/'stop').unlink()
    maximum = 600 if stage == 'preflight' else 180 if stage == 'recovery' else 3600
    if ledger.snapshot()['remaining_seconds'] <= 30:
        raise TimeoutError('Too little cumulative GPU budget to safely checkpoint')
    token, seconds = ledger.reserve(str(output), maximum)
    job = {'profile': str(root), 'output': str(output), 'ledger': str(ledger.path), 'token': token,
        'binding_id': binding, 'stage': stage, 'seed': seed, 'variant': variant, 'pause_at': pause_at}
    job_path = output/('job-'+token+'.json')
    atomic_json(job_path, job)
    policy = SandboxPolicy(network='host', env_allowlist=('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'CUDA_VISIBLE_DEVICES'))
    started = time.monotonic()
    with (output/('stdout-'+token+'.txt')).open('wb') as stdout, (output/('stderr-'+token+'.txt')).open('wb') as stderr:
        try:
            process = subprocess.Popen([manifest['python'], '-I', '-B', '-u', str(root/'worker.py'), str(job_path)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=_sanitized_environment(policy, {}), start_new_session=os.name == 'posix',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError:
            ledger.finish(token, 'FAILED', time.monotonic()-started)
            raise
        while process.poll() is None:
            if runtime.db.get_run(run_id)['metadata'].get('cancel_requested') or time.monotonic()-started >= seconds-20:
                (output/'stop').touch()
            if time.monotonic()-started >= seconds:
                _terminate_process(process)
                process.wait(timeout=20)
                break
            time.sleep(.5)
    # Recovery settles a confirmed dead, unreceipted process conservatively.
    ledger.recover()
    confirm_exits(ledger)
    path = output/('receipt-'+token+'.json')
    if not path.exists():
        raise TrainingBusy('Worker exited without a committed receipt; checkpoint inspected before any resume')
    report = json.loads(path.read_bytes())
    if process.returncode != 0 or report.get('status') == 'FAILED':
        raise ValueError(report.get('error', 'Worker failed; inspect stderr'))
    return report, path


def _reference(path):
    return {'path': str(path), 'sha256': file_sha(path)}


def _verify_reference(reference, folder):
    path = Path(reference['path'])
    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()) or file_sha(path) != reference['sha256']:
        raise ValueError('Previously verified report changed')
    return json.loads(path.read_bytes())


def publish_summary(runtime, run_id, card, state, *, complete=False, error=None):
    summary = {'status': 'PAIRED_TRAINING_VERIFIED' if complete else 'PARTIAL', 'binding_id': card['binding_id'],
        'protocol': card['protocol'], 'checks': state.get('checks', {}), 'trials': paired_summary(state.get('trials', {})),
        'gpu_budget': campaign(runtime).snapshot(), 'scientific_hypothesis_supported': False,
        'scientific_acceptance': 'DESCRIPTIVE_ONLY', 'new_llm_calls': 0, 'error': error,
        'limitations': ['512 train/256 val, no augmentation; not author protocol or test benchmark',
                        'No statistical significance or innovation claim', 'Host-local reviewed execution']}
    folder = runtime.workspace/'method-training'/'runs'/run_id
    folder.mkdir(parents=True, exist_ok=True)
    # Live projection plus immutable snapshots; a later resume never rewrites an artifact.
    from .reference_methods import encoded, sha
    path = folder/('verification-'+sha(encoded(summary))+'.json')
    if not path.exists():
        atomic_json(path, summary)
    atomic_json(folder/'verification.json', summary)
    runtime.db.update_run_metadata(run_id, {'method_comparison': summary, 'training_state': summary['status']})
    artifacts = runtime.db.list_artifacts(run_id)
    if not any(a['path'] == str(path) for a in artifacts):
        step_id = runtime.db.list_tool_calls(run_id)[0]['step_id']
        runtime.db.add_artifact(run_id, step_id, {'kind': 'paired-training-verification', **_reference(path)})
    return summary


def execute_training(runtime, run_id, *, runner=run_job):
    run = runtime.db.get_run(run_id)
    root, manifest = load_training(runtime, run['metadata']['binding_id'])
    ledger = campaign(runtime)
    ledger.recover()
    folder = runtime.workspace/'method-training'/'runs'/run_id
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder/'state.json'
    state = json.loads(state_path.read_bytes()) if state_path.exists() else {'trials': {}, 'checks': {}}
    for check in state['checks'].values():
        for key in ('receipt', 'continuous', 'resumed'):
            if key in check:
                _verify_reference(check[key], folder)
    # Revalidate every stored raw report; changed results are not silently reused.
    for trial in state['trials'].values():
        if trial['status'] == 'VERIFIED':
            raw = _verify_reference(trial['receipt'], folder)
            trial['metrics'] = verify_trial(raw, root, manifest, seed=raw['identity']['seed'], variant=raw['identity']['variant'])
    if not state['checks'].get('preflight'):
        raw, path = runner(runtime, run_id, root, manifest, folder/'preflight', ledger, stage='preflight')
        if raw['status'] != 'COMPLETED':
            raise TimeoutError('Preflight paused without passing')
        mean_step_seconds = raw['launch_elapsed_seconds']/manifest['protocol']['preflight']['steps']
        estimate = mean_step_seconds*15360*1.25
        if (not raw.get('learning', {}).get('passed') or raw['peak_memory_bytes'] > 15*1024**3
                or estimate > ledger.snapshot()['remaining_seconds']):
            raise ValueError('Preflight learning/memory/time constraint failed; freeze another protocol before formal results')
        state['checks']['preflight'] = {'receipt': _reference(path), 'estimated_formal_seconds': estimate,
            'learning': raw['learning'], 'peak_memory_bytes': raw['peak_memory_bytes']}
        atomic_json(state_path, state)
    if not state['checks'].get('recovery'):
        full, p1 = runner(runtime, run_id, root, manifest, folder/'recovery-continuous', ledger, stage='recovery')
        split, p2 = runner(runtime, run_id, root, manifest, folder/'recovery-resumed', ledger, stage='recovery', pause_at=3)
        if split['status'] == 'PAUSED':
            split, p2 = runner(runtime, run_id, root, manifest, folder/'recovery-resumed', ledger, stage='recovery')
        keys = ('step', 'final_state_sha256', 'optimizer_sha256', 'rng_sha256', 'losses', 'gradient_norms')
        if full['status'] != 'COMPLETED' or split['status'] != 'COMPLETED' or any(full[k] != split[k] for k in keys):
            raise ValueError('Continuous versus checkpoint recovery differs; do not start formal training')
        state['checks']['recovery'] = {'continuous': _reference(p1), 'resumed': _reference(p2),
            'exactly_equal': list(keys)}
        atomic_json(state_path, state)
    for seed in (7, 11, 19):
        for variant in ('baseline', 'candidate'):
            name = f'{seed}-{variant}'
            if state['trials'].get(name, {}).get('status') == 'VERIFIED':
                continue
            if ledger.snapshot()['remaining_seconds'] <= 30:
                break
            try:
                raw, path = runner(runtime, run_id, root, manifest, folder/'trials'/name, ledger,
                                   stage='formal', seed=seed, variant=variant)
            except (TrainingBusy, TimeoutError):
                raise
            except Exception as exc:
                state['trials'][name] = {'status': 'FAILED', 'error': type(exc).__name__+': '+str(exc)}
                atomic_json(state_path, state)
                raise
            if raw['status'] == 'PAUSED':
                state['trials'][name] = {'status': 'PAUSED', 'step': raw['step'], 'receipt': _reference(path)}
                atomic_json(state_path, state)
                return state
            metrics = verify_trial(raw, root, manifest, seed=seed, variant=variant)
            state['trials'][name] = {'status': 'VERIFIED', 'metrics': metrics,
                'initial_state_sha256': raw['initial_state_sha256'], 'final_state_sha256': raw['final_state_sha256'],
                'elapsed_seconds': raw['elapsed_seconds'], 'peak_memory_bytes': raw['peak_memory_bytes'],
                'receipt': _reference(path)}
            paired_summary(state['trials'])
            atomic_json(state_path, state)
            runtime.db.emit_event(run_id, 'training.trial_verified', {'seed': seed, 'variant': variant, 'metrics': metrics})
    return state


def resume_training(runtime, run_id):
    run = runtime.db.get_run(run_id)
    if run['status'] in ('COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'):
        return runtime.status(run_id)
    binding = run['metadata']['binding_id']
    card = describe_training(runtime, binding)
    approvals = runtime.db.list_approvals(run_id)
    if len(approvals) != 1 or approvals[0]['request'] != card:
        raise ValueError('Long-training approval is not bound to the current protocol')
    approval = approvals[0]
    call = runtime.db.get_tool_call(approval['tool_call_id'])
    if call['tool_name'] != 'method.compare' or call['args'] != {'binding_id': binding}:
        raise ValueError('Training call differs from its frozen approval')
    if approval['status'] == 'PENDING':
        return runtime.status(run_id)
    if approval['status'] != 'APPROVED':
        runtime.db.update_run_status(run_id, RunStatus.CANCELLED, error='Training approval rejected')
        return runtime.status(run_id)
    call_id = approval['tool_call_id']
    step_id = call['step_id']
    try:
        campaign(runtime).recover()
        runtime.db.update_tool_call(call_id, ToolCallStatus.RUNNING)
        runtime.db.update_run_status(run_id, RunStatus.RUNNING, current_step_id=step_id)
        state = execute_training(runtime, run_id)
        complete = sum(t['status'] == 'VERIFIED' for t in state['trials'].values()) == 6
        summary = publish_summary(runtime, run_id, card, state, complete=complete)
        runtime.db.update_run_outcome(run_id, ResearchOutcome.PARTIAL)
        if runtime.db.get_run(run_id)['status'] == 'CANCELLED':
            return runtime.status(run_id)
        if complete:
            runtime.db.update_tool_call(call_id, ToolCallStatus.COMPLETED, result={'ok': True, 'output': summary})
            runtime.db.update_step(step_id, status=StepStatus.COMPLETED, output=summary)
            runtime.db.update_run_status(run_id, RunStatus.COMPLETED)
        elif campaign(runtime).snapshot()['remaining_seconds'] <= 30:
            runtime.db.update_run_status(run_id, RunStatus.BUDGET_EXHAUSTED, error='Cumulative GPU budget exhausted; checkpoint retained')
        else:
            runtime.db.update_run_metadata(run_id, {'training_state': 'PAUSED'})
            runtime.db.update_run_status(run_id, RunStatus.PENDING, current_step_id=step_id)
    except TrainingBusy as exc:
        runtime.db.emit_event(run_id, 'training.resume_blocked', {'reason': str(exc)})
    except TimeoutError as exc:
        _publish_failure(runtime, run_id, card, str(exc))
        runtime.db.update_run_outcome(run_id, ResearchOutcome.PARTIAL)
        runtime.db.update_run_status(run_id, RunStatus.BUDGET_EXHAUSTED, error=str(exc))
    except Exception as exc:
        _publish_failure(runtime, run_id, card, str(exc))
        runtime.db.update_tool_call(call_id, ToolCallStatus.FAILED, error=str(exc))
        runtime.db.update_run_outcome(run_id, ResearchOutcome.PARTIAL)
        runtime.db.update_run_status(run_id, RunStatus.FAILED, error=type(exc).__name__+': '+str(exc))
    return runtime.status(run_id)


def _publish_failure(runtime, run_id, card, error):
    path = runtime.workspace/'method-training'/'runs'/run_id/'state.json'
    state = json.loads(path.read_bytes()) if path.exists() else {'trials': {}, 'checks': {}}
    publish_summary(runtime, run_id, card, state, error=error)
