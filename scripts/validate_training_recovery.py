"""Real process interruption and CPU corruption checks for the frozen campaign.

Run only while the approved formal worker is paused. Uses the same eight-hour ledger.
Preserves the original checkpoints; fault injection applies to owned copies only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime  # noqa: E402
from paperpilot_agentic.models import SandboxPolicy  # noqa: E402
from paperpilot_agentic.paired_training import campaign, run_job  # noqa: E402
from paperpilot_agentic.sandbox import _sanitized_environment, _terminate_process  # noqa: E402
from paperpilot_agentic.training_profiles import load_training  # noqa: E402
from paperpilot_agentic.training_state import atomic_json, file_sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--db', required=True)
    args = parser.parse_args()
    record = json.loads((Path(args.db).parent/'campaign.json').read_bytes())
    runtime = ResearchAgentRuntime(workspace=args.workspace, db_path=args.db)
    binding, run_id = record['binding_id'], record['run_id']
    root, manifest = load_training(runtime, binding)
    ledger = campaign(runtime)
    ledger.recover()
    folder = runtime.workspace/'method-training'/'runs'/run_id
    destination = folder/'fault-validation'
    if destination.exists():
        raise ValueError('Fault validation already started; inspect existing receipts, never erase them')
    destination.mkdir()
    state = json.loads((folder/'state.json').read_bytes())
    source = folder/'recovery-resumed'/'checkpoints'
    first = next(json.loads(line) for line in (source/'commits.jsonl').read_text().splitlines()
                 if json.loads(line)['step'] == 3)
    if file_sha(source/first['file']) != first['sha256']:
        raise ValueError('Original step-three checkpoint changed')
    output = destination/'interrupted'
    checkpoint = output/'checkpoints'
    checkpoint.mkdir(parents=True)
    shutil.copy2(source/first['file'], checkpoint/first['file'])
    atomic_json(checkpoint/'latest.json', first)
    token, seconds = ledger.reserve(str(output), 180)
    job = {'profile': str(root), 'output': str(output), 'ledger': str(ledger.path), 'token': token,
           'binding_id': binding, 'stage': 'recovery', 'seed': 7, 'variant': 'baseline', 'pause_at': 3}
    job_path = output/('job-'+token+'.json')
    atomic_json(job_path, job)
    env = _sanitized_environment(SandboxPolicy(network='host', env_allowlist=('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP')), {})
    with (output/'forced-stdout.txt').open('wb') as stdout, (output/'forced-stderr.txt').open('wb') as stderr:
        process = subprocess.Popen([manifest['python'], '-I', '-B', '-u', str(root/'worker.py'), str(job_path)],
            cwd=root, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, env=env,
            start_new_session=os.name == 'posix', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        started = time.monotonic()
        killed = False
        while process.poll() is None and time.monotonic()-started < min(seconds, 30):
            if (checkpoint/'commits.jsonl').exists():
                _terminate_process(process)
                killed = True
                break
            time.sleep(.005)
        if process.poll() is None:
            _terminate_process(process)
        process.wait(timeout=20)
    ledger.recover()
    row = next(r for r in ledger.snapshot()['launches'] if r['token'] == token)
    if not killed or row['status'] != 'INTERRUPTED' or (output/('receipt-'+token+'.json')).exists():
        raise ValueError('Forced interruption did not hit the uncommitted receipt window; keep logs as failed verification')
    restarted, path = run_job(runtime, run_id, root, manifest, output, ledger, stage='recovery')
    continuous = json.loads(Path(state['checks']['recovery']['continuous']['path']).read_bytes())
    keys = ('step', 'final_state_sha256', 'optimizer_sha256', 'rng_sha256', 'losses', 'gradient_norms')
    if restarted['status'] != 'COMPLETED' or any(restarted[k] != continuous[k] for k in keys):
        raise ValueError('Forced process recovery differs from continuous run')
    # Corrupt another copy; torch loads on CPU only, never alters the original.
    bad = destination/'corrupted'/'checkpoints'
    bad.mkdir(parents=True)
    latest = json.loads((checkpoint/'latest.json').read_bytes())
    shutil.copy2(checkpoint/latest['file'], bad/latest['file'])
    atomic_json(bad/'latest.json', latest)
    with (bad/latest['file']).open('r+b') as stream:
        stream.seek(100)
        stream.write(b'corrupt')
    bad_job = destination/'corrupt-job.json'
    atomic_json(bad_job, {'folder': str(bad), 'identity': latest['identity'], 'state_module': str(root/'training_state.py')})
    code = ('import importlib.util,json,sys; from pathlib import Path; '
            'j=json.loads(Path(sys.argv[1]).read_bytes()); '
            's=importlib.util.spec_from_file_location("owned",j["state_module"]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'm.load_checkpoint(j["folder"],j["identity"])')
    check = subprocess.run([manifest['python'], '-I', '-B', '-c', code, str(bad_job)],
                           capture_output=True, text=True, env=env, timeout=45)
    if check.returncode == 0 or 'checkpoint source, identity or payload changed' not in check.stderr:
        raise ValueError('Corrupted checkpoint not rejected as expected')
    report = {'binding_id': binding, 'forced_pid': process.pid, 'process_exit_code': process.returncode,
        'unknown_receipt_replayed': False, 'dead_confirmed_before_restart': True,
        'interruption_charge_seconds': row['charged'], 'checkpoint_step': 3,
        'exactly_equal': list(keys), 'resumed_receipt': {'path': str(path), 'sha256': file_sha(path)},
        'corrupted_checkpoint_rejected': True, 'corruption_check_used_gpu': False,
        'budget': ledger.snapshot(), 'new_llm_calls': 0}
    atomic_json(destination/'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
