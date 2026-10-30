"""Host launcher for one fixed numerical recovery probe under the original budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime  # noqa: E402
from paperpilot_agentic.paired_training import campaign, confirm_exits  # noqa: E402
from paperpilot_agentic.models import SandboxPolicy  # noqa: E402
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
    if runtime.db.get_run(record['run_id'])['status'] != 'COMPLETED':
        raise ValueError('Run this probe only after the approved formal worker has finished')
    root, manifest = load_training(runtime, record['binding_id'])
    ledger = campaign(runtime)
    ledger.recover()
    confirm_exits(ledger)
    output = runtime.workspace/'method-training'/'runs'/record['run_id']/'rng-cursor-probe'
    if output.exists():
        raise ValueError('Numerical probe already started; preserve receipts and inspect rather than rerun')
    output.mkdir()
    source = Path(__file__).with_name('probe_training_rng.py')
    token, seconds = ledger.reserve(str(output), 60)
    job_path = output/'job.json'
    atomic_json(job_path, {'profile': str(root), 'output': str(output), 'ledger': str(ledger.path),
        'token': token, 'binding_id': record['binding_id'], 'source_sha256': file_sha(source)})
    env = _sanitized_environment(SandboxPolicy(network='host', env_allowlist=('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP')), {})
    with (output/'stdout.txt').open('wb') as stdout, (output/'stderr.txt').open('wb') as stderr:
        process = subprocess.Popen([manifest['python'], '-I', '-B', str(source), str(job_path)],
            cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=os.name == 'posix', creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            process.wait(timeout=20)
    ledger.recover()
    confirm_exits(ledger)
    if process.returncode or not (output/'verification.json').exists():
        raise ValueError('Numerical recovery probe failed; preserve stderr and reservation')
    print((output/'verification.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
