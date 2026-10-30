"""Recorded, explicit cumulative-budget extension of a paused diagnostic.

Does not resume UNKNOWN outcomes, change models/code, authorize training or reset
usage. Dry-run by default. Original failures and initial caps remain on disk.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from check_session_entry_live import RecordedProvider, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--additional-requests', type=int, default=12)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output.resolve()
    if not folder.is_relative_to((root / 'outputs').resolve()) or not 1 <= args.additional_requests <= 24:
        parser.error('Only an isolated repository output and 1..24 additional requests are allowed')
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    prior = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    if prior['state'] != 'PAUSED_BUDGET':
        raise ValueError('Only a known budget pause can be extended; UNKNOWN is never replayed')
    ceiling = prior['actual_requests'] + args.additional_requests
    identity = hashlib.sha256()
    for path in sorted((root / 'src').rglob('*.py')):
        identity.update(path.relative_to(root).as_posix().encode())
        identity.update(path.read_bytes())
    if identity.hexdigest() != protocol['code_sha256']:
        raise ValueError('Source changed; do not reinterpret paid responses under new code')
    if not args.execute_paid:
        print(json.dumps({'prior_requests': prior['actual_requests'], 'new_total_ceiling': ceiling,
                          'actual_requests': 0, 'model': protocol['model'],
                          'source_fingerprint_verified': True,
                          'runtime_state_verified': False}))
        return
    if not args.key_file:
        parser.error('Supply external key path, never a key in arguments')
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.config import RuntimeSettings, read_dashscope_api_key_file
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.research_task import ResearchTask
    from paperpilot_agentic.web_api import create_fastapi_router
    settings = replace(RuntimeSettings.from_env(), provider='qwen', model=protocol['model'],
        thinking_enabled=False, api_key=read_dashscope_api_key_file(args.key_file), timeout_seconds=90,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', workspace=folder,
        runtime_db=folder / 'runtime.sqlite3')
    os.environ['PAPERPILOT_ALLOW_MODEL_DOWNLOAD'] = 'false'
    runtime = build_runtime(settings)
    rid = prior['run_id']
    task = ResearchTask(runtime, rid)
    task.check_protocol()
    task.check_unknown()
    state = task.load()
    run = runtime.db.get_run(rid)
    receipts = sorted(folder.glob('request-*.json'))
    numbers = sorted(int(p.stem.split('-')[1]) for p in receipts)
    if state['state'] != 'PAUSED_BUDGET' or numbers != list(range(1, run['used_llm_calls'] + 1)):
        raise ValueError('Runtime state/receipts changed; inspect before extending')
    budget = {**state['approved_budget'], 'max_llm_calls': ceiling}
    provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), folder, ceiling)
    provider.calls = run['used_llm_calls']
    runtime.llm_runtime.model = provider
    tag = f'resume-{run["used_llm_calls"]}-{ceiling}'
    save(folder / f'{tag}-protocol.json', {'old_budget': state['approved_budget'], 'new_budget': budget,
        'plan_revision': state['plan_revision'], 'model': protocol['model'],
        'code_sha256': identity.hexdigest(), 'training_authorized': False,
        'authorization': 'User explicitly allowed real API diagnostic testing; no billing settings changed'})
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime, prefix='/api/agentic'))
    started = time.monotonic()
    with TestClient(app) as client:
        view = client.get(f'/api/agentic/runs/{rid}/view').json()
        response = client.post('/api/agentic/turns', json={'request_id': 'comparison-' + tag,
            'action': 'continue_research', 'conversation_id': view['conversation_id'],
            'parent_run_id': rid, 'plan_revision': state['plan_revision'], 'budget': budget})
        response.raise_for_status()
        while True:
            response = client.get(f'/api/agentic/runs/{rid}/view')
            response.raise_for_status()
            view = response.json()
            if not view['worker_active'] and view['research_task']['state'] not in {'READY', 'RUNNING'}:
                break
            if time.monotonic() - started > 600:
                client.post(f'/api/agentic/runs/{rid}/cancel')
                save(folder / f'{tag}-timeout.json', view)
                raise TimeoutError('Observation timeout; no automatic replay')
            time.sleep(1)
        save(folder / f'{tag}-view.json', view)
        save(folder / f'{tag}-result.json', runtime.status(rid))
        comparison = view['research_task'].get('comparison') or {}
        summary = {'run_id': rid, 'state': view['research_task']['state'], 'status': view['status'],
            'total_requests': provider.calls, 'additional_requests': provider.calls - run['used_llm_calls'],
            'tokens': view['budget']['used_tokens'], 'records': len(comparison.get('records', [])),
            'covered_papers': sum(p['record_count'] > 0 for p in comparison.get('coverage', [])),
            'fact_correctness': 'NOT_INDEPENDENTLY_SCORED'}
        save(folder / f'{tag}-summary.json', summary)
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
