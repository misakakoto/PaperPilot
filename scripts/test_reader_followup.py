"""Real API follow-up on a preserved QASPER workspace; no gold or manual evidence."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from paperpilot_agentic.bootstrap import build_runtime  # noqa: E402
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.web_api import create_fastapi_router  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--parent-run', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--env-file', required=True, type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous attempts; choose a new output')
    if not args.execute_paid:
        print(json.dumps({'question': args.question, 'calls': 0}))
        return
    load_dotenv(args.env_file, override=False)
    base = RuntimeSettings.from_env(require_api_key=True)
    os.environ.setdefault('PAPERPILOT_EMBEDDING_CACHE', str(base.workspace/'model-cache/embedding'))
    os.environ.setdefault('PAPERPILOT_RERANKER_CACHE', str(base.workspace/'model-cache/reranker'))
    workspace = args.workspace.resolve()
    if not (workspace/'runtime.sqlite3').is_file() or not (workspace/'knowledge.sqlite3').is_file():
        raise FileNotFoundError('Existing QASPER runtime and knowledge databases required')
    settings = dataclasses.replace(base, workspace=workspace, runtime_db=workspace/'runtime.sqlite3',
        memory_db=workspace/'knowledge.sqlite3', memory_namespace='qasper-case', paper_roots=(workspace/'corpus',))
    runtime = build_runtime(settings)
    parent = runtime.status(args.parent_run, include_trace=False)
    if parent['run']['metadata'].get('engine') != 'answer_first_text':
        raise ValueError('Expected text-reader parent')
    args.output.mkdir(parents=True)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        response = client.post(f'/agentic/runs/{args.parent_run}/follow-up', json={'question': args.question})
        response.raise_for_status()
        if response.status_code != 202:
            raise ValueError('Expected 202')
        run = response.json()['run']
        (args.output/'pending.json').write_text(json.dumps(response.json(), ensure_ascii=False), encoding='utf-8')
        deadline = time.monotonic() + run['budget']['max_seconds'] + 30
        while True:
            response = client.get(f'/agentic/runs/{run["run_id"]}')
            response.raise_for_status()
            result = response.json()
            pending = args.output/'result.pending'
            pending.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            pending.replace(args.output/'result.json')
            if result['run']['status'] in {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Inspect saved run; do not blindly resubmit')
            time.sleep(1)
    report = result['run']['metadata'].get('reader_report', {})
    artifact = {'question': args.question, 'run_id': run['run_id'],
        'evaluation_scope': 'FORMAL_API_FOLLOW_UP_REAL_PROVIDER_TEXT_ONLY', 'model': settings.model,
        'reader': {'provider_attempts': 0, 'calls': [], **report},
        'persisted_run': result['run'], 'evidence': result['evidence']}
    (args.output/'reader-result.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'run_id': run['run_id'], 'status': result['run']['status'],
                      'calls': result['run']['used_llm_calls'], 'tokens': result['run']['used_tokens']}))


if __name__ == '__main__':
    main()
