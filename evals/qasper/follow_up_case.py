"""Exercise a real same-conversation follow-up against an existing isolated case."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from paperpilot_agentic.bootstrap import build_runtime  # noqa: E402
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.web_api import create_fastapi_router  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier results; never resubmit on observation timeout')
    parent = json.loads((args.workspace / 'reader-result.json').read_text(encoding='utf-8'))
    if not args.execute_paid:
        print('No request; pass --execute-paid to create a follow-up')
        return
    settings = RuntimeSettings.from_env(require_api_key=True)
    workspace = args.workspace.resolve()
    settings = dataclasses.replace(settings, workspace=workspace,
        runtime_db=workspace/'runtime.sqlite3', memory_db=workspace/'knowledge.sqlite3',
        memory_namespace='qasper-case', paper_roots=(workspace/'corpus',))
    runtime = build_runtime(settings)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    args.output.mkdir(parents=True)
    with TestClient(app) as client:
        response = client.post(f"/agentic/runs/{parent['run_id']}/follow-up",
                               json={'question': args.question})
        response.raise_for_status()
        if response.status_code != 202:
            raise ValueError('Expected background acceptance')
        pending = response.json()
        rid = pending['run']['run_id']
        (args.output/'pending.json').write_text(json.dumps(pending, ensure_ascii=False), encoding='utf-8')
        deadline = time.monotonic() + pending['run']['budget']['max_seconds'] + 30
        while True:
            current = client.get(f'/agentic/runs/{rid}')
            current.raise_for_status()
            result = current.json()
            temporary = args.output/'result.pending'
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(args.output/'result.json')
            if result['run']['status'] in {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Inspect this run ID; do not submit another request')
            time.sleep(1)
        # Recreate the runtime to verify history is persisted rather than in-memory only.
        restored = build_runtime(settings).status(rid, include_trace=False)
        checks = {'same_conversation': result['run']['metadata']['conversation_id'] ==
                  parent['persisted_run']['metadata']['conversation_id'],
                  'answer_restored': restored.get('final_answer') == result.get('final_answer'),
                  'citations_owned': set(result.get('final_answer', {}).get('citation_ids', [])) <=
                  {e['evidence_id'] for e in result['evidence']},
                  'history_is_evidence': result['run']['metadata'].get('conversation_context_is_evidence')}
        (args.output/'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        print(json.dumps({'run_id': rid, 'status': result['run']['status'],
                          'calls': result['run']['used_llm_calls'], 'tokens': result['run']['used_tokens'],
                          'checks': checks}))


if __name__ == '__main__':
    main()
