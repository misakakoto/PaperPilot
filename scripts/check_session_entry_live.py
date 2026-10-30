"""Known-paper HTTP entry regression, not an independent accuracy benchmark.

Dry-run by default. Only original local public papers are used; no budget approval,
experiment execution, model download or previous-run replay. All receipts are kept
in a fresh ignored output directory, including unsuccessful answers.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CASES = [
    {'id': 'ordinary-inventory', 'papers': ['ChangeMamba'], 'expected': 'answer',
     'question': '列出 ChangeMamba 论文使用的数据集，并说明各自对应的任务。'},
    {'id': 'ordinary-training-question', 'papers': ['FlowMamba'], 'expected': 'answer',
     'question': 'FlowMamba 论文是用什么数据训练的，真实灾害场景验证的数据又是什么？'},
    {'id': 'exhaustive-review', 'papers': ['ChangeMamba', 'FlowMamba'], 'expected': 'budget',
     'question': '请逐篇覆盖这两篇论文，制作数据集、方法模块、实验条件的完整对照表，并把缺失项单独列出。'},
    {'id': 'maintained-notebook', 'papers': ['ChangeMamba'], 'expected': 'budget',
     'question': '请建立持续维护的研究笔记，记录这篇论文的待核验假设和后续研究待办，供以后多轮更新。'},
]


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


class RecordedProvider:
    def __init__(self, provider, folder, max_requests=32):
        self.provider = provider
        self.folder = folder
        self.calls = 0
        self.max_requests = max_requests

    def __getattr__(self, key):
        return getattr(self.provider, key)

    def complete(self, messages, *, before_attempt, **kwargs):
        from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
        number = None

        def reserve():
            nonlocal number
            if self.calls >= self.max_requests:
                raise LLMBudgetExhausted('Diagnostic shared request cap reached')
            before_attempt()
            self.calls += 1
            number = self.calls
            save(self.folder / f'request-{number:03}.json', {'messages': messages, **kwargs})

        try:
            response = self.provider.complete(messages, before_attempt=reserve, **kwargs)
        except Exception as exc:
            if number is not None:
                from paperpilot_agentic.provider_diagnostics import safe_provider_failure
                diagnostics = safe_provider_failure(exc)
                save(self.folder / f'failure-{number:03}.json', {
                    **diagnostics, 'error_category': diagnostics['category'],
                    'requested_model': getattr(self.provider, 'model', None),
                })
            raise
        save(self.folder / f'response-{number:03}.json', asdict(response))
        return response


def run(args):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.config import RuntimeSettings, read_dashscope_api_key_file
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.web_api import create_fastapi_router
    from paperpilot_agentic.workbench import TERMINAL

    root = Path(__file__).resolve().parents[1]
    cases = [case for case in CASES if args.case is None or case['id'] == args.case]
    (root / 'outputs').mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='session-entry-live-', dir=root / 'outputs'))
    base = RuntimeSettings.from_env()
    settings = replace(base, provider='qwen', model=args.model, thinking_enabled=False,
        api_key=read_dashscope_api_key_file(args.key_file),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', timeout_seconds=60,
        workspace=folder, runtime_db=folder / 'runtime.sqlite3')
    os.environ['PAPERPILOT_ALLOW_MODEL_DOWNLOAD'] = 'false'
    runtime = build_runtime(settings)
    provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), folder,
                                max_requests=args.max_requests)
    runtime.llm_runtime.model = provider
    identity = hashlib.sha256()
    for source in sorted((root / 'src').rglob('*.py')):
        identity.update(source.relative_to(root).as_posix().encode())
        identity.update(source.read_bytes())
    save(folder / 'protocol.json', {'cases': cases, 'model': args.model,
        'code_sha256': identity.hexdigest(), 'max_actual_requests': args.max_requests,
        'independent_benchmark': False, 'question_policy': 'known-paper development regression',
        'visual_policy': 'off', 'experiment_authorization': False,
        'reader_protocol': runtime.text_reader_protocol})
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime, prefix='/api/agentic'))
    print(json.dumps({'output': str(folder)}, ensure_ascii=True), flush=True)
    rows = []
    with TestClient(app) as client:
        ids = {}
        for query in sorted({paper for case in cases for paper in case['papers']}):
            response = client.get('/api/agentic/papers', params={'query': query})
            response.raise_for_status()
            candidates = response.json()['items']
            if len(candidates) != 1:
                raise ValueError(f'{query}: select exactly one known indexed paper before running')
            ids[query] = candidates[0]['sha256']
        save(folder / 'scope.json', ids)
        for case in cases:
            started = time.monotonic()
            body = {'request_id': f"entry-{case['id']}", 'action': 'auto', 'visual_policy': 'off',
                    'question': case['question'], 'paper_ids': [ids[name] for name in case['papers']]}
            save(folder / f"{case['id']}-submission.json", body)
            response = client.post('/api/agentic/turns', json=body)
            response.raise_for_status()
            rid = response.json()['run_id']
            while True:
                response = client.get(f'/api/agentic/runs/{rid}/view')
                response.raise_for_status()
                view = response.json()
                if view['status'] in TERMINAL | {'WAITING_APPROVAL'} and not view['worker_active']:
                    break
                if time.monotonic() - started > 900:
                    client.post(f'/api/agentic/runs/{rid}/cancel')
                    raise TimeoutError('Observation timed out; no automatic retry')
                time.sleep(.5)
            result = runtime.status(rid)
            save(folder / f"{case['id']}-result.json", result)
            save(folder / f"{case['id']}-view.json", view)
            replay = client.post('/api/agentic/turns', json=body)
            replay.raise_for_status()
            task = view['research_task']
            passed = (view['status'] == 'COMPLETED' and bool(view['message']['answer']) and not task
                      if case['expected'] == 'answer'
                      else bool(task and task['state'] == 'AWAITING_BUDGET'))
            row = {'case': case['id'], 'run_id': rid, 'entry_pass': passed,
                'status': view['status'], 'answer_chars': len(view['message']['answer']),
                'calls': view['budget']['used_llm_calls'], 'tokens': view['budget']['used_tokens'],
                'same_submission_reused': (replay.json().get('reused') is True
                    and replay.json().get('run_id') == rid
                    and runtime.status(rid)['run']['used_llm_calls'] == view['budget']['used_llm_calls']),
                'elapsed_seconds': round(time.monotonic() - started, 2),
                'fact_correctness': 'NOT_INDEPENDENTLY_SCORED',
                'routing': result['run']['metadata'].get('research_route_decision')}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    save(folder / 'summary.json', {'rows': rows, 'actual_requests': provider.calls})
    if not all(r['entry_pass'] and r['same_submission_reused'] for r in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3-30b-a3b-instruct-2507')
    parser.add_argument('--case', choices=[case['id'] for case in CASES])
    parser.add_argument('--max-requests', type=int, default=32)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 32:
        parser.error('max-requests must be between 1 and 32')
    cases = [case for case in CASES if args.case is None or case['id'] == args.case]
    if not args.execute_paid:
        print(json.dumps({'cases': cases, 'model': args.model, 'actual_requests': 0,
                          'maximum_if_executed': args.max_requests}, ensure_ascii=True))
        return
    if not args.key_file:
        parser.error('Provide an external key file, not a credential in arguments')
    run(args)


if __name__ == '__main__':
    main()
