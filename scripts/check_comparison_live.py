"""Freeze local paper scope, then run an isolated HTTP comparison task.

This is a real-paper usability check, not QASPER or independently scored accuracy.
No training is authorized. Every request and unchanged result is retained.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from check_session_entry_live import RecordedProvider, save

QUESTION = ('请逐篇整理所选变化检测论文的本文方法、优秀模块、实验任务、数据集、数据划分、'
            '训练与评估条件、指标口径和报告结果。每个不同实验设置分别记录，缺失项保留；'
            '只在有依据的相同条件下对照，不把前人工作或不同任务结果混为一谈。')


def frozen_scope(settings):
    with sqlite3.connect(settings.memory_db.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT sha256,title,chunk_fingerprint,page_count FROM papers '
            'WHERE namespace=? AND active=1 AND lower(title) LIKE ?',
            (settings.memory_namespace, '%change detection%'))]
    rows.sort(key=lambda row: hashlib.sha256(('20261001:' + row['sha256']).encode()).hexdigest())
    if len(rows) < 20:
        raise ValueError('Fewer than 20 eligible indexed papers; do not silently widen selection')
    return {'selection': 'first 20 by SHA256(20261001:paper_sha256), active local change detection titles',
            'papers': rows[:20], 'question': QUESTION, 'independent_accuracy_benchmark': False}


def main():
    from paperpilot_agentic.config import RuntimeSettings, read_dashscope_api_key_file
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--scope', type=Path)
    parser.add_argument('--limit', type=int, choices=[2, 20], default=2)
    parser.add_argument('--model', default='qwen3-235b-a22b-instruct-2507')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--max-requests', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=400000)
    parser.add_argument('--execute-paid', action='store_true')
    args = parser.parse_args()
    settings = RuntimeSettings.from_env()
    if args.freeze:
        save(args.freeze, frozen_scope(settings))
        print(json.dumps({'scope': str(args.freeze), 'papers': 20, 'actual_requests': 0}))
        return
    if not args.scope or not 1 <= args.max_requests <= 240 or args.max_tokens <= 0:
        parser.error('Supply frozen scope and positive caps (at most 240 requests)')
    scope = json.loads(args.scope.read_text(encoding='utf-8'))
    papers = scope['papers'][:args.limit]
    if not args.execute_paid:
        print(json.dumps({'papers': [p['title'] for p in papers], 'question': scope['question'],
                          'model': args.model, 'actual_requests': 0}, ensure_ascii=False))
        return
    if not args.key_file:
        parser.error('Supply external key file, never a key in arguments')
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.bootstrap import build_runtime
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    from paperpilot_agentic.web_api import create_fastapi_router
    root = Path(__file__).resolve().parents[1]
    folder = Path(tempfile.mkdtemp(prefix='comparison-live-', dir=root / 'outputs'))
    settings = replace(settings, provider='qwen', model=args.model, thinking_enabled=False,
        api_key=read_dashscope_api_key_file(args.key_file), timeout_seconds=90,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', workspace=folder,
        runtime_db=folder / 'runtime.sqlite3')
    os.environ['PAPERPILOT_ALLOW_MODEL_DOWNLOAD'] = 'false'
    runtime = build_runtime(settings)
    with sqlite3.connect(settings.memory_db.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        current = {r['sha256']: dict(r) for r in db.execute(
            'SELECT sha256,chunk_fingerprint FROM papers WHERE namespace=? AND active=1',
            (settings.memory_namespace,))}
    if any(p['sha256'] not in current or current[p['sha256']]['chunk_fingerprint'] != p['chunk_fingerprint'] for p in papers):
        raise ValueError('Frozen paper identity changed')
    provider = RecordedProvider(OpenAICompatibleLLM(settings, max_retries=0), folder, args.max_requests)
    runtime.llm_runtime.model = provider
    identity = hashlib.sha256()
    for path in sorted((root / 'src').rglob('*.py')):
        identity.update(path.relative_to(root).as_posix().encode())
        identity.update(path.read_bytes())
    save(folder / 'protocol.json', {'scope': scope, 'executed_papers': papers, 'model': args.model,
        'code_sha256': identity.hexdigest(), 'max_requests': args.max_requests, 'max_tokens': args.max_tokens,
        'training_authorized': False, 'independent_accuracy_benchmark': False})
    print(json.dumps({'output': str(folder)}, ensure_ascii=True), flush=True)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime, prefix='/api/agentic'))
    started = time.monotonic()
    with TestClient(app) as client:
        body = {'request_id': 'frozen-comparison', 'action': 'evidence_comparison', 'question': scope['question'],
                'visual_policy': 'off', 'paper_ids': [p['sha256'] for p in papers]}
        response = client.post('/api/agentic/turns', json=body)
        response.raise_for_status()
        rid = response.json()['run_id']
        while True:
            view = client.get(f'/api/agentic/runs/{rid}/view').json()
            if not view['worker_active']:
                break
            time.sleep(.5)
        task = view.get('research_task')
        if not task or task['state'] != 'AWAITING_BUDGET':
            save(folder / 'entry-failure.json', view)
            raise ValueError('Frozen comparison did not reach research budget confirmation')
        budget = {**task['suggested_budget'], 'max_llm_calls': args.max_requests,
                  'max_tokens': args.max_tokens, 'max_seconds': 3600}
        approve = {'request_id': 'frozen-comparison-budget', 'action': 'approve_research',
                   'conversation_id': view['conversation_id'], 'parent_run_id': rid,
                   'plan_revision': task['plan_revision'], 'budget': budget}
        response = client.post('/api/agentic/turns', json=approve)
        response.raise_for_status()
        while True:
            response = client.get(f'/api/agentic/runs/{rid}/view')
            response.raise_for_status()
            view = response.json()
            if not view['worker_active'] and view['research_task']['state'] not in {'READY', 'RUNNING'}:
                break
            if time.monotonic() - started > 3700:
                client.post(f'/api/agentic/runs/{rid}/cancel')
                save(folder / 'timeout-view.json', view)
                raise TimeoutError('Observation exceeded one hour; no automatic replay')
            time.sleep(1)
        save(folder / 'view.json', view)
        save(folder / 'result.json', runtime.status(rid))
        comparison = view['research_task'].get('comparison') or {}
        summary = {'run_id': rid, 'status': view['status'], 'state': view['research_task']['state'],
                   'papers': len(papers), 'records': len(comparison.get('records', [])),
                   'covered_papers': sum(p['record_count'] > 0 for p in comparison.get('coverage', [])),
                   'actual_requests': provider.calls, 'tokens': view['budget']['used_tokens'],
                   'elapsed_seconds': round(time.monotonic() - started, 2),
                   'fact_correctness': 'NOT_INDEPENDENTLY_SCORED'}
        save(folder / 'summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
