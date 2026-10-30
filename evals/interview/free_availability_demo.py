"""Offline fault injection through the real ledger; never calls a remote SDK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import APIConnectionError, APIStatusError

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_attempts import AttemptLedger, UnknownRequestResult
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
from paperpilot_agentic.models import Budget, LLMCallPolicy
from paperpilot_agentic.providers import OpenAICompatibleLLM

MESSAGES = [{'role': 'user', 'content': 'Return JSON for an offline ledger demonstration.'}]
POLICY = LLMCallPolicy('offline_demo', 100, False, True, reserve_class='repair')


def bridge(db, sdk_send):
    settings = RuntimeSettings(provider='qwen', api_key='offline-only-not-a-real-key',
        model=PRIMARY_MODEL, fallback_models=(BACKUP_MODEL,),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        thinking_enabled=False, temperature=0, max_tokens=100, timeout_seconds=1,
        workspace=db.path.parent, runtime_db=db.path, memory_db=None,
        memory_namespace='offline-demo', paper_roots=())
    provider = OpenAICompatibleLLM(settings, max_retries=0)
    # Replace the entire SDK seam before any operation; no credentials are read.
    provider._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=sdk_send)))
    return PersistentLLMRuntime(db, provider)


def quota_then_restore(folder):
    db = RuntimeDatabase(folder / 'quota.sqlite3')
    rid = db.create_run('Offline quota fault.', Budget(max_tokens=200000),
                       {'execution_profile': 'daily'})
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        if kwargs['model'] == PRIMARY_MODEL:
            raise APIStatusError('Injected free quota error', response=httpx.Response(403,
                request=httpx.Request('POST', 'https://example.invalid/offline')),
                body={'error': {'code': 'AllocationQuota.FreeTierOnly'}})
        return SimpleNamespace(model=kwargs['model'], id='offline-simulated-receipt',
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ready":true}'), finish_reason='stop')],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    runtime = bridge(db, send)
    runtime.complete(rid, MESSAGES, policy=POLICY)
    before = runtime.budget_snapshot(rid)
    restored = bridge(RuntimeDatabase(db.path), send)
    assert restored.budget_snapshot(rid) == before
    # This is a deliberately requested *new stage*, not a replay of the
    # completed response. Restoring the ledger alone sends nothing.
    restored.complete(rid, [{'role': 'user', 'content': 'Verify the next offline stage as JSON.'}],
                      policy=LLMCallPolicy('offline_next_stage', 100, False, True, reserve_class='repair'))
    after = restored.budget_snapshot(rid)
    assert sent == [PRIMARY_MODEL, BACKUP_MODEL, BACKUP_MODEL]
    assert before['used_llm_calls'] == 2 and after['used_llm_calls'] == 3
    assert before['reserved_unknown_tokens'] == after['reserved_unknown_tokens'] > 0
    return {'scenario': 'quota-fallback-restart', 'status': 'PASS', 'sdk_models': sent,
        'continuation_kind': 'explicit_next_stage_not_replay',
        'calls_before_restart': before['used_llm_calls'], 'calls_after_restart': after['used_llm_calls'],
        'unknown_usage_hold_before': before['reserved_unknown_tokens'],
        'unknown_usage_hold_after': after['reserved_unknown_tokens'],
        'states': [r['state'] for r in AttemptLedger(db).snapshot(rid)['attempts']],
        'simulated_confirmed_tokens': after['used_tokens'], 'remote_requests': 0}


def unknown_then_restore(folder):
    db = RuntimeDatabase(folder / 'unknown.sqlite3')
    rid = db.create_run('Offline connection fault.', Budget(max_tokens=200000),
                       {'execution_profile': 'daily'})
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        raise APIConnectionError(request=httpx.Request('POST', 'https://example.invalid/offline'))
    runtime = bridge(db, send)
    try:
        runtime.complete(rid, MESSAGES, policy=POLICY)
    except APIConnectionError:
        pass
    else:
        raise AssertionError('The injected unknown request must stop')
    before = runtime.budget_snapshot(rid)
    restored = bridge(RuntimeDatabase(db.path), send)
    try:
        restored.complete(rid, MESSAGES, policy=POLICY)
    except UnknownRequestResult:
        pass
    else:
        raise AssertionError('An unknown outcome cannot be replayed')
    after = restored.budget_snapshot(rid)
    assert before == after and len(sent) == after['used_llm_calls'] == 1
    return {'scenario': 'unknown-restart-block', 'status': 'PASS', 'sdk_models': sent,
        'calls_before_restart': before['used_llm_calls'], 'calls_after_restart': after['used_llm_calls'],
        'unknown_usage_hold_before': before['reserved_unknown_tokens'],
        'unknown_usage_hold_after': after['reserved_unknown_tokens'],
        'states': [r['state'] for r in AttemptLedger(db).snapshot(rid)['attempts']],
        'remote_requests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Never delete or overwrite an earlier demonstration or acceptance ledger.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {'mode': 'OFFLINE_FAULT_INJECTION', 'remote_requests': 0,
              'usage_is_simulated': True, 'answer_quality': 'NOT_EVALUATED',
              'results': [quota_then_restore(args.output_dir), unknown_then_restore(args.output_dir)]}
    (args.output_dir / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                              encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
