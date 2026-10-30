"""Unknown transport outcomes must retain safe causes and avoid blind retries."""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from evals.interview.budget import CampaignBudget, RecordedProvider, UncertainRequest
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget, LLMCallPolicy
from paperpilot_agentic.providers import OpenAICompatibleLLM


def lost_connection():
    request = httpx.Request('POST', 'https://example.invalid/SECRET',
                            headers={'Authorization': 'SECRET'})
    os_error = PermissionError(13, 'SECRET')
    os_error.winerror = 10013
    connection = httpx.ConnectError('SECRET', request=request)
    connection.__cause__ = os_error
    error = APIConnectionError(message='SECRET', request=request)
    error.__cause__ = connection
    return error


def provider(error, *, retries=2):
    model = object.__new__(OpenAICompatibleLLM)
    model.provider, model.model, model.thinking_enabled = 'qwen', 'fixed', False
    model.max_tokens, model.temperature, model.max_retries = 100, 0, retries
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        raise error
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    return model, calls


@pytest.mark.parametrize('kind', ['connect', 'timeout'])
def test_unknown_connection_outcome_is_not_retried_even_with_default_retry_allowance(monkeypatch, kind):
    error = lost_connection() if kind == 'connect' else APITimeoutError(
        request=httpx.Request('POST', 'https://example.invalid'))
    model, calls = provider(error)
    monkeypatch.setattr('paperpilot_agentic.providers.time.sleep', lambda _: None)
    with pytest.raises(type(error)) as failure:
        model.complete([], before_attempt=lambda: None)
    assert failure.value is error
    assert len(calls) == 1


def test_connection_cause_is_retained_in_campaign_and_runtime_without_secret_messages(tmp_path):
    model, calls = provider(lost_connection(), retries=0)
    budget = CampaignBudget(tmp_path/'receipts', {'model': 'fixed', 'source_sha': 'fixture'})
    recorded = RecordedProvider(model, budget, phase='answer')
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Question', Budget())
    runtime = PersistentLLMRuntime(db, recorded)
    with pytest.raises(APIConnectionError):
        runtime.complete(rid, [], policy=LLMCallPolicy(
            stage='answer_first', max_output_tokens=100, thinking_enabled=False))
    receipt = json.loads((tmp_path/'receipts/failure-001.json').read_text())
    assert receipt['category'] == 'CONNECTION_ERROR'
    assert receipt['transport_phase'] == 'connect'
    assert receipt['cause_chain'] == [
        {'type': 'APIConnectionError'}, {'type': 'ConnectError'},
        {'type': 'PermissionError', 'errno': 13, 'winerror': 10013}]
    stored = db.list_llm_calls(rid)[0]['error']
    assert '10013' in stored and 'ConnectError' in stored
    assert 'SECRET' not in stored
    assert 'SECRET' not in json.dumps(receipt)
    with pytest.raises(UncertainRequest):
        recorded.complete([], max_tokens=100)
    assert len(calls) == budget.snapshot()['actual_requests'] == 1


def test_explicit_retryable_status_still_uses_guarded_retry(monkeypatch):
    request = httpx.Request('POST', 'https://example.invalid')
    error = APIStatusError('busy', response=httpx.Response(503, request=request), body={})
    model, calls = provider(error, retries=1)
    guarded = []
    monkeypatch.setattr('paperpilot_agentic.providers.time.sleep', lambda _: None)
    with pytest.raises(APIStatusError):
        model.complete([], before_attempt=lambda: guarded.append(True))
    assert len(calls) == len(guarded) == 2
