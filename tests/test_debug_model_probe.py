import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from scripts import check_debug_model as module
from paperpilot_agentic.providers import LLMResponse


@pytest.mark.parametrize(('status', 'code', 'expected'), [
    (400, 'Arrearage', 'BILLING_OR_QUOTA'),
    (403, 'AllocationQuota.FreeTierOnly', 'BILLING_OR_QUOTA'),
    (429, 'insufficient_quota', 'BILLING_OR_QUOTA'),
    (429, 'Throttling.RateQuota', 'RATE_LIMIT'),
    (401, 'invalid_api_key', 'AUTH_OR_PERMISSION'),
    (404, 'model_not_found', 'MODEL_OR_ENDPOINT_UNAVAILABLE'),
])
def test_error_categories_do_not_equate_rate_limits_with_empty_balance(status, code, expected):
    error = APIStatusError('do not log body', response=httpx.Response(status,
        request=httpx.Request('POST', 'https://example.invalid')),
        body={'error': {'code': code}})
    assert module.failure_category(error) == expected


def test_timeout_does_not_claim_balance_is_empty():
    error = APITimeoutError(request=httpx.Request('POST', 'https://example.invalid'))
    assert module.failure_category(error) == 'TIMEOUT_UNKNOWN_OUTCOME'


@pytest.mark.parametrize('reason', ['stop', 'length'])
def test_probe_preserves_raw_response_and_never_serializes_credentials(tmp_path, monkeypatch, reason):
    response = LLMResponse(text='{"comparable":false,"reason":"Different splits"}',
                           finish_reason=reason, model='returned-model', prompt_tokens=100)
    class Fake:
        def __init__(self, settings, max_retries):
            assert max_retries == 0
        def complete(self, messages, **kwargs):
            assert kwargs['thinking_enabled'] is False
            kwargs['before_attempt']()
            return response
    monkeypatch.setattr(module, 'OpenAICompatibleLLM', Fake)
    result = module.probe(SimpleNamespace(provider='qwen', model='old-model', api_key='SECRET'),
                          tmp_path/'receipt')
    assert result['actual_requests'] == 1
    assert (result['status'] == 'STRUCTURED_RESPONSE') == (reason == 'stop')
    assert json.loads((tmp_path/'receipt/response.json').read_text())['text'] == response.text
    assert 'SECRET' not in ''.join(p.read_text() for p in (tmp_path/'receipt').iterdir())
