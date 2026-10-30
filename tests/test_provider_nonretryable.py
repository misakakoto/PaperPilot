from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from paperpilot_agentic.providers import OpenAICompatibleLLM


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
def test_nonretryable_provider_error_is_not_masked_by_guard(status, monkeypatch):
    provider = object.__new__(OpenAICompatibleLLM)
    provider.provider, provider.model = 'qwen', 'fixture'
    provider.thinking_enabled, provider.temperature = False, 0
    provider.max_tokens, provider.max_retries = 100, 2
    error = APIStatusError('fixture permission or request error', response=httpx.Response(
        status, request=httpx.Request('POST', 'https://example.invalid')), body=None)
    attempts = []
    def send(**kwargs):
        raise error
    def guard():
        attempts.append(1)
        if len(attempts) > 1:
            raise RuntimeError('Local budget exhausted')
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    monkeypatch.setattr('paperpilot_agentic.providers.time.sleep',
                        lambda _: pytest.fail('Must not retry permanent provider error'))
    with pytest.raises(APIStatusError) as caught:
        provider.complete([], before_attempt=guard)
    assert caught.value is error
    assert len(attempts) == 1
