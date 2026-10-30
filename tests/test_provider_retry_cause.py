from types import SimpleNamespace
import pytest
from openai import OpenAIError
from paperpilot_agentic.providers import OpenAICompatibleLLM


def test_local_retry_guard_preserves_original_provider_failure(monkeypatch):
    provider = object.__new__(OpenAICompatibleLLM)
    provider.provider = 'qwen'
    provider.model = 'fixture'
    provider.thinking_enabled = False
    provider.temperature = 0
    provider.max_tokens = 100
    provider.max_retries = 2
    initial = OpenAIError('Initial transport error')
    requests = []
    def send(**kwargs):
        requests.append(kwargs)
        raise initial
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    monkeypatch.setattr('paperpilot_agentic.providers.time.sleep', lambda _: None)
    def guard():
        if requests:
            raise RuntimeError('Attempt cap')
    with pytest.raises(RuntimeError, match='Attempt cap') as caught:
        provider.complete([], before_attempt=guard)
    assert caught.value.__cause__ is initial
    assert len(requests) == 1
