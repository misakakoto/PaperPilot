from dataclasses import replace
from types import SimpleNamespace

import pytest

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_attempts import AttemptLedger
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.model_policy import FREE_MODELS
from paperpilot_agentic.models import Budget, LLMCallPolicy
from paperpilot_agentic.qwen_vision import QwenVisionEvidenceAdapter
from test_daily_availability import fixture_provider, provider_error, success


MODEL = 'qwen3.5-plus-2026-02-15'


@pytest.mark.parametrize('model', [MODEL, 'qwen3.5-flash-2026-02-23', 'qwen3.5-plus-2026-04-20'])
def test_hybrid_text_json_uses_non_thinking_and_durable_receipt(tmp_path, model):
    sent = []

    def send(**kwargs):
        sent.append(kwargs)
        return success(kwargs['model'])

    provider = fixture_provider(tmp_path, send)
    provider.model, provider.fallback_models = model, ()
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Return JSON.', Budget(), {'execution_profile': 'daily'})
    runtime = PersistentLLMRuntime(db, provider)
    response = runtime.complete(rid, [{'role': 'user', 'content': 'Return JSON.'}],
                                policy=LLMCallPolicy('fixture', 100, False, True))
    assert response.model == model
    assert sent[0]['extra_body'] == {'enable_thinking': False}
    assert sent[0]['response_format'] == {'type': 'json_object'}
    assert db.get_run(rid)['used_llm_calls'] == 1
    assert db.get_run(rid)['used_tokens'] == 15
    assert db.get_run(rid)['metadata']['model_policy']['capabilities'][model]['thinking']


@pytest.mark.parametrize('model', [MODEL, 'qwen3.5-flash-2026-02-23', 'qwen3.5-plus-2026-04-20'])
def test_hybrid_visual_json_retains_thinking_control_and_ledger(tmp_path, model):
    sent = []

    def send(**kwargs):
        sent.append(kwargs)
        return success(kwargs['model'])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    adapter = QwenVisionEvidenceAdapter(api_key='offline-fixture', model=model, client=client)
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Read image.', Budget(max_tokens=200000), {'execution_profile': 'daily'})
    bound = adapter.for_run(db, rid)
    assert bound.durable_dispatch
    bound.request_executor.send({'model': model, 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fixture'}},
        {'type': 'text', 'text': 'Return JSON.'}]}], 'max_tokens': 100,
        'response_format': {'type': 'json_object'}, 'extra_body': {'enable_thinking': False}},
        stage='fixture')
    assert sent[0]['extra_body'] == {'enable_thinking': False}
    receipt = AttemptLedger(db).snapshot(rid)
    assert len(sent) == db.get_run(rid)['used_llm_calls'] == 1
    assert receipt['attempts'][0]['state'] == 'RETURNED'
    assert receipt['attempts'][0]['reserved_tokens'] >= 16384
    assert receipt['reserved_unknown_tokens'] == 0


def test_single_new_model_quota_error_does_not_call_legacy_backup(tmp_path):
    sent = []

    def send(**kwargs):
        sent.append(kwargs['model'])
        raise provider_error()

    provider = fixture_provider(tmp_path, send)
    provider.model, provider.fallback_models = MODEL, ()
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Q', Budget(), {'execution_profile': 'daily'})
    runtime = PersistentLLMRuntime(db, provider)
    with pytest.raises(type(provider_error())):
        runtime.complete(rid, [{'role': 'user', 'content': 'Return JSON.'}],
                         policy=LLMCallPolicy('fixture', 100, False, True))
    assert sent == [MODEL]
    assert AttemptLedger(db).snapshot(rid)['reserved_unknown_tokens'] > 0


def test_new_config_does_not_resume_checkpoint_with_old_model(tmp_path):
    provider = fixture_provider(tmp_path, lambda **kwargs: success(kwargs['model']))
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Q', Budget(), {'execution_profile': 'daily'})
    provider.for_run(db, rid)
    old_identity = db.get_run(rid)['metadata']['model_policy']
    provider.model, provider.fallback_models = MODEL, ()
    with pytest.raises(ValueError, match='Model policy changed'):
        provider.for_run(db, rid)
    assert db.get_run(rid)['metadata']['model_policy'] == old_identity


def test_model_configuration_supports_new_single_model_without_dotenv(monkeypatch):
    from paperpilot_agentic.config import RuntimeSettings
    monkeypatch.setattr('paperpilot_agentic.config.load_dotenv', lambda **kwargs: False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'offline-fixture')
    monkeypatch.setenv('PAPERPILOT_LLM_PROVIDER', 'qwen')
    monkeypatch.setenv('PAPERPILOT_LLM_MODEL', MODEL)
    monkeypatch.setenv('PAPERPILOT_LLM_FALLBACK_MODELS', '')
    settings = replace(RuntimeSettings.from_env(), thinking_enabled=False)
    assert settings.model == MODEL and settings.fallback_models == ()
    assert FREE_MODELS[MODEL]['free_tier_only_confirmed']
