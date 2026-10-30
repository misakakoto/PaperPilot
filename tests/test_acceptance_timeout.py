"""Acceptance configuration must not shorten the configured SDK deadline."""
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evals.interview import run
from paperpilot_agentic.config import RuntimeSettings
from evals.interview.timeouts import request_timeouts


def fixture_settings(tmp_path):
    return RuntimeSettings(provider='qwen', api_key='offline', model='qwen3-vl-plus',
        base_url='https://example.invalid/v1', thinking_enabled=False, temperature=0,
        max_tokens=128, timeout_seconds=120, workspace=tmp_path,
        runtime_db=tmp_path/'runtime.sqlite3', memory_db=tmp_path/'original.sqlite3',
        memory_namespace='offline', paper_roots=())


def setup_configuration(monkeypatch, tmp_path):
    settings = fixture_settings(tmp_path)
    monkeypatch.setattr(RuntimeSettings, 'from_env', lambda: settings)
    monkeypatch.setattr(run, 'FOLDER', tmp_path/'new')
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path/'new')
    return settings


def test_new_campaign_uses_configured_timeout_before_freeze(monkeypatch, tmp_path):
    setup_configuration(monkeypatch, tmp_path)
    assert run.local_configuration().timeout_seconds == 120


@pytest.mark.parametrize('protocol,expected', [({}, 60), ({'timeout_seconds': 90}, 90)])
def test_existing_protocol_preserves_legacy_deadline(monkeypatch, tmp_path, protocol, expected):
    setup_configuration(monkeypatch, tmp_path)
    run.FOLDER.mkdir()
    (run.FOLDER/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    assert run.local_configuration().timeout_seconds == expected


def freeze_fixture(monkeypatch, tmp_path):
    settings = setup_configuration(monkeypatch, tmp_path)
    with sqlite3.connect(settings.memory_db) as conn:
        conn.execute('CREATE TABLE placeholder (id INTEGER)')
    observed = []
    store = SimpleNamespace(paper_metadata=lambda ids: [{'sha256': s} for s in (run.CHANGE, run.FLOW)],
                            paper_outline=lambda sha: [], get_chunks=lambda ids: [])
    def build(config):
        observed.append(config.timeout_seconds)
        return SimpleNamespace(search_adapter=SimpleNamespace(store=store), text_reader_protocol='offline')
    monkeypatch.setattr('paperpilot_agentic.bootstrap.build_runtime', build)
    monkeypatch.setattr(run, 'source_fingerprint', lambda root: 'offline')
    monkeypatch.setattr(run, 'acceptance_cases', lambda root: [{'id': 'one', 'question': 'offline'}])
    run.freeze(max_requests=30, max_tokens=300000)
    protocol = json.loads((run.FOLDER/'protocol.json').read_text(encoding='utf-8'))
    return protocol, observed


def test_freeze_records_timeout_for_ordinary_model(monkeypatch, tmp_path):
    protocol, observed = freeze_fixture(monkeypatch, tmp_path)
    assert observed == [120]
    assert protocol['timeout_seconds'] == 120
    assert protocol['request_timeout']['text_seconds'] == 120
    assert protocol['request_timeout']['vision_seconds'] == 120
    assert protocol['identity']['request_timeout'] == protocol['request_timeout']
    assert run.campaign_budget(protocol).snapshot()['actual_requests'] == 0


@pytest.mark.parametrize('legacy,expected', [({}, (60, 60)), ({'timeout_seconds': 90}, (90, 60))])
def test_legacy_text_and_vision_deadlines_remain_distinct(legacy, expected):
    actual = request_timeouts(legacy, configured=120)
    assert (actual['text_seconds'], actual['vision_seconds']) == expected


def test_frozen_configuration_ignores_environment_drift(monkeypatch, tmp_path):
    protocol, _ = freeze_fixture(monkeypatch, tmp_path)
    changed = replace(fixture_settings(tmp_path), timeout_seconds=600)
    monkeypatch.setattr(RuntimeSettings, 'from_env', lambda: changed)
    assert run.local_configuration().timeout_seconds == 120
    assert request_timeouts(protocol, 600)['vision_seconds'] == 120


@pytest.mark.parametrize('value', [0, -1, True, '120', float('inf'), float('nan')])
def test_invalid_new_deadline_rejected_before_freeze(value):
    with pytest.raises(ValueError, match='finite positive'):
        request_timeouts(None, value)


@pytest.mark.parametrize('mutation', ['text', 'vision', 'mirror', 'identity', 'version'])
def test_frozen_deadline_tampering_rejected_at_configuration(monkeypatch, tmp_path, mutation):
    protocol, _ = freeze_fixture(monkeypatch, tmp_path)
    unchanged_budget = run.campaign_budget(protocol)
    if mutation == 'text':
        protocol['request_timeout']['text_seconds'] = 300
    elif mutation == 'vision':
        protocol['request_timeout']['vision_seconds'] = 300
    elif mutation == 'mirror':
        protocol['timeout_seconds'] = 300
    elif mutation == 'identity':
        del protocol['identity']['request_timeout']
    else:
        protocol['request_timeout']['version'] = 'unknown'
    (run.FOLDER/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    with pytest.raises(ValueError, match='identity changed'):
        run.local_configuration()
    assert unchanged_budget.snapshot()['actual_requests'] == 0


@pytest.mark.parametrize('legacy', [False, True])
def test_execute_constructs_sdk_adapters_with_frozen_deadlines(monkeypatch, tmp_path, legacy):
    protocol, _ = freeze_fixture(monkeypatch, tmp_path)
    if legacy:
        del protocol['request_timeout']
        del protocol['identity']['request_timeout']
        # Use a separate fixture ledger with the same historical identity.
        monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path/'legacy-ledger')
        protocol['timeout_seconds'] = 90
        (run.FOLDER/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    observed = {}
    def text_sdk(settings, *, max_retries):
        observed['text'] = settings.timeout_seconds
        assert max_retries == 0
        return SimpleNamespace(model=settings.model, max_retries=0)
    def vision_sdk(**kwargs):
        observed['vision'] = kwargs['timeout_seconds']
        return SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace())))
    class ReachedRuntime(Exception):
        pass
    def runtime(settings, **kwargs):
        assert settings.timeout_seconds == observed['text']
        raise ReachedRuntime
    monkeypatch.setattr('paperpilot_agentic.providers.OpenAICompatibleLLM', text_sdk)
    monkeypatch.setattr('paperpilot_agentic.qwen_vision.QwenVisionEvidenceAdapter', vision_sdk)
    monkeypatch.setattr('evals.interview.transport.transport_preflight', lambda *args: {'reachable': True})
    monkeypatch.setattr(run, 'model_preflight', lambda *args: True)
    monkeypatch.setattr('paperpilot_agentic.bootstrap.build_runtime', runtime)
    with pytest.raises(ReachedRuntime):
        run.execute(stage='answers')
    assert observed == ({'text': 90, 'vision': 60} if legacy else {'text': 120, 'vision': 120})
    assert run.campaign_budget(protocol).snapshot()['actual_requests'] == 0


def test_timeout_change_does_not_resume_unknown_or_create_clients(monkeypatch, tmp_path):
    protocol, _ = freeze_fixture(monkeypatch, tmp_path)
    budget = run.campaign_budget(protocol)
    with budget.transaction() as conn:
        conn.execute('INSERT INTO attempts VALUES(1,?,?,?,?,?,?,?)',
            ('UNKNOWN', 45548, 0, protocol['identity']['model'], 'audit', 60, '{}'))
    before = budget.snapshot()
    def forbidden(*args, **kwargs):
        pytest.fail('UNKNOWN must stop before SDK or transport setup')
    monkeypatch.setattr(run, 'local_configuration', forbidden)
    monkeypatch.setattr('paperpilot_agentic.providers.OpenAICompatibleLLM', forbidden)
    monkeypatch.setattr('paperpilot_agentic.qwen_vision.QwenVisionEvidenceAdapter', forbidden)
    monkeypatch.setattr('evals.interview.transport.transport_preflight', forbidden)
    run.execute(stage='answers')
    assert budget.snapshot() == before
    assert before['blocked'] and before['actual_requests'] == 1
    assert before['unsettled_upper_bound'] == 45548
