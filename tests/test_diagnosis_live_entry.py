"""The real entry must satisfy the shared ledger before constructing a provider."""
import json
from types import SimpleNamespace

import pytest

from evals.result_diagnosis import live
from paperpilot_agentic.research_harness import research_model_identity


def test_live_entry_accepts_authorized_unlimited_policy_without_a_model_request(tmp_path, monkeypatch):
    monkeypatch.setattr(live.RuntimeSettings, 'from_env', lambda **kwargs: SimpleNamespace(
        model='synthetic-model', provider='qwen', base_url='https://example.invalid/v1'))
    monkeypatch.setattr(live, 'source_fingerprint', lambda root: 'synthetic-source')
    original = live.CampaignBudget
    checked = []
    class StopBeforePreflight(Exception):
        pass
    def stop_after_real_ledger(folder, identity, **limits):
        ledger = original(folder, identity, **limits)
        checked.append(ledger.snapshot())
        raise StopBeforePreflight
    monkeypatch.setattr(live, 'CampaignBudget', stop_after_real_ledger)
    with pytest.raises(StopBeforePreflight):
        live.run(tmp_path, model='synthetic-model')
    assert checked[0]['actual_requests'] == 0
    protocol = json.loads((tmp_path / 'protocol.json').read_bytes())
    assert protocol['usage_limit_mode'] == 'unlimited'
    assert protocol['max_decision_turns'] == 12


def test_recording_preserves_actual_model_identity_without_exposing_transport(tmp_path):
    raw = SimpleNamespace(model='synthetic-model', provider='qwen',
        base_url='https://example.invalid/v1', temperature=0.1,
        complete=lambda *args, **kwargs: pytest.fail('No model request expected'))
    ledger = live.CampaignBudget(tmp_path, {'model': raw.model})
    wrapped = live.RecordedProvider(raw, ledger, phase='synthetic')
    def identity(model):
        return research_model_identity(SimpleNamespace(llm_runtime=SimpleNamespace(model=model)))
    assert wrapped.provider == 'qwen'
    assert identity(raw) == identity(wrapped)
    assert ledger.snapshot()['actual_requests'] == 0
