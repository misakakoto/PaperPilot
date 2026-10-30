from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from evals.interview.budget import CampaignBudget, RecordedProvider
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
from paperpilot_agentic.providers import LLMResponse

IDENTITY = {'model': PRIMARY_MODEL, 'provider': 'qwen', 'source_sha256': 'offline-fixed',
            'capability_preflight_models': [PRIMARY_MODEL, BACKUP_MODEL]}
MESSAGES = [{'role': 'user', 'content': 'Return JSON.'}]


def test_two_capability_models_share_one_ceiling_but_answers_stay_fixed(tmp_path):
    campaign = CampaignBudget(tmp_path, IDENTITY, max_requests=4, max_tokens=300000)
    sent = []
    class Adapter:
        max_retries = 0
        def __init__(self, model):
            self.model = model
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            sent.append(self.model)
            return LLMResponse('{"ready":true}', 10, 5, model=self.model, finish_reason='stop')
    for model in (PRIMARY_MODEL, BACKUP_MODEL):
        for kind in ('text', 'image'):
            RecordedProvider(Adapter(model), campaign, phase='capability:' + kind).complete(MESSAGES, max_tokens=128)
    assert len(sent) == CampaignBudget(tmp_path, IDENTITY, max_requests=4, max_tokens=300000).snapshot()['actual_requests'] == 4
    assert campaign.snapshot()['reported_tokens'] == 60
    with pytest.raises(ValueError, match='silently switch'):
        RecordedProvider(Adapter(BACKUP_MODEL), campaign, phase='single_question').complete(MESSAGES, max_tokens=128)
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(Adapter(PRIMARY_MODEL), campaign, phase='single_question').complete(MESSAGES, max_tokens=128)
    assert len(sent) == 4


def settings(tmp_path):
    return RuntimeSettings(provider='qwen', api_key='offline', model=PRIMARY_MODEL,
        fallback_models=(), base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        thinking_enabled=False, temperature=0, max_tokens=128, timeout_seconds=1,
        workspace=tmp_path, runtime_db=tmp_path/'runtime.sqlite3', memory_db=None,
        memory_namespace='offline', paper_roots=())


def test_four_preflights_are_reused_without_a_fifth_ready_call(tmp_path):
    from evals.interview.capabilities import capability_preflight
    campaign = CampaignBudget(tmp_path/'receipts', IDENTITY, max_requests=30, max_tokens=300000)
    sent = []
    class Adapter:
        max_retries = 0
        def __init__(self, config):
            self.model = config.model
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            sent.append((self.model, messages))
            image = isinstance(messages[-1]['content'], list)
            return LLMResponse(json.dumps({'color': 'red'} if image else {'ready': True}),
                               20, 5, model=self.model, finish_reason='stop')
    config = settings(tmp_path)
    assert capability_preflight(config, campaign, tmp_path, factory=Adapter)
    assert capability_preflight(replace(config), campaign, tmp_path, factory=Adapter)
    assert len(sent) == campaign.snapshot()['actual_requests'] == 4
    assert json.loads((tmp_path/'preflight-result.json').read_text())['ready']
    # Simulate interruption after durable SDK receipts but before projections.
    (tmp_path/'preflight-result.json').unlink()
    for path in tmp_path.glob('capability-*.json'):
        path.unlink()
    assert capability_preflight(config, campaign, tmp_path, factory=Adapter)
    assert len(sent) == campaign.snapshot()['actual_requests'] == 4


def test_bad_preflight_stops_and_cannot_be_replayed(tmp_path):
    from evals.interview.capabilities import capability_preflight
    campaign = CampaignBudget(tmp_path/'receipts', IDENTITY, max_requests=30, max_tokens=300000)
    sent = []
    def factory(config):
        def complete(messages, *, before_attempt, **kwargs):
            before_attempt()
            sent.append(config.model)
            return LLMResponse('{"wrong":true}', 20, 5, model=config.model, finish_reason='stop')
        return SimpleNamespace(model=config.model, max_retries=0, complete=complete)
    assert not capability_preflight(settings(tmp_path), campaign, tmp_path, factory=factory)
    assert not capability_preflight(settings(tmp_path), campaign, tmp_path, factory=factory)
    assert sent == [PRIMARY_MODEL] and campaign.snapshot()['blocked']


def test_code_revision_reuses_bound_four_probes_without_changing_provider_configuration(tmp_path):
    import hashlib
    from evals.interview.capabilities import inherit_preflight
    config = settings(tmp_path)
    rows = [{'model': model, 'kind': kind, 'ready': True, 'finish_reason': 'stop',
             'parsed_json': {'ready': True} if kind == 'text' else {'color': 'red'}}
            for model in (PRIMARY_MODEL, BACKUP_MODEL) for kind in ('text', 'image')]
    saved = {'ready': True, 'campaign_identity': IDENTITY, 'capability_results': rows,
             'actual_requests': 4}
    path = tmp_path/'preflight-result.json'
    path.write_text(json.dumps(saved), encoding='utf-8')
    parent = {'identity': IDENTITY, 'temperature': config.temperature,
              'base_url_sha256': hashlib.sha256(config.base_url.encode()).hexdigest(),
              'thinking_enabled': False}
    identity = {**IDENTITY, 'source_sha256': 'repaired', 'revision': '0001'}
    proof = inherit_preflight(parent, identity, tmp_path, config)
    assert proof['ready'] and proof['campaign_identity'] == identity
    assert proof['reused_preflight']['result_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text()) == saved
    with pytest.raises(ValueError, match='configuration'):
        inherit_preflight(parent, identity, tmp_path, replace(config, temperature=0.9))
    saved['capability_results'][1]['ready'] = False
    path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='passed'):
        inherit_preflight(parent, identity, tmp_path, config)
