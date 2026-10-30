"""Exercise the acceptance SDK seam that prematurely stopped the real run."""
import json
from types import SimpleNamespace

import pytest

from evals.interview.budget import CampaignBudget, RecordedCompletions, UncertainRequest
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.vision_dispatch import vision_input_upper_bound


MODEL = 'qwen3-vl-30b-a3b-instruct'
POLICY = 'qwen3-vl-still-image-v1'
IDENTITY = {'model': MODEL, 'provider': 'qwen', 'request_cost_policy': POLICY}


def request():
    # Same encoded length as the retained FlowMamba crop; no real API/image needed.
    return {'model': MODEL, 'max_tokens': 6000, 'messages': [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Read conditions, spans and numeric cells.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'A' * 209116}}]}]}


def test_acceptance_uses_daily_image_bound_at_preflight_and_sdk_dispatch(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_tokens=143313)
    kwargs = request()
    sent = []
    raw = SimpleNamespace(model_dump=lambda **kw: {'usage': {'prompt_tokens': 500, 'completion_tokens': 50}})
    sdk = SimpleNamespace(create=lambda **kw: sent.append(kw) or raw)
    budget.ensure_capacity(MODEL, [kwargs])
    RecordedCompletions(sdk, budget, phase='vision').create(**kwargs)
    snap = CampaignBudget(tmp_path, IDENTITY, max_tokens=143313).snapshot()
    assert sent == [kwargs]
    assert snap['actual_requests'] == 1 and snap['reported_tokens'] == 550
    assert snap['attempts'][0]['reservation'] == vision_input_upper_bound(kwargs['messages']) + 6000


def test_legacy_protocol_keeps_original_reservation_and_stop(tmp_path):
    identity = {'model': MODEL, 'provider': 'qwen'}
    budget = CampaignBudget(tmp_path, identity, max_tokens=143313)
    with pytest.raises(LLMBudgetExhausted):
        RecordedCompletions(SimpleNamespace(create=lambda **kw: pytest.fail('must not send')),
                            budget, phase='vision').create(**request())
    assert budget.snapshot()['actual_requests'] == 0
    assert budget.snapshot()['blocked']


def test_policy_change_cannot_reopen_ledger_or_release_unknown(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_tokens=143313)
    def interrupted(**kw):
        raise OSError('interrupted')
    sdk = RecordedCompletions(SimpleNamespace(create=interrupted), budget, phase='vision')
    with pytest.raises(OSError):
        sdk.create(**request())
    snapshot = budget.snapshot()
    assert snapshot['unsettled_upper_bound'] == vision_input_upper_bound(request()['messages']) + 6000
    with pytest.raises(UncertainRequest):
        CampaignBudget(tmp_path, IDENTITY, max_tokens=143313).ensure_capacity(MODEL, [request()])
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, {**IDENTITY, 'request_cost_policy': 'legacy-utf8-v1'}, max_tokens=143313)
    saved = json.loads((tmp_path / 'request-001.json').read_text())
    assert saved['protocol_identity'] == IDENTITY


def test_real_insufficient_balance_records_estimate_without_dispatch(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_tokens=10000)
    with pytest.raises(LLMBudgetExhausted):
        RecordedCompletions(SimpleNamespace(create=lambda **kw: pytest.fail('must not send')),
                            budget, phase='vision').create(**request())
    stopped = json.loads((tmp_path / 'stopped.json').read_text())
    assert stopped['request_cost_policy'] == POLICY
    assert stopped['available_tokens'] == 10000
    assert stopped['required_upper_bound'] == vision_input_upper_bound(request()['messages']) + 6000
    assert budget.snapshot()['actual_requests'] == 0


@pytest.mark.parametrize('identity', [
    {**IDENTITY, 'provider': 'other'}, {**IDENTITY, 'model': 'unverified-model'},
    {**IDENTITY, 'request_cost_policy': 'unverified-policy'}])
def test_unknown_provider_model_or_policy_cannot_use_image_discount(tmp_path, identity):
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, identity)


def test_revision_cannot_reprice_historical_unknown_or_settled_calls(tmp_path):
    legacy = {key: value for key, value in IDENTITY.items() if key != 'request_cost_policy'}
    budget = CampaignBudget(tmp_path, legacy)
    with pytest.raises(ValueError, match='retains the frozen'):
        budget.authorize_settled_revision(IDENTITY, evidence={'instruction': 'continue'})
    with pytest.raises(ValueError, match='retains the frozen'):
        budget.authorize_continuation(IDENTITY, unknown_bounds={1: 100}, evidence={'instruction': 'continue'})
    assert budget.snapshot()['actual_requests'] == 0
