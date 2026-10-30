"""Actual dispatch seam and reopened ledger, not a pricing/quality benchmark."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.interview.budget import CampaignBudget, RecordedProvider, RecordedCompletions, UncertainRequest
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.providers import LLMResponse

IDENTITY = {'model': 'fixed', 'source_sha': 'frozen'}
MESSAGES = [{'role': 'user', 'content': 'question'}]


class Provider:
    model, max_retries = 'fixed', 0
    def __init__(self, response=None):
        self.calls = 0
        self.response = response or LLMResponse('answer', 100, 50, finish_reason='stop')
    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_shared_text_and_vision_ceiling_survives_new_instances(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_requests=2)
    provider = Provider()
    RecordedProvider(provider, budget, phase='preflight').complete(MESSAGES, max_tokens=100)
    raw = SimpleNamespace(model_dump=lambda **kw: {'usage': {'prompt_tokens': 20, 'completion_tokens': 10}})
    sdk = SimpleNamespace(create=lambda **kw: raw)
    reopened = CampaignBudget(tmp_path, IDENTITY, max_requests=2)
    RecordedCompletions(sdk, reopened, phase='vision').create(model='fixed', messages=MESSAGES, max_tokens=100)
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(provider, reopened, phase='audit').complete(MESSAGES, max_tokens=100)
    assert provider.calls == 1
    assert reopened.snapshot()['actual_requests'] == 2
    assert reopened.snapshot()['reported_tokens'] == 180


def test_insufficient_input_output_reservation_dispatches_nothing(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_tokens=2100)
    provider = Provider()
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(provider, budget, phase='large').complete(MESSAGES, max_tokens=1000)
    assert provider.calls == budget.snapshot()['actual_requests'] == 0


def test_local_guard_does_not_spend_global_allowance(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY)
    provider = Provider()
    def stop():
        raise LLMBudgetExhausted('local limit')
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(provider, budget, phase='guard').complete(MESSAGES, max_tokens=100, before_attempt=stop)
    assert provider.calls == budget.snapshot()['actual_requests'] == 0
    assert budget.snapshot()['attempts'][0]['state'] == 'NOT_SENT'


def test_unknown_request_is_retained_and_never_replayed(tmp_path):
    provider = Provider(OSError('Do not log credentials or provider body'))
    budget = CampaignBudget(tmp_path, IDENTITY)
    with pytest.raises(OSError):
        RecordedProvider(provider, budget, phase='lost').complete(MESSAGES, max_tokens=100)
    reopened = CampaignBudget(tmp_path, IDENTITY)
    with pytest.raises(UncertainRequest):
        RecordedProvider(Provider(), reopened, phase='retry').complete(MESSAGES, max_tokens=100)
    assert reopened.snapshot()['actual_requests'] == 1
    assert reopened.snapshot()['unsettled_upper_bound'] > 0
    assert 'Do not log' not in (tmp_path/'failure-001.json').read_text()


def test_authorized_revision_carries_unknown_cost_and_global_request_ceiling(tmp_path):
    old = CampaignBudget(tmp_path, IDENTITY, max_requests=2, max_tokens=10000)
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('lost')), old, phase='old').complete(MESSAGES, max_tokens=100)
    failure = (tmp_path/'failure-001.json').read_bytes()
    row = old.snapshot()['attempts'][0]
    identity = {**IDENTITY, 'source_sha': 'fixed-code', 'revision': '0001'}
    old.authorize_continuation(identity, unknown_bounds={1: 7000},
        evidence={'human_instruction': 'continue', 'basis': 'documented provider hard limit'})
    resumed = CampaignBudget(tmp_path, identity, max_requests=2, max_tokens=10000)
    assert old.snapshot()['blocked'] and not resumed.snapshot()['blocked']
    assert resumed.snapshot()['attempts'][0] == row
    assert resumed.snapshot()['unsettled_upper_bound'] == 7000
    provider = Provider()
    RecordedProvider(provider, resumed, phase='new-protocol').complete(MESSAGES, max_tokens=100)
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(provider, resumed, phase='over-limit').complete(MESSAGES, max_tokens=100)
    assert provider.calls == 1
    assert resumed.snapshot()['actual_requests'] == 2
    assert resumed.snapshot()['reported_tokens'] == 150
    assert (tmp_path/'failure-001.json').read_bytes() == failure


def test_carried_unknown_tokens_cannot_be_spent_by_the_new_protocol(tmp_path):
    old = CampaignBudget(tmp_path, IDENTITY, max_tokens=10000)
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('lost')), old, phase='old').complete(MESSAGES, max_tokens=100)
    identity = {**IDENTITY, 'revision': '0001'}
    old.authorize_continuation(identity, unknown_bounds={1: 9000}, evidence={'basis': 'reviewed hard limit'})
    resumed = CampaignBudget(tmp_path, identity, max_tokens=10000)
    provider = Provider()
    with pytest.raises(LLMBudgetExhausted):
        RecordedProvider(provider, resumed, phase='new').complete(MESSAGES, max_tokens=100)
    assert provider.calls == 0 and resumed.snapshot()['actual_requests'] == 1


def test_new_unknown_after_authorization_still_blocks_without_automatic_retry(tmp_path):
    old = CampaignBudget(tmp_path, IDENTITY)
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('lost')), old, phase='old').complete(MESSAGES, max_tokens=100)
    identity = {**IDENTITY, 'revision': '0001'}
    with pytest.raises(ValueError):
        old.authorize_continuation(identity, unknown_bounds={1: 0}, evidence={'basis': 'guess'})
    with pytest.raises(ValueError):
        old.authorize_continuation(identity, unknown_bounds={}, evidence={'basis': 'missing'})
    old.authorize_continuation(identity, unknown_bounds={1: 5000}, evidence={'basis': 'hard limit'})
    resumed = CampaignBudget(tmp_path, identity)
    with pytest.raises(OSError):
        RecordedProvider(Provider(OSError('lost again')), resumed, phase='new').complete(MESSAGES, max_tokens=100)
    assert resumed.snapshot()['blocked']
    with pytest.raises(UncertainRequest):
        RecordedProvider(Provider(), resumed, phase='retry').complete(MESSAGES, max_tokens=100)
    assert resumed.snapshot()['actual_requests'] == 2


def test_missing_usage_is_unknown_not_free(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY)
    sdk = SimpleNamespace(create=lambda **kw: SimpleNamespace(model_dump=lambda **kw: {'usage': None}))
    with pytest.raises(UncertainRequest):
        RecordedCompletions(sdk, budget, phase='vision').create(model='fixed', messages=MESSAGES, max_tokens=100)
    assert budget.snapshot()['blocked']


def test_text_sdk_missing_usage_is_not_normalized_into_free_response(tmp_path):
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    settings = SimpleNamespace(api_key='test', provider='qwen', model='fixed',
        thinking_enabled=False, temperature=0, max_tokens=100, timeout_seconds=1,
        base_url='http://localhost')
    provider = OpenAICompatibleLLM(settings, max_retries=0)
    raw = SimpleNamespace(usage=None, choices=[SimpleNamespace(
        message=SimpleNamespace(content='answer'), finish_reason='stop')],
        model_dump=lambda **kw: {'usage': None, 'choices': [{'message': {'content': 'answer'}}]})
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: raw)))
    budget = CampaignBudget(tmp_path, IDENTITY)
    with pytest.raises(UncertainRequest):
        RecordedProvider(provider, budget, phase='text').complete(MESSAGES, max_tokens=100)
    assert budget.snapshot()['actual_requests'] == 1 and budget.snapshot()['blocked']
    assert 'choices' in (tmp_path/'response-001.json').read_text()


def test_identity_and_budget_cannot_be_changed(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY)
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, {**IDENTITY, 'source_sha': 'changed'})
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, IDENTITY, max_requests=59)
    with pytest.raises(ValueError):
        budget.dispatch(model='different', phase='probe', request={'messages': MESSAGES, 'max_tokens': 100},
                        operation=lambda: None)
    assert budget.snapshot()['actual_requests'] == 0


def test_unsettled_dispatch_blocks_a_second_process_connection(tmp_path):
    first = CampaignBudget(tmp_path, IDENTITY)
    first._reserve('fixed', 'interrupted', {'messages': MESSAGES, 'max_tokens': 100})
    with pytest.raises(UncertainRequest):
        CampaignBudget(tmp_path, IDENTITY)._reserve('fixed', 'second', {'messages': MESSAGES, 'max_tokens': 100})


def test_global_guard_does_not_create_a_phantom_paid_runtime_record(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.models import Budget, LLMCallPolicy
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Question', Budget())
    campaign = CampaignBudget(tmp_path/'receipts', IDENTITY, max_requests=1)
    raw = Provider()
    runtime = PersistentLLMRuntime(db, RecordedProvider(raw, campaign, phase='answer'))
    policy = LLMCallPolicy(stage='answer_audit', max_output_tokens=100, thinking_enabled=False)
    runtime.complete(rid, MESSAGES, policy=policy)
    with pytest.raises(LLMBudgetExhausted):
        runtime.complete(rid, MESSAGES, policy=policy)
    assert raw.calls == campaign.snapshot()['actual_requests'] == db.get_run(rid)['used_llm_calls'] == 1
    assert len(db.list_llm_calls(rid)) == 1
