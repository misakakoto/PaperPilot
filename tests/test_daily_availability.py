"""Daily admission must differ from a frozen acceptance allowance."""
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_attempts import AttemptLedger, UnknownRequestResult
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime, LLMBudgetExhausted
from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL, ModelQuotaExhausted
from paperpilot_agentic.models import LLMCallPolicy
from paperpilot_agentic.providers import OpenAICompatibleLLM
from paperpilot_agentic.models import Budget
from paperpilot_agentic.orchestrator import ResearchAgentRuntime


def test_daily_reader_default_can_admit_known_writer_audit_pair(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path / 'runtime.sqlite3', workspace=tmp_path)
    run = runtime.enqueue(question='Compare the methods.',
                          metadata={'engine': 'answer_first_text'})['run']
    assert run['budget']['max_tokens'] == 200_000
    assert run['budget']['max_llm_calls'] == 20
    assert run['metadata']['execution_profile'] == 'daily'
    class NoDispatch:
        model = 'offline'
        def complete(self, *args, **kwargs):
            pytest.fail('Admission is a zero-call check')
    bridge = PersistentLLMRuntime(runtime.db, NoDispatch())
    # Same failure pattern as the retained 157,753-token mandatory pair;
    # source bodies are held-out ASCII, not private acceptance answers.
    requests = [{'messages': [{'role': 'user', 'content': 'source ' * 10000}], 'max_tokens': output}
                for output in (4000, 5000)]
    bridge.ensure_capacity(run['run_id'], requests, reserve_class='synthesis')
    assert runtime.db.get_run(run['run_id'])['used_llm_calls'] == 0


def test_explicit_and_frozen_allowances_keep_their_original_identity(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path / 'runtime.sqlite3', workspace=tmp_path)
    for metadata, allowance in [
        ({'engine': 'answer_first_text'}, Budget(max_tokens=12_345)),
        ({'engine': 'answer_first_text', 'execution_profile': 'frozen_acceptance'}, None),
    ]:
        run = runtime.enqueue(question='A frozen question.', metadata=metadata,
                              budget=allowance)['run']
        assert run['budget']['max_tokens'] == (12_345 if allowance else 100_000)


def fixture_provider(tmp_path, send):
    settings = RuntimeSettings(provider='qwen', api_key='offline-fixture', model=PRIMARY_MODEL,
        fallback_models=(BACKUP_MODEL,), base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        thinking_enabled=False, temperature=0, max_tokens=100, timeout_seconds=1,
        workspace=tmp_path, runtime_db=tmp_path / 'runtime.sqlite3', memory_db=None,
        memory_namespace='fixture', paper_roots=(tmp_path,))
    provider = OpenAICompatibleLLM(settings, max_retries=0)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    return provider


def provider_error(code='AllocationQuota.FreeTierOnly', status=403):
    return APIStatusError('offline fixture', response=httpx.Response(status,
        request=httpx.Request('POST', 'https://example.invalid')), body={'error': {'code': code}})


def success(model, *, usage=True):
    return SimpleNamespace(model=model, id='offline-receipt',
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ready":true}'), finish_reason='stop')],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5) if usage else None)


def fixture_run(tmp_path, send, *, budget=None, profile='daily'):
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Read the evidence.', budget or Budget(),
                       metadata={'execution_profile': profile})
    return db, rid, PersistentLLMRuntime(db, fixture_provider(tmp_path, send))


POLICY = LLMCallPolicy('fixture', 100, False, True, reserve_class='repair')
MESSAGES = [{'role': 'user', 'content': 'Return JSON.'}]


def test_quota_fallback_keeps_failed_hold_and_actual_model_across_restart(tmp_path):
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        assert 'enable_thinking' not in kwargs.get('extra_body', {})
        if kwargs['model'] == PRIMARY_MODEL:
            raise provider_error()
        return success(kwargs['model'])
    db, rid, runtime = fixture_run(tmp_path, send)
    assert runtime.complete(rid, MESSAGES, policy=POLICY).model == BACKUP_MODEL
    snapshot = runtime.budget_snapshot(rid)
    assert snapshot['used_llm_calls'] == 2 and snapshot['used_tokens'] == 15
    assert snapshot['reserved_unknown_tokens'] > 0 and snapshot['unknown_results'] == 0
    assert [c['model'] for c in db.list_llm_calls(rid)] == [PRIMARY_MODEL, BACKUP_MODEL]
    restored = PersistentLLMRuntime(RuntimeDatabase(db.path), fixture_provider(tmp_path, send))
    assert restored.budget_snapshot(rid) == snapshot
    restored.complete(rid, MESSAGES, policy=POLICY)
    assert sent == [PRIMARY_MODEL, BACKUP_MODEL, BACKUP_MODEL]
    receipts = AttemptLedger(db).snapshot(rid)['attempts']
    assert len(receipts) == 3 and all(r['policy_sha256'] for r in receipts)
    assert all(r['request_bytes'] > 0 and r['request_sha256'] for r in receipts)
    assert receipts[1]['cached_tokens_known'] == 0


@pytest.mark.parametrize('profile,expected', [('daily', 2), ('frozen_acceptance', 1)])
def test_all_quota_exhausted_and_frozen_model_never_blindly_retry(tmp_path, profile, expected):
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        raise provider_error()
    db, rid, runtime = fixture_run(tmp_path, send, profile=profile)
    with pytest.raises(APIStatusError):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert len(sent) == db.get_run(rid)['used_llm_calls'] == expected
    with pytest.raises(ModelQuotaExhausted):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert len(sent) == expected


@pytest.mark.parametrize('code,status', [('access_denied', 403), ('invalid_api_key', 401),
                                       ('arrearage', 403), ('insufficient_quota', 429)])
def test_auth_billing_and_other_quota_codes_do_not_switch_models(tmp_path, code, status):
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        raise provider_error(code, status)
    db, rid, runtime = fixture_run(tmp_path, send)
    with pytest.raises(APIStatusError):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert sent == [PRIMARY_MODEL] and db.get_run(rid)['used_llm_calls'] == 1


def test_unknown_stops_all_further_dispatch_even_after_restart(tmp_path):
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        raise APIConnectionError(request=httpx.Request('POST', 'https://example.invalid'))
    db, rid, runtime = fixture_run(tmp_path, send)
    with pytest.raises(APIConnectionError):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    restored = PersistentLLMRuntime(RuntimeDatabase(db.path), fixture_provider(tmp_path, send))
    with pytest.raises(UnknownRequestResult):
        restored.complete(rid, MESSAGES, policy=POLICY)
    assert sent == [PRIMARY_MODEL] and restored.budget_snapshot(rid)['unknown_results'] == 1


def test_fallback_cannot_reset_or_bypass_remaining_call_budget(tmp_path):
    sent = []
    def send(**kwargs):
        sent.append(kwargs['model'])
        raise provider_error()
    db, rid, runtime = fixture_run(tmp_path, send, budget=Budget(max_llm_calls=1))
    with pytest.raises(LLMBudgetExhausted):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert sent == [PRIMARY_MODEL] and db.get_run(rid)['used_llm_calls'] == 1


def test_missing_returned_usage_keeps_conservative_hold(tmp_path):
    db, rid, runtime = fixture_run(tmp_path, lambda **kw: success(kw['model'], usage=False))
    runtime.complete(rid, MESSAGES, policy=POLICY)
    snapshot = runtime.budget_snapshot(rid)
    assert snapshot['used_tokens'] == 0 and snapshot['reserved_unknown_tokens'] > 0
    assert snapshot['unknown_usage_attempts'] == 1 and snapshot['unknown_results'] == 0


def test_policy_change_is_rejected_before_dispatch(tmp_path):
    sent = []
    db, rid, runtime = fixture_run(tmp_path, lambda **kw: sent.append(kw) or success(kw['model']))
    runtime.complete(rid, MESSAGES, policy=POLICY)
    runtime.model.fallback_models = ()
    with pytest.raises(ValueError, match='Model policy changed'):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert len(sent) == 1


@pytest.mark.parametrize('profile', ['daily', 'frozen_acceptance'])
def test_confirmed_quota_expiry_stops_before_any_dispatch(tmp_path, monkeypatch, profile):
    from datetime import date
    from paperpilot_agentic import model_policy
    class ExpiredDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 28)
    monkeypatch.setattr(model_policy, 'date', ExpiredDate)
    sent = []
    db, rid, runtime = fixture_run(tmp_path, lambda **kw: sent.append(kw), profile=profile)
    with pytest.raises(ModelQuotaExhausted):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert sent == [] and db.get_run(rid)['used_llm_calls'] == 0


def test_daily_window_cannot_exceed_confirmed_model_serving_limit(tmp_path):
    db, rid, runtime = fixture_run(tmp_path, lambda **kw: pytest.fail('No dispatch expected'))
    runtime.model.context_window_tokens = 262144
    assert runtime.model.for_run(db, rid).context_window_tokens == 131072


def test_binding_cannot_unwrap_the_frozen_campaign_accounting_decorator(tmp_path):
    from evals.interview.budget import CampaignBudget, RecordedProvider
    sent = []
    def send(**kwargs):
        sent.append(kwargs)
        raw = success(kwargs['model'])
        raw.model_dump = lambda **kw: {'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                                      'choices': [{'message': {'content': '{"ready":true}'}}]}
        return raw
    db, rid, runtime = fixture_run(tmp_path, send, profile='frozen_acceptance')
    runtime.model.fallback_models = ()
    campaign = CampaignBudget(tmp_path / 'receipts', {'model': PRIMARY_MODEL}, max_requests=1)
    runtime.model = RecordedProvider(runtime.model, campaign, phase='fixture')
    runtime.complete(rid, MESSAGES, policy=POLICY)
    with pytest.raises(LLMBudgetExhausted):
        runtime.ensure_capacity(rid, [{'messages': MESSAGES, 'max_tokens': 100}])
    with pytest.raises(LLMBudgetExhausted):
        runtime.complete(rid, MESSAGES, policy=POLICY)
    assert len(sent) == campaign.snapshot()['actual_requests'] == db.get_run(rid)['used_llm_calls'] == 1
