from dataclasses import dataclass
import pytest

from evals.interview.budget import CampaignBudget, RecordedProvider, UncertainRequest


IDENTITY = {'campaign': 'authorized-unlimited', 'model': 'fixed',
            'usage_limit_mode': 'unlimited', 'source_sha256': 'fixture'}


@dataclass(frozen=True)
class Response:
    text: str = '{}'
    finish_reason: str = 'stop'
    prompt_tokens: int = 400000
    completion_tokens: int = 100
    usage_known: bool = True


class Provider:
    model = 'fixed'

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def complete(self, messages, *, before_attempt=None, **kwargs):
        if before_attempt:
            before_attempt()
        self.calls += 1
        if self.fail:
            raise OSError('Lost result')
        return Response()


def test_unlimited_campaign_still_counts_beyond_old_token_cap_and_reopens(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_requests=None, max_tokens=None)
    p = Provider()
    provider = RecordedProvider(p, budget, phase='answers')
    for _ in range(3):
        provider.complete([{'role': 'user', 'content': 'Explain.'}], max_tokens=32)
    snapshot = CampaignBudget(tmp_path, IDENTITY, max_requests=None, max_tokens=None).snapshot()
    assert snapshot['actual_requests'] == p.calls == 3
    assert snapshot['reported_tokens'] == 1200300
    assert snapshot['max_requests'] is snapshot['max_tokens'] is None
    assert not snapshot['blocked']


def test_unlimited_does_not_bypass_unknown_or_change_old_bounded_ledger(tmp_path):
    budget = CampaignBudget(tmp_path, IDENTITY, max_requests=None, max_tokens=None)
    with pytest.raises(OSError):
        RecordedProvider(Provider(fail=True), budget, phase='answers').complete(
            [{'role': 'user', 'content': 'Explain.'}], max_tokens=32)
    with pytest.raises(UncertainRequest):
        RecordedProvider(Provider(), budget, phase='retry').complete(
            [{'role': 'user', 'content': 'Explain.'}], max_tokens=32)
    old = tmp_path / 'old'
    CampaignBudget(old, {'model': 'fixed'}, max_requests=2, max_tokens=3000)
    with pytest.raises(ValueError):
        CampaignBudget(old, IDENTITY, max_requests=None, max_tokens=None)


def test_unlimited_run_preflight_dispatch_accounting_and_budget_roundtrip(tmp_path):
    from evals.interview.run import answer_budget
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
    from paperpilot_agentic.models import Budget, LLMCallPolicy
    budget = answer_budget({'identity': IDENTITY, 'max_requests': None, 'max_tokens': None})
    assert Budget.from_dict(budget.to_dict()).unlimited_llm_usage
    # Artificial small finite fields must not become hidden ceilings.
    budget.max_llm_calls = 1
    budget.max_tokens = 10
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Explain.', budget)
    campaign = CampaignBudget(tmp_path / 'receipts', IDENTITY, max_requests=None, max_tokens=None)
    runtime = PersistentLLMRuntime(db, RecordedProvider(Provider(), campaign, phase='answers'))
    request = {'messages': [{'role': 'user', 'content': 'Explain.'}], 'max_tokens': 32,
               'stage': 'answer_first', 'reserve_class': 'synthesis'}
    runtime.ensure_capacity(rid, [request, request], reserve_class='synthesis')
    for _ in range(3):
        runtime.complete(rid, request['messages'], policy=LLMCallPolicy(
            stage='answer_first', max_output_tokens=32, thinking_enabled=False, reserve_class='synthesis'))
    snapshot = runtime.budget_snapshot(rid)
    assert snapshot['used_tokens'] == 1200300
    assert snapshot['used_llm_calls'] == campaign.snapshot()['actual_requests'] == 3
    assert snapshot['remaining_tokens'] is snapshot['remaining_llm_calls'] is None
    assert snapshot['max_tokens'] is snapshot['max_llm_calls'] is None
    assert not snapshot['warning']
