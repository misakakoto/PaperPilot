"""Reject a known unfinished writer/auditor pair without paid dispatch."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.interview.budget import CampaignBudget, RecordedProvider, UncertainRequest  # noqa: E402
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import DurableResearchModel
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted, PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.providers import LLMResponse


REQUESTS = [{'messages': [{'role': 'user', 'content': 'source'}], 'max_tokens': 4000},
            {'messages': [{'role': 'user', 'content': 'source'}], 'max_tokens': 5000}]


class Provider:
    model, max_retries = 'fixed', 0

    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        return LLMResponse('unused', 10, 5, finish_reason='stop')


@pytest.mark.parametrize('requests,tokens', [(1, 300000), (30, 12000)])
def test_campaign_checks_both_known_requests_without_inserting_attempts(tmp_path, requests, tokens):
    campaign = CampaignBudget(tmp_path, {'model': 'fixed'}, max_requests=requests, max_tokens=tokens)
    before = campaign.snapshot()
    with pytest.raises(LLMBudgetExhausted):
        campaign.ensure_capacity('fixed', REQUESTS)
    assert campaign.snapshot() == before


def test_preflight_does_not_reserve_or_bypass_stop(tmp_path):
    campaign = CampaignBudget(tmp_path, {'model': 'fixed'})
    campaign.ensure_capacity('fixed', REQUESTS)
    assert campaign.snapshot()['attempts'] == []
    (tmp_path/'stopped.json').write_text('{"reason":"budget_exhausted"}')
    before = campaign.snapshot()
    with pytest.raises(LLMBudgetExhausted):
        campaign.ensure_capacity('fixed', REQUESTS)
    assert campaign.snapshot() == before


def test_preflight_cannot_clear_unknown_or_change_model(tmp_path):
    campaign = CampaignBudget(tmp_path, {'model': 'fixed'})

    class Lost(Provider):
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            raise OSError('offline lost response fixture')

    with pytest.raises(OSError):
        RecordedProvider(Lost(), campaign, phase='lost').complete(REQUESTS[0]['messages'], max_tokens=4000)
    before = campaign.snapshot()
    with pytest.raises(UncertainRequest):
        campaign.ensure_capacity('fixed', REQUESTS)
    with pytest.raises(ValueError):
        campaign.ensure_capacity('other', REQUESTS)
    assert campaign.snapshot() == before


def test_runtime_countdown_is_in_both_cost_preflight_and_actual_request(tmp_path):
    class Capture(Provider):
        def ensure_capacity(self, requests):
            self.requests = requests

        def complete(self, messages, **kwargs):
            self.messages = messages
            return super().complete(messages, **kwargs)

    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Question?', Budget())
    provider = Capture()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid, stage='answer_first',
                                 reserve_class='synthesis')
    model.ensure_capacity(REQUESTS)
    assert all('[PAPERPILOT_RUNTIME_BUDGET]' in r['messages'][0]['content'] for r in provider.requests)
    model.complete(REQUESTS[0]['messages'], max_tokens=4000)
    assert provider.requests[0]['messages'] == provider.messages


def test_durable_bridge_checks_run_allowance_before_writer(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('What fact?', Budget(max_llm_calls=1, repair_reserve_calls=0))
    provider = Provider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid, stage='answer_first',
                                 reserve_class='synthesis')
    report = answer_then_audit('What fact?', [{'evidence_id': 't', 'content': 'A source fact.'}], model)
    assert report['error_type'] == 'LLMBudgetExhausted'
    assert report['provider_attempts'] == provider.calls == db.get_run(rid)['used_llm_calls'] == 0
    assert report['budget_preflight']['mandatory_requests'] == 2


def test_durable_bridge_forwards_complete_known_inputs_to_campaign(tmp_path):
    campaign = CampaignBudget(tmp_path/'campaign', {'model': 'fixed'}, max_requests=1)
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('What fact?', Budget(max_llm_calls=20))
    provider = Provider()
    wrapped = RecordedProvider(provider, campaign, phase='answer_first')
    model = DurableResearchModel(PersistentLLMRuntime(db, wrapped), rid, stage='answer_first',
                                 reserve_class='synthesis')
    report = answer_then_audit('What fact?', [{'evidence_id': 't', 'content': 'A source fact.'}], model)
    assert report['error_type'] == 'LLMBudgetExhausted'
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == campaign.snapshot()['actual_requests'] == 0


def test_preflight_uses_the_same_history_and_aliases_as_actual_writer():
    class Capture:
        def ensure_capacity(self, requests):
            self.requests = requests

        def complete(self, messages, **kwargs):
            self.messages = messages
            kwargs['before_attempt']()
            raise LLMBudgetExhausted('offline test stop')

    model = Capture()
    history = [{'role': 'initial_request', 'user_request': 'Keep the 8GB limit.'}]
    source_id = 'evidence_' + 'a' * 32
    answer_then_audit('Resolved?', [{'evidence_id': source_id, 'content': 'A source fact.'}],
                      model, original_question='Actual?', request_history=history, quantity_policy=True)
    assert model.requests[0]['messages'] == model.messages
    for request in model.requests:
        payload = json.loads(request['messages'][-1]['content'])
        assert payload['original_question'] == 'Actual?'
        assert payload['request_history'] == history
        assert source_id not in json.dumps(request['messages'])


def test_normal_budget_still_uses_writer_and_auditor():
    outputs = iter([
        {'facts': [{'id': 'f', 'text': 'A source fact.', 'evidence_ids': ['t']}], 'direct_answer_id': 'f'},
        {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                          'span_ids': ['s0'], 'reason': 'Source states it.'}],
         'coverage': 'COMPLETE', 'missing_aspects': []}])

    class Capture:
        calls = 0

        def ensure_capacity(self, requests):
            self.requests = requests

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(json.dumps(next(outputs)), finish_reason='stop')

    model = Capture()
    report = answer_then_audit('What fact?', [{'evidence_id': 't', 'content': 'A source fact.'}], model)
    assert model.calls == report['provider_attempts'] == 2
    assert report['answer'].startswith('A source fact.')
    assert report['budget_preflight']['unestimated'] == ['candidate facts', 'repair', 'presentation']


@pytest.mark.parametrize('limit,repair_reserve,expected', [(2, 0, 1), (3, 1, 3)])
def test_writer_repair_requires_the_following_audit_without_double_counting_reserve(
        tmp_path, limit, repair_reserve, expected):
    outputs = iter([
        {'facts': 'invalid'},
        {'facts': [{'id': 'f', 'text': 'A source fact.', 'evidence_ids': ['t']}], 'direct_answer_id': 'f'},
        {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                          'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': []}])

    class RepairProvider(Provider):
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            return LLMResponse(json.dumps(next(outputs)), 10, 5, finish_reason='stop')

    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('What fact?', Budget(max_llm_calls=limit, max_tokens=300000,
                        repair_reserve_calls=repair_reserve))
    provider = RepairProvider()
    bridge = DurableResearchModel(PersistentLLMRuntime(db, provider), rid,
                                  stage='answer_first', reserve_class='synthesis')
    result = answer_then_audit('What fact?', [{'evidence_id': 't', 'content': 'A source fact.'}], bridge)
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == expected
    assert result['repair_budget_preflights'][0]['mandatory_requests'] == 2
    if expected == 1:
        assert result['error_type'] == 'LLMBudgetExhausted'
        assert not result.get('answer') and 'repaired_draft' not in result
    else:
        assert result['status'] == 'REVIEW_REQUIRED' and result['answer'].startswith('A source fact.')
