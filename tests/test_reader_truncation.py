import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import DurableResearchModel, run_durable_reader
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.providers import LLMResponse


class TruncatingProvider:
    def __init__(self, truncated_calls):
        self.truncated_calls = truncated_calls
        self.calls = []

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls.append(messages)
        if len(self.calls) in self.truncated_calls:
            return LLMResponse(text='{"incomplete":', prompt_tokens=10,
                               completion_tokens=20, finish_reason='length')
        payload = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        if 'evidence' in payload:
            value = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'A source fact.',
                               'evidence_ids': [payload['evidence'][0]['evidence_id']]}],
                     'direct_answer_id': 'f1', 'limitations': []}
        else:
            value = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                                     'relevance': 'DIRECT',
                                     'span_ids': [payload['facts'][0]['allowed_span_ids'][0]]}],
                     'coverage': 'COMPLETE', 'missing_aspects': []}
        return LLMResponse(text=json.dumps(value), prompt_tokens=10,
                           completion_tokens=20, finish_reason='stop')


def run_reader(tmp_path, truncated_calls, **options):
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('What is the fact?', Budget())
    provider = TruncatingProvider(truncated_calls)
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid,
                                 stage='answer_first', reserve_class='synthesis')
    report = answer_then_audit('What is the fact?',
                              [{'evidence_id': 'source1', 'content': 'A source fact.'}],
                              model, **options)
    return report, db, rid, provider


@pytest.mark.parametrize('first_truncated', [1, 2])
def test_durable_reader_repairs_truncated_writer_or_audit_once(tmp_path, first_truncated):
    report, db, rid, provider = run_reader(tmp_path, {first_truncated})
    assert report['status'] == 'REVIEW_REQUIRED', report.get('error_type')
    assert report['answer'] == 'A source fact. [source1]'
    assert report['provider_attempts'] == len(provider.calls) == 3
    assert db.get_run(rid)['used_llm_calls'] == 3
    assert db.get_run(rid)['used_tokens'] == 90
    assert report['calls'][first_truncated - 1]['finish_reason'] == 'length'
    assert report['calls'][first_truncated - 1]['text'] == '{"incomplete":'
    repair_request = provider.calls[first_truncated]
    assert not any(m['role'] == 'assistant' for m in repair_request)
    assert 'SAME output limit' in repair_request[-1]['content']
    assert db.list_llm_calls(rid)[first_truncated]['stage'] == 'json_repair'


@pytest.mark.parametrize('error', [InterruptedError('Cancelled'), RuntimeError('Unavailable')])
def test_runtime_failures_do_not_trigger_a_format_repair(tmp_path, error):
    class Broken:
        def complete(self, *args, **kwargs):
            raise error
    report = answer_then_audit('Question?', [{'evidence_id': 'e1', 'content': 'Text.'}], Broken())
    assert report['status'] == 'FAILED'
    assert 'draft_repair' not in report
    assert report['provider_attempts'] == 0


def test_global_budget_blocks_writer_when_audit_has_no_call_allowance(tmp_path):
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('What is the fact?', Budget(max_llm_calls=1, repair_reserve_calls=0))
    provider = TruncatingProvider({1})
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid,
                                 stage='answer_first', reserve_class='synthesis')
    report = answer_then_audit('What is the fact?',
        [{'evidence_id': 'source1', 'content': 'A source fact.'}], model)
    assert report['status'] == 'FAILED' and report['error_type'] == 'LLMBudgetExhausted'
    assert len(provider.calls) == db.get_run(rid)['used_llm_calls'] == 0
    assert not report.get('answer')


def test_truncation_repair_still_obeys_global_budget_after_admission(tmp_path):
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('What is the fact?', Budget(max_llm_calls=2, repair_reserve_calls=0))
    provider = TruncatingProvider({1})
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid,
                                 stage='answer_first', reserve_class='synthesis')
    report = answer_then_audit('What is the fact?',
        [{'evidence_id': 'source1', 'content': 'A source fact.'}], model)
    assert report['status'] == 'FAILED' and report['error_type'] == 'LLMBudgetExhausted'
    # The repair plus its following audit no longer fit after the truncated
    # writer. Admission must stop before paying for that unusable repair.
    assert len(provider.calls) == db.get_run(rid)['used_llm_calls'] == 1
    assert report['draft_repair']['attempted'] is True
    assert report['repair_budget_preflights'][0]['mandatory_requests'] == 2
    assert all(call['stage'] != 'json_repair' for call in db.list_llm_calls(rid))
    assert not report.get('answer')


def test_recovered_checkpoint_reopens_without_repeating_generation(tmp_path):
    path = tmp_path / 'runtime.sqlite3'
    db = RuntimeDatabase(path)
    rid = db.create_run('What is the fact?', Budget())
    provider = TruncatingProvider({1})
    model = DurableResearchModel(PersistentLLMRuntime(db, provider), rid,
                                 stage='answer_first', reserve_class='synthesis')
    evidence = [{'evidence_id': 'source1', 'content': 'A source fact.'}]
    report = run_durable_reader(db, rid, evidence, model, protocol_id='test-recovery')
    reopened = RuntimeDatabase(path)
    assert run_durable_reader(reopened, rid, evidence, model, protocol_id='test-recovery') == report
    assert report['calls'][0]['text'] == '{"incomplete":'
    assert len(provider.calls) == 3


@pytest.mark.parametrize('truncated_calls,allow_repair,expected_calls', [
    ({1, 2}, True, 2), ({1, 3}, True, 3), ({1}, False, 1),
])
def test_truncation_never_publishes_partial_json_or_loops(
        tmp_path, truncated_calls, allow_repair, expected_calls):
    report, db, rid, provider = run_reader(tmp_path, truncated_calls, allow_repair=allow_repair)
    assert report['status'] == 'FAILED'
    assert not report.get('answer')
    assert len(provider.calls) == expected_calls
    assert db.get_run(rid)['used_llm_calls'] == expected_calls
