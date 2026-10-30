"""Literal table questions need a reading contract, without losing safeguards."""
import json

import pytest

from paperpilot_agentic.answer_pipeline import (
    AnswerOptions, initialize, prepared_messages, writer_messages,
)
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.literal_table_writer import VERSION
from test_table_claim_rows import table


@pytest.mark.parametrize('question', [
    'Table IV 的 † 记号是什么意思？是否表示实测零分？',
    'Explain the missing-value marker in Table IV and its legend.',
])
def test_marker_reading_profile_preserves_all_original_evidence(question):
    evidence = [table(), {'evidence_id': 'body', 'source_uri': 'paper://p/body',
        'content': 'Table IV legend: † means an unavailable observation.'}]
    context = initialize(question, evidence, object(), AnswerOptions(), None)
    assert context.report['writer_profile'] == VERSION
    messages = writer_messages(context)
    payload = json.loads(messages[-1]['content'])
    assert payload['evidence'] == context.rows
    assert payload['question'] == question
    if '记号' in question:
        assert payload['prose_language'] == 'Chinese'
    assert 'SOURCE_FACT' in messages[0]['content'] and 'INFERENCE' in messages[0]['content']
    assert 'TABLE_FACT' in messages[0]['content']
    prepared = prepared_messages(context, messages)
    assert json.loads(prepared[-1]['content'])['read_packet_sha256'] == context.report['read_packet']['packet_sha256']
    assert not context.report.get('all_facts_supported')


@pytest.mark.parametrize('question,options', [
    ('Compare Table IV F1 differences and explain the marker.', AnswerOptions()),
    ('Table IV 的最高 F1 与图例符号分别是什么？', AnswerOptions()),
    ('Explain the mechanism in Table IV and the marker.', AnswerOptions()),
    ('What does the symbol mean in equation (4)?', AnswerOptions()),
    ('Explain the legend in Table IV.', AnswerOptions(draft_contract='layered-v1')),
    ('Explain the legend in Table IV.', AnswerOptions(request_history=[
        {'role': 'initial_request', 'user_request': 'Also compare the two methods.'}])),
])
def test_other_tasks_keep_the_existing_writer(question, options):
    context = initialize(question, [table()], object(), options, None)
    assert 'writer_profile' not in context.report


def test_frozen_draft_does_not_gain_a_new_generation_profile():
    frozen = {'facts': [{'id': 'f', 'text': 'A source statement.', 'evidence_ids': ['t']}],
              'direct_answer_id': 'f', 'limitations': []}
    context = initialize('Explain the legend in Table IV.', [table()], object(),
                         AnswerOptions(frozen_draft=frozen), None)
    assert 'writer_profile' not in context.report


def test_compact_writer_still_requires_independent_support():
    class Model:
        calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                value = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                    'text': 'The marker represents a measured zero.',
                    'evidence_ids': [payload['evidence'][0]['evidence_id']]}],
                    'direct_answer_id': 'f1', 'limitations': []}
            else:
                value = {'assessments': [{'fact_id': 'f1', 'verdict': 'UNSUPPORTED',
                    'relevance': 'DIRECT', 'reason': 'No legend supports the claim.'}],
                    'coverage': 'PARTIAL', 'missing_aspects': ['Marker meaning not established.']}
            return LLMResponse(json.dumps(value), finish_reason='stop')

    model = Model()
    report = answer_then_audit('What does the marker in Table IV mean?', [table()], model)
    assert report['writer_profile'] == VERSION
    assert model.calls == 2
    assert not report.get('answer')
    assert report['audit']['assessments'][0]['verdict'] == 'UNSUPPORTED'
