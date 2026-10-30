"""A provider's stop receipt does not certify a complete JSON document."""
import json

import pytest

from paperpilot_agentic.answer_pipeline import repair_messages
from paperpilot_agentic.providers import LLMResponse
from test_reader_truncation import TruncatingProvider, run_reader


@pytest.mark.parametrize('fragment', [
    '{"assessments":[],"source_inventory_navigation_only":[{"paper_sha256":"abc',
    '{"assessments":[],"source_inventory_navigation_only":[{"paper_sha256":',
    '{broken',
])
def test_invalid_json_regenerates_without_echoing_the_damaged_document(fragment):
    original = [{'role': 'system', 'content': 'Audit only the supplied facts.'},
                {'role': 'user', 'content': '{"facts":["owned fact"],"sources":["original"]}'}]
    with pytest.raises(json.JSONDecodeError) as failure:
        json.loads(fragment)
    retry = repair_messages(original, failure.value, 'Keep original citation ownership.', fragment)
    assert retry[:len(original)] == original
    assert not any(message['role'] == 'assistant' for message in retry)
    assert fragment not in retry[-1]['content']
    assert 'SAME output limit' in retry[-1]['content']
    assert 'Keep original citation ownership.' in retry[-1]['content']


def test_parsed_but_invalid_contract_keeps_targeted_feedback():
    previous = '{"assessments":[{"span_ids":["wrong-source"]}]}'
    retry = repair_messages([], ValueError('Invalid citation'), 'Recheck ownership.', previous)
    assert retry == [{'role': 'assistant', 'content': previous},
                     {'role': 'user', 'content': 'Recheck ownership.'}]


@pytest.mark.parametrize('first_broken', [1, 2])
def test_stop_with_malformed_json_uses_one_repair_and_preserves_receipt(tmp_path, monkeypatch, first_broken):
    import test_reader_truncation as reader_fixture
    class StoppedIncomplete(TruncatingProvider):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            if response.finish_reason == 'length':
                return LLMResponse(response.text, 10, 20, finish_reason='stop')
            return response
    monkeypatch.setattr(reader_fixture, 'TruncatingProvider', StoppedIncomplete)
    report, db, rid, provider = run_reader(tmp_path, {first_broken})
    assert report['provider_attempts'] == db.get_run(rid)['used_llm_calls'] == 3
    assert report['answer'] == 'A source fact. [source1]'
    assert report['calls'][first_broken - 1]['finish_reason'] == 'stop'
    assert report['calls'][first_broken - 1]['text'] == '{"incomplete":'
    assert not any(message['role'] == 'assistant' for message in provider.calls[first_broken])
    assert db.list_llm_calls(rid)[first_broken]['stage'] == 'json_repair'


def test_two_malformed_stop_responses_fail_without_a_third_attempt(tmp_path, monkeypatch):
    import test_reader_truncation as reader_fixture
    class StoppedIncomplete(TruncatingProvider):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            if response.finish_reason == 'length':
                return LLMResponse(response.text, 10, 20, finish_reason='stop')
            return response
    monkeypatch.setattr(reader_fixture, 'TruncatingProvider', StoppedIncomplete)
    report, db, rid, provider = run_reader(tmp_path, {1, 2})
    assert report['status'] == 'FAILED'
    assert not report.get('answer')
    assert len(provider.calls) == db.get_run(rid)['used_llm_calls'] == 2
    assert all(call['finish_reason'] == 'stop' for call in report['calls'])
