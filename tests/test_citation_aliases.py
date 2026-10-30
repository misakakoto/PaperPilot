import json

import pytest

from paperpilot_agentic.citation_aliases import CitationAliases
from paperpilot_agentic.answer_audit import validate_draft
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


LONG = 'evidence_' + 'a' * 32


def test_alias_roundtrip_preserves_source_content_and_offsets():
    aliases = CitationAliases([LONG, 'ref1'])
    assert aliases.encode_map[LONG] == 'ref2'
    payload = {'evidence': [{'evidence_id': LONG, 'content': LONG, 'start': 7}],
               'facts': [{'evidence_ids': [LONG], 'text': LONG}]}
    encoded = aliases.transform(payload)
    assert encoded['evidence'][0]['content'] == LONG
    assert encoded['facts'][0]['text'] == LONG
    assert aliases.transform(encoded, decode=True) == payload
    messages = [{'role': 'system', 'content': 'Do not change'},
                {'role': 'user', 'content': json.dumps(payload)}]
    assert json.loads(aliases.messages(messages)[1]['content']) == encoded
    assert json.loads(messages[1]['content']) == payload


@pytest.mark.parametrize('bad', ['ref999', 'ref', LONG[:-1]])
def test_unknown_and_truncated_ids_are_not_guessed(bad):
    aliases = CitationAliases([LONG])
    draft = {'facts': [{'id': 'f1', 'text': 'Claim', 'evidence_ids': [bad]}],
             'direct_answer_id': 'f1'}
    with pytest.raises(ValueError, match='supplied evidence'):
        validate_draft(aliases.transform(draft, decode=True), [{'evidence_id': LONG}])


def test_mapping_is_request_local_and_order_independent():
    other = 'evidence_' + 'b' * 32
    assert CitationAliases([LONG, other]).encode_map == CitationAliases([other, LONG]).encode_map
    assert CitationAliases([other]).decode_map['ref1'] == other
    assert CitationAliases([LONG]).decode_map['ref1'] == LONG


def test_writer_and_auditor_use_aliases_but_published_citations_stay_durable():
    class Model:
        calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                assert payload['evidence'][0]['evidence_id'] == 'ref1'
                value = {'facts': [{'id': 'f1', 'text': 'Uses TransE.',
                                    'evidence_ids': ['ref1']}], 'direct_answer_id': 'f1'}
            else:
                assert payload['facts'][0]['evidence_ids'] == ['ref1']
                assert payload['sources'][0]['evidence_id'] == 'ref1'
                value = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                         'relevance': 'DIRECT', 'span_ids': ['s0']}],
                         'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop')

    report = answer_then_audit('Baseline?', [{'evidence_id': LONG, 'content': 'Uses TransE.'}], Model())
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['draft']['facts'][0]['evidence_ids'] == [LONG]
    assert f'[{LONG}]' in report['answer']
    assert '[ref1]' not in report['answer']
    assert 'ref1' in report['calls'][0]['text']  # Original model output remains unchanged.
    assert report['citation_aliases'] == {'ref1': LONG}
