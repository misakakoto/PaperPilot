from copy import deepcopy
import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit, validate_draft
from paperpilot_agentic.draft_schema import canonical_draft
from paperpilot_agentic.providers import LLMResponse


EVIDENCE = [{'evidence_id': 'e1', 'content': 'The method uses Corpus A on the validation split.'}]
DRAFT = {'answer_facts': [{'id': 'a1', 'kind': 'SOURCE_FACT', 'text': 'The method uses Corpus A.',
                          'evidence_ids': ['e1'], 'depends_on': []}],
         'context_facts': [], 'limitations': []}
AUDIT = {'assessments': [{'fact_id': 'a1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT', 'span_ids': ['s0']}],
         'coverage': 'COMPLETE', 'missing_aspects': []}


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.calls.append(messages)
        return LLMResponse(text=json.dumps(next(self.outputs)), finish_reason='stop')


def test_layered_draft_is_canonicalized_without_rewriting_any_claim():
    raw = deepcopy(DRAFT)
    canonical = canonical_draft(raw)
    assert raw == DRAFT
    assert canonical['facts'] == [{**raw['answer_facts'][0], 'writer_role': 'answer'}]
    assert validate_draft(canonical, EVIDENCE) == canonical


@pytest.mark.parametrize('bad', [None, {}, {'answer_facts': 'wrong', 'context_facts': [], 'limitations': []},
    {**DRAFT, 'direct_answer_id': 'a1'}, {**DRAFT, 'context_facts': [DRAFT['answer_facts'][0]]}])
def test_invalid_or_duplicate_layered_claims_cannot_bypass_validation(bad):
    with pytest.raises(ValueError):
        validate_draft(canonical_draft(bad), EVIDENCE)


@pytest.mark.parametrize('relevance,verdict,published', [
    ('DIRECT', 'SUPPORTED', True), ('NECESSARY_CONTEXT', 'SUPPORTED', True),
    ('DIRECT', 'UNSUPPORTED', False), ('TANGENTIAL', 'SUPPORTED', False),
])
def test_writer_role_is_not_an_audit_verdict(relevance, verdict, published):
    audit = deepcopy(AUDIT)
    audit['assessments'][0].update(relevance=relevance, verdict=verdict)
    model = Model([DRAFT, audit])
    report = answer_then_audit('Which corpus is used?', EVIDENCE, model, draft_contract='layered-v1')
    assert report['status'] == 'REVIEW_REQUIRED'
    assert bool(report['answer']) == published
    assert report['direct_answer_supported'] == (published and relevance == 'DIRECT')
    assert report['raw_draft'] == DRAFT
    assert report['provider_attempts'] == 2
    assert 'writer_role is only a fallible' in model.calls[1][0]['content']


def test_layered_format_repair_keeps_same_schema_and_shared_allowance():
    model = Model([{'facts': []}, DRAFT, AUDIT])
    report = answer_then_audit('Which corpus is used?', EVIDENCE, model, draft_contract='layered-v1')
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['raw_draft'] == {'facts': []}
    assert report['repaired_draft'] == DRAFT
    assert report['provider_attempts'] == 3
    assert 'with answer_facts, context_facts and limitations' in model.calls[1][-1]['content']


def test_fact_that_only_supplies_background_does_not_become_direct_when_answer_is_empty():
    value = {'answer_facts': [], 'context_facts': DRAFT['answer_facts'],
             'limitations': ['The current evidence cannot determine the speed.']}
    audit = deepcopy(AUDIT)
    audit['assessments'][0]['relevance'] = 'NECESSARY_CONTEXT'
    report = answer_then_audit('How fast?', EVIDENCE, Model([value, audit]), draft_contract='layered-v1')
    assert report['answer_scope'] == 'BACKGROUND_ONLY'
    assert report['answer_sections'][0]['kind'] == 'context'


def test_unknown_contract_is_rejected_before_billing():
    with pytest.raises(ValueError, match='Unknown'):
        answer_then_audit('Question?', EVIDENCE, None, draft_contract='other')
