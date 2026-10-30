"""Malformed coverage metadata must not bypass the existing bounded patch."""
import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.answer_contracts import (
    AuditSupportError, conservative_audit_contract, evidence_spans, validate_audit,
)
from paperpilot_agentic.providers import LLMResponse


SOURCE = [{'evidence_id': 'e1', 'content': 'The encoder uses attention. The decoder uses interpolation.'}]
DRAFT = {'facts': [
    {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'The decoder uses interpolation.'},
    {'id': 'f2', 'kind': 'SOURCE_FACT', 'text': 'The encoder uses attention.',
     'evidence_ids': ['e1'], 'depends_on': []}],
    'direct_answer_id': 'f2', 'limitations': ['The current evidence cannot determine applicability.']}


def rejected(draft):
    return {'assessments': [{'fact_id': f['id'], 'verdict': 'UNSUPPORTED',
             'relevance': 'DIRECT', 'reason': 'Rejected pending review.'} for f in draft['facts']],
        'coverage': 'PARTIAL', 'missing_aspects': [], 'requirement_assessments': [],
        'limitation_assessments': [{'limitation_index': 0, 'role': 'REQUEST_GAP',
                                  'requirement': 'invented requirement', 'reason': 'Not established.'}]}


def test_rejected_only_normalization_is_explicit_and_never_adds_support():
    draft = {**DRAFT, 'facts': DRAFT['facts'][1:]}
    raw = rejected(draft)
    spans = evidence_spans(SOURCE)
    with pytest.raises(AuditSupportError):
        conservative_audit_contract(raw, draft, spans, question='Explain the encoder and decoder.')
    result = conservative_audit_contract(raw, draft, spans,
        question='Explain the encoder and decoder.', allow_rejected_only=True)
    validate_audit(result, draft, SOURCE, spans=spans, require_relevance=True,
                   question='Explain the encoder and decoder.')
    assert all(a['verdict'] == 'UNSUPPORTED' for a in result['assessments'])
    assert result['coverage'] == 'PARTIAL'
    assert result['withheld_limitation_indices'] == [0]


class Model:
    def __init__(self, *, patch_error=False):
        self.calls = []
        self.patch_error = patch_error

    def ensure_capacity(self, requests):
        pass

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        self.calls.append(messages)
        payload = user_payload(messages)
        if len(self.calls) == 1:
            value = DRAFT
        elif 'replacement_contract' in payload:
            if self.patch_error:
                raise OSError('Patch response unknown: stop this chain.')
            assert set(payload['replacement_contract']['target_ids']) == {'f1', 'f2'}
            value = {'replacements': {
                'f1': {'kind': 'SOURCE_FACT', 'text': 'The decoder uses interpolation.',
                       'evidence_ids': ['e1'], 'depends_on': []},
                'f2': {'kind': 'SOURCE_FACT', 'text': 'The encoder uses attention.',
                       'evidence_ids': ['e1'], 'depends_on': []}}, 'limitations': []}
        elif any('replacement_contract' in user_payload(m) for m in self.calls[:-1]):
            value = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED',
                'relevance': 'DIRECT', 'span_ids': ['s0']} for f in payload['facts']],
                'coverage': 'COMPLETE', 'missing_aspects': []}
        else:
            value = rejected({'facts': payload['facts']})
        return LLMResponse(json.dumps(value), finish_reason='stop')


def user_payload(messages):
    return next(json.loads(m['content']) for m in messages
                if m['role'] == 'user' and m['content'].lstrip().startswith('{'))


def test_actual_flow_reaches_one_patch_then_fresh_audit_without_publishing_rejected_draft():
    model = Model()
    progress = []
    report = answer_then_audit('Explain the encoder and decoder.', SOURCE, model,
                              on_progress=lambda r: progress.append(json.loads(json.dumps(r))))
    assert report['status'] == 'REVIEW_REQUIRED', report.get('failure_detail')
    assert report['draft_isolation']['repair_status'] == 'PATCH_AUDITED'
    assert len(model.calls) == 5
    assert sum('replacement_contract' in user_payload(m) for m in model.calls) == 1
    assert 'decoder uses interpolation' in report['answer']
    assert progress and not progress[0].get('answer')
    assert report['raw_draft'] == DRAFT


def test_disabled_repair_never_uses_rejected_only_escape_hatch():
    model = Model()
    report = answer_then_audit('Explain the encoder and decoder.', SOURCE, model,
                              allow_repair=False)
    assert report['status'] == 'FAILED'
    assert len(model.calls) == 2
    assert not report.get('answer')


def test_unknown_patch_stops_without_another_request_or_unreviewed_answer():
    model = Model(patch_error=True)
    report = answer_then_audit('Explain the encoder and decoder.', SOURCE, model)
    assert report['status'] == 'FAILED'
    assert report['error_type'] == 'OSError'
    assert len(model.calls) == 4
    assert report['draft_isolation']['repair_status'] == 'STOPPED_ERROR'
    assert not report.get('answer')
