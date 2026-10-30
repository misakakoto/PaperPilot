"""Scope boundaries are not unanswered requests; completion is not correctness."""
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_contracts import audited_presentation, validate_audit
from paperpilot_agentic.answer_boundaries import safe_limitations
from paperpilot_agentic.durable_reader import reader_completion_state


def sample():
    draft = {'facts': [{'id': 'f1', 'text': 'Orion uses feature warping.',
                        'evidence_ids': ['e1']}], 'direct_answer_id': 'f1',
             'limitations': ['Cannot determine from the supplied passages whether Orion is best on other datasets.']}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                             'relevance': 'DIRECT', 'quotes': [{'evidence_id': 'e1',
                             'quote': 'Orion uses feature warping.'}]}],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [{'requirement': 'Describe the mechanism.',
                                          'fact_ids': ['f1'], 'gap': ''}],
             'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY',
                                        'reason': 'Other-dataset ranking was not requested.'}]}
    return draft, audit


def test_audited_boundary_does_not_trigger_missing_evidence_recovery():
    draft, audit = sample()
    result = audited_presentation(draft, audit)
    assert result['answer_scope'] == 'MODEL_REVIEWED'
    assert result['limitations']  # Keep the boundary visible.
    assert result['completion_assurance'] == 'MODEL_REVIEWED_NOT_FACTUAL_ACCEPTANCE'


@pytest.mark.parametrize('change', ['unclassified', 'request_gap', 'missing_aspect', 'unsupported'])
def test_actual_gaps_and_legacy_unclassified_boundaries_remain_partial(change):
    draft, audit = sample()
    if change == 'unclassified':
        audit.pop('limitation_assessments')
    elif change == 'request_gap':
        audit['limitation_assessments'][0].update(role='REQUEST_GAP', requirement='Describe the mechanism.')
        audit['requirement_assessments'][0]['gap'] = 'Feature representation is unresolved.'
    elif change == 'missing_aspect':
        audit['missing_aspects'] = ['The requested baseline is unresolved.']
    else:
        audit['assessments'][0]['verdict'] = 'UNCERTAIN'
    assert audited_presentation(draft, audit)['answer_scope'] != 'MODEL_REVIEWED'


def test_runtime_answered_requires_explicit_model_reviewed_completion():
    draft, audit = sample()
    report = {'status': 'REVIEW_REQUIRED', 'audit': audit, **audited_presentation(draft, audit)}
    assert reader_completion_state(report)[1].value == 'ANSWERED'
    assert reader_completion_state({**report, 'answer_scope': 'PARTIAL'})[1].value == 'PARTIAL'
    assert reader_completion_state({'status': 'REVIEW_REQUIRED', 'answer': 'Unreviewed'})[1].value == 'PARTIAL'


@pytest.mark.parametrize('change', ['duplicate', 'foreign_index', 'unknown_role', 'unbound_gap'])
def test_invalid_limitation_classification_cannot_release_completion(change):
    draft, audit = sample()
    value = deepcopy(audit)
    row = value['limitation_assessments'][0]
    if change == 'duplicate':
        value['limitation_assessments'].append(dict(row))
    elif change == 'foreign_index':
        row['limitation_index'] = 2
    elif change == 'unknown_role':
        row['role'] = 'CORRECT'
    else:
        row.update(role='REQUEST_GAP', requirement='Unrequested hardware.')
    with pytest.raises(ValueError, match='[Ll]imitation'):
        validate_audit(value, draft, [{'evidence_id': 'e1', 'content': draft['facts'][0]['text']}])


def test_english_boundary_does_not_carry_an_unreviewed_causal_suffix():
    raw = 'The supplied passages do not establish whether Orion handles offsets, as a learned warp proves robustness.'
    visible, withheld = safe_limitations([raw])
    assert visible == ['Cannot determine from the supplied passages whether Orion handles offsets.']
    assert withheld == [raw]


def test_formal_answer_entry_keeps_boundary_and_sends_classification_contract():
    import json
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse
    draft, audit = sample()
    class Model:
        def __init__(self):
            self.calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(text=json.dumps(draft), prompt_tokens=10, completion_tokens=10,
                                   finish_reason='stop')
            assert 'limitation_assessments' in messages[0]['content']
            assert json.loads(messages[-1]['content'])['limitations'] == draft['limitations']
            value = {**audit, 'assessments': [{
                'fact_id': 'f1', 'span_ids': ['s0'], 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                'reason': 'The supplied method paragraph describes feature warping.'}]}
            return LLMResponse(text=json.dumps(value), prompt_tokens=10, completion_tokens=10,
                               finish_reason='stop')
    model = Model()
    result = answer_then_audit('Describe the mechanism.',
        [{'evidence_id': 'e1', 'content': draft['facts'][0]['text']}], model)
    assert result['answer_scope'] == 'MODEL_REVIEWED'
    assert result['limitations'] and model.calls == 2
    assert result['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'


@pytest.mark.parametrize('role', ['DIRECT', 'NECESSARY_CONTEXT', 'TANGENTIAL'])
def test_mixed_fact_relevance_role_is_rejected_with_owned_actionable_error(role):
    draft, audit = sample()
    audit['limitation_assessments'][0]['role'] = role
    before = deepcopy(audit)
    with pytest.raises(ValueError, match='limitation_index=0') as exc:
        validate_audit(audit, draft, [{'evidence_id': 'e1', 'content': draft['facts'][0]['text']}])
    assert role in str(exc.value)
    assert 'SCOPE_BOUNDARY or REQUEST_GAP only' in str(exc.value)
    assert audit == before  # Rejection never guesses a replacement role.


def test_bounded_repair_of_mixed_limitation_role_keeps_source_checks_and_facts():
    import json
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse
    draft, audit = sample()
    audit['assessments'][0].update(span_ids=['s0'], reason='The source states feature warping.')
    audit['assessments'][0].pop('quotes')
    bad = deepcopy(audit)
    bad['limitation_assessments'][0]['role'] = 'NECESSARY_CONTEXT'
    class Model:
        calls = 0
        def ensure_capacity(self, requests):
            pass
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            if self.calls == 1:
                value = draft
            elif self.calls == 2:
                value = bad
            else:
                assert self.calls == 3
                prompt = '\n'.join(str(m['content']) for m in messages)
                assert 'limitation_index=0' in prompt
                assert 'Allowed limitation roles are SCOPE_BOUNDARY or REQUEST_GAP only' in prompt
                assert draft['facts'][0]['text'] in prompt
                value = audit
            return LLMResponse(json.dumps(value), prompt_tokens=10, completion_tokens=10, finish_reason='stop')
    model = Model()
    result = answer_then_audit('Describe the mechanism.',
        [{'evidence_id': 'e1', 'content': draft['facts'][0]['text']}], model)
    assert model.calls == 3
    assert result['answer_scope'] == 'MODEL_REVIEWED'
    assert result['raw_audit']['limitation_assessments'][0]['role'] == 'NECESSARY_CONTEXT'
    assert result['audit']['limitation_assessments'][0]['role'] == 'SCOPE_BOUNDARY'
    assert result['audit_repair']['validation_error'].startswith('Invalid limitation role')
    assert draft['facts'][0]['text'] in '\n'.join(result['answer_units'])
