import copy
import json
import pytest
from paperpilot_agentic.answer_contracts import conservative_audit_contract


def test_bad_request_quote_downgrades_coverage_without_erasing_source_audit():
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': 'Alpha uses attention.',
                       'evidence_ids': ['e']}], 'direct_answer_id': 'f', 'limitations': []}
    spans = [{'span_id': 's0', 'evidence_id': 'e', 'text': 'Alpha uses attention.'}]
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                             'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [{'requirement': 'Explain method', 'fact_ids': ['f'],
                                         'gap': '', 'request_quote': 'Explain...method'}]}
    before = copy.deepcopy(audit)
    result = conservative_audit_contract(audit, draft, spans, question='Explain the method.')
    assert result['assessments'] == audit['assessments']
    assert result['coverage'] == 'PARTIAL' and result['missing_aspects']
    assert result['requirement_assessments'] == []
    assert result['requirement_contract_errors'][0]['raw_assessment'] == audit['requirement_assessments'][0]
    assert audit == before


def test_local_repair_extras_are_quarantined_and_target_still_requires_independent_validation():
    from paperpilot_agentic.answer_isolation import isolate_repair_targets
    target = {'id': 'bad', 'kind': 'INFERENCE', 'text': 'Candidate has the highest tested F1.',
              'evidence_ids': ['t'], 'depends_on': ['baseline', 'candidate']}
    extra = {'id': 'new', 'kind': 'SOURCE_FACT', 'text': 'Invented benefit.', 'evidence_ids': ['t']}
    raw = {'facts': [target, extra], 'limitations': []}
    patch, trace = isolate_repair_targets(raw, ['bad'])
    assert patch == {'facts': [target], 'limitations': []}
    assert trace == [extra]
    assert raw['facts'] == [target, extra]


@pytest.mark.parametrize('ids', [['extra'], ['bad', 'bad'], ['bad', 'extra', 'extra']])
def test_missing_or_ambiguous_repair_targets_never_normalized(ids):
    from paperpilot_agentic.answer_isolation import isolate_repair_targets
    value = {'facts': [{'id': fid} for fid in ids], 'limitations': []}
    with pytest.raises(ValueError):
        isolate_repair_targets(value, ['bad'])


def test_formal_audit_repair_with_bad_quotes_preserves_facts_as_partial():
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse
    class Model:
        calls = 0
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            if self.calls == 1:
                result = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': 'Alpha uses attention.',
                                    'evidence_ids': ['e']}], 'direct_answer_id': 'f', 'limitations': []}
            else:
                result = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                                          'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': [],
                          'requirement_assessments': [{'requirement': 'Explain method', 'fact_ids': ['f'],
                                                      'gap': '', 'request_quote': 'Explain...method'}]}
            return LLMResponse(json.dumps(result), finish_reason='stop')
    model = Model()
    report = answer_then_audit('Explain the method.', [{'evidence_id': 'e', 'content': 'Alpha uses attention.'}], model)
    assert model.calls == 3
    assert report['answer_scope'] == 'PARTIAL'
    assert 'Alpha uses attention.' in report['answer']
    assert report['repaired_audit']['coverage'] == 'COMPLETE'
    assert report['audit']['coverage'] == 'PARTIAL'


def test_formal_content_patch_quarantines_addition_and_reaudits_target_only():
    from paperpilot_agentic.answer_audit import answer_then_audit
    from paperpilot_agentic.providers import LLMResponse
    class Model:
        calls = 0
        def ensure_capacity(self, requests):
            pass
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if self.calls == 1:
                result = {'facts': [
                    {'id': 'method', 'kind': 'SOURCE_FACT', 'text': 'Encoder uses attention.', 'evidence_ids': ['e']},
                    {'id': 'bad', 'kind': 'SOURCE_FACT', 'text': 'Decoder uses F = concat_c(T1,T2).',
                     'evidence_ids': ['e']}],
                    'direct_answer_id': 'method', 'limitations': []}
            elif self.calls == 3:
                result = {'facts': [
                    {'id': 'bad', 'kind': 'SOURCE_FACT', 'text': 'Decoder uses interpolation.', 'evidence_ids': ['e']},
                    {'id': 'extra', 'kind': 'SOURCE_FACT', 'text': 'Invented benefit.', 'evidence_ids': ['e']}],
                    'limitations': []}
            else:
                assert 'extra' not in [f['id'] for f in payload['facts']]
                result = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                                          'span_ids': f['allowed_span_ids']} for f in payload['facts']],
                          'coverage': 'PARTIAL' if self.calls == 2 else 'COMPLETE',
                          'missing_aspects': ['Decoder missing'] if self.calls == 2 else []}
            return LLMResponse(json.dumps(result), finish_reason='stop')
    model = Model()
    report = answer_then_audit('Explain encoder and decoder in words.', [{'evidence_id': 'e',
        'content': 'Encoder uses attention. Decoder uses interpolation.'}], model)
    assert model.calls == 4
    assert report['draft_isolation']['repair_status'] == 'PATCH_AUDITED'
    assert 'Decoder uses interpolation.' in report['answer'] and 'Invented benefit.' not in report['answer']
    assert report['quarantined_repair_additions'][0]['id'] == 'extra'
    assert len(report['raw_local_repair']['facts']) == 2
