"""Publication regressions: boundaries, output language and citation revision."""
from copy import deepcopy
import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit, validate_draft
from paperpilot_agentic.answer_coverage import apply_final_coverage, request_contract
from paperpilot_agentic.providers import LLMResponse


def boundary_sample():
    draft = {'facts': [{'id': 'f', 'text': 'Beta estimates spatial displacement.',
                        'evidence_ids': ['b']}], 'direct_answer_id': 'f',
             'limitations': ['Cannot determine from the supplied passages whether Beta is best on other datasets.']}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [
                 {'requirement': 'Describe Beta', 'fact_ids': ['f'], 'gap': ''},
                 {'requirement': 'State the comparison boundary', 'fact_ids': [],
                  'limitation_indices': [0], 'gap': ''}],
             'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY',
                                        'reason': 'The requested boundary limits extrapolation.'}]}
    return draft, audit


def test_requested_reviewed_visible_boundary_covers_a_subpart_without_becoming_a_fact():
    from paperpilot_agentic.answer_contracts import audited_presentation
    draft, audit = boundary_sample()
    original = deepcopy(draft)
    actual, report = apply_final_coverage(draft, audit, [], [], request_contract('Describe Beta and its boundary.', []))
    assert actual['coverage'] == 'COMPLETE'
    assert report['requirements'][1]['retained_limitation_indices'] == [0]
    shown = audited_presentation(draft, actual, question='Describe Beta and its boundary.')
    assert shown['answer_scope'] == 'MODEL_REVIEWED'
    assert shown['limitations'] == draft['limitations']
    assert not shown['internal_gap_diagnostics']['request_gaps']
    assert draft == original


@pytest.mark.parametrize('mutation', ['unclassified', 'request_gap', 'positive_claim', 'wrong_language'])
def test_unreviewed_or_unrenderable_boundary_cannot_satisfy_requested_work(mutation):
    draft, audit = boundary_sample()
    question = 'Describe Beta and its boundary.'
    if mutation == 'unclassified':
        audit.pop('limitation_assessments')
    elif mutation == 'request_gap':
        audit['limitation_assessments'][0]['role'] = 'REQUEST_GAP'
        audit['limitation_assessments'][0]['requirement'] = 'State the comparison boundary'
        audit['requirement_assessments'][1]['gap'] = 'The requested boundary remains unresolved.'
        audit['coverage'] = 'PARTIAL'
    elif mutation == 'positive_claim':
        draft['limitations'] = ['Beta is best on every dataset.']
    else:
        question = '请用中文说明机制和结论边界。'
    actual, report = apply_final_coverage(draft, audit, [], [], request_contract(question, []))
    assert actual['coverage'] == 'PARTIAL'
    assert report['requirements'][1]['retained_limitation_indices'] == []


@pytest.mark.parametrize('indices', [[True], [-1], [1], [0, 0], ['0'], '0'])
def test_requirement_boundary_indices_must_be_unique_and_owned(indices):
    from paperpilot_agentic.answer_coverage import validate_requirement_assessments
    draft, audit = boundary_sample()
    audit['requirement_assessments'][1]['limitation_indices'] = indices
    with pytest.raises(ValueError, match='limitation'):
        validate_requirement_assessments(audit, draft)


def test_chinese_boundary_is_checked_before_independent_audit_and_uses_one_repair():
    draft = {'facts': [{'id': 'f', 'text': 'Beta 估计空间位移。', 'evidence_ids': ['b']}],
             'direct_answer_id': 'f', 'limitations': ['Cannot determine whether Beta is best on other datasets.']}
    corrected = deepcopy(draft)
    corrected['limitations'] = ['当前证据无法确定 Beta 是否在其他数据集上最优。']
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
                assert 'limitation_index=0' in messages[-1]['content']
                value = corrected
            else:
                assert self.calls == 3
                payload = json.loads(messages[-1]['content'])
                assert payload['limitations'] == corrected['limitations']
                value = {'assessments': [{'fact_id': 'f', 'span_ids': ['s0'], 'verdict': 'SUPPORTED',
                                         'relevance': 'DIRECT'}], 'coverage': 'COMPLETE', 'missing_aspects': [],
                         'requirement_assessments': [{'requirement': '说明机制', 'fact_ids': ['f'], 'gap': ''}],
                         'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY',
                                                    'reason': 'Other-dataset ranking was not requested.'}]}
            return LLMResponse(json.dumps(value), prompt_tokens=10, completion_tokens=10, finish_reason='stop')
    model = Model()
    report = answer_then_audit('请用中文说明机制。',
        [{'evidence_id': 'b', 'content': 'Beta 估计空间位移。'}], model)
    assert model.calls == 3
    assert report['answer_scope'] == 'MODEL_REVIEWED'
    assert report['raw_draft'] == draft
    assert report['draft']['limitations'] == corrected['limitations']


def test_owned_english_override_still_accepts_an_english_boundary():
    draft, _ = boundary_sample()
    assert validate_draft(draft, [{'evidence_id': 'b'}], question='请用英文回答。') == draft


def test_an_explicit_gap_remains_partial_even_when_a_true_fact_answers_part_of_it():
    draft, audit = boundary_sample()
    audit['requirement_assessments'][0]['gap'] = 'The second requested mechanism is unresolved.'
    actual, report = apply_final_coverage(draft, audit, [], [], request_contract('Describe both mechanisms.', []))
    assert actual['coverage'] == 'PARTIAL'
    assert report['requirements'][0]['status'] == 'UNRESOLVED'


def test_a_reviewed_boundary_removed_from_final_layout_cannot_keep_complete(monkeypatch):
    from paperpilot_agentic.read_packet import build_read_packet
    from paperpilot_agentic import answer_layout
    draft, audit = boundary_sample()
    audit['assessments'][0]['span_ids'] = ['s0']
    class Model:
        calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(json.dumps(draft if self.calls == 1 else audit), finish_reason='stop')
    def incomplete_layout(question, presentation, draft, audit):
        return {**presentation, 'limitations_in_answer': True}
    monkeypatch.setattr(answer_layout, 'host_requested_layout', incomplete_layout)
    question = 'Describe Beta and its boundary.'
    packet = build_read_packet(question, [{'evidence_id': 'b', 'content': draft['facts'][0]['text']}])
    report = answer_then_audit(question, packet.rows, Model(), shared_read_packet=packet)
    assert report['answer_scope'] == 'PARTIAL'
    assert report['final_request_coverage']['coverage'] == 'PARTIAL'
    assert report['final_request_coverage']['requirements'][1]['status'] == 'MISSING_IN_DISPLAY'
    assert report['final_request_coverage']['displayed_limitation_indices'] == []


@pytest.mark.parametrize('max_calls', [4, 5, 20])
@pytest.mark.parametrize('final_verdict', ['SUPPORTED', 'UNSUPPORTED'])
def test_daily_runtime_revises_misbound_citation_despite_reviewed_scope_boundary(tmp_path, max_calls, final_verdict):
    from test_reader_pipeline import factory
    from paperpilot_agentic.models import Budget
    class Search:
        def search(self, query, **kwargs):
            return [{'stable_id': name, 'paper_sha256': 'sha', 'source_uri': f'paper://sha/{name}',
                     'content': text, 'page': 3 if name == 'refs' else 1} for name, text in [
                         ('a', 'Alpha uses temporal tokens.'), ('b', 'Beta estimates spatial displacement.'),
                         ('refs', 'REFERENCES\nBibliography entry.')]]
    class Model:
        calls = 0
        writes = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            payload = json.loads(messages[-1]['content'])
            if 'unread_candidates' in payload:
                value = {'action': 'STOP', 'reason': 'No unread evidence', 'evidence_ids': []}
            elif 'evidence' in payload:
                self.writes += 1
                sources = payload['evidence']
                name = 'Alpha' if self.writes == 1 else 'Beta'
                eid = next(e['evidence_id'] for e in sources if name in e['content'])
                if self.writes == 2:
                    assert payload['prior_review_untrusted']['citation_bindings']['changed_fact_ids'] == ['f']
                draft, _ = boundary_sample()
                draft['facts'][0]['evidence_ids'] = [eid]
                value = draft
            else:
                source = next(s for s in payload['sources'] if any('Beta' in text for text in s['spans'].values()))
                sid = next(iter(source['spans']))
                _, value = boundary_sample()
                value['assessments'][0]['span_ids'] = [sid]
                if self.writes == 2:
                    value['assessments'][0]['verdict'] = final_verdict
                    if final_verdict == 'UNSUPPORTED':
                        value['assessments'][0]['span_ids'] = []
                        value['coverage'] = 'PARTIAL'
            return LLMResponse(json.dumps(value), prompt_tokens=10, completion_tokens=10, finish_reason='stop')
    model = Model()
    runtime = factory(tmp_path, model)
    runtime.search_adapter = Search()
    rid = runtime.enqueue(question='Using Alpha and Beta sources, describe Beta and state the comparison boundary.',
                          metadata={'engine': 'answer_first_text', 'paper_sha256s': ['sha']},
                          budget=Budget(max_llm_calls=max_calls))['run']['run_id']
    result = runtime.resume(rid)
    if max_calls == 4:
        assert result['run']['research_outcome'] == 'UNRESOLVED'
        assert model.calls == result['run']['used_llm_calls'] == 2
        assert result['run']['metadata']['reader_recovery_status'] == 'NOT_STARTED_BUDGET'
        runtime.resume(rid)
        assert model.calls == 2
        return
    assert result['run']['research_outcome'] == ('ANSWERED' if final_verdict == 'SUPPORTED' else 'UNRESOLVED')
    assert model.calls == result['run']['used_llm_calls'] == 4
    rounds = result['run']['metadata']['reader_rounds']
    assert rounds[0]['read_packet']['packet_sha256'] == rounds[1]['read_packet']['packet_sha256']
    assert rounds[0]['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert rounds[0]['draft']['facts'][0]['evidence_ids'] != rounds[1]['draft']['facts'][0]['evidence_ids']
    assert rounds[1]['audit']['assessments'][0]['verdict'] == final_verdict
    runtime.resume(rid)
    assert model.calls == 4


def test_combined_draft_errors_report_all_wrong_language_caveats_in_one_repair():
    from paperpilot_agentic.answer_contracts import draft_repair_diagnostics
    draft, _ = boundary_sample()
    draft['limitations'].append('Cannot determine whether further tests generalize.')
    diagnostic = draft_repair_diagnostics(draft, question='请用中文回答。', primary=ValueError('Unknown citation'))
    assert 'Unknown citation' in diagnostic
    assert 'limitation_index=0' in diagnostic and 'limitation_index=1' in diagnostic


@pytest.mark.parametrize('mutation', ['request_gap', 'other_rejected_fact', 'changed_source',
                                     'missing_boundary_map', 'original_model_gap'])
def test_citation_shortcut_does_not_bypass_an_unresolved_request_or_changed_original(mutation):
    from paperpilot_agentic.reader_recovery import citation_revision_plan
    evidence = [{'evidence_id': 'a', 'content': 'Alpha uses temporal tokens.'},
                {'evidence_id': 'b', 'content': 'Beta estimates spatial displacement.'}]
    draft, audit = boundary_sample()
    draft['facts'][0]['evidence_ids'] = ['a']
    audit['assessments'][0]['span_ids'] = ['s1']
    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(json.dumps(audit), finish_reason='stop')
    report = answer_then_audit('Describe Beta and its boundary.', evidence, Model(), frozen_draft=draft)
    assert citation_revision_plan(report, evidence, question=report['question'])['provider_attempts'] == 0
    if mutation == 'request_gap':
        report['audit']['limitation_assessments'][0].update(role='REQUEST_GAP',
            requirement='State the comparison boundary')
        report['audit']['requirement_assessments'][1]['gap'] = 'The boundary is unresolved.'
    elif mutation == 'other_rejected_fact':
        report['draft']['facts'].append({'id': 'extra', 'text': 'Another unresolved mechanism.',
                                         'evidence_ids': ['b']})
        report['audit']['assessments'].append({'fact_id': 'extra', 'verdict': 'UNCERTAIN'})
    elif mutation == 'changed_source':
        evidence[1]['content'] = 'Replaced original.'
    elif mutation == 'missing_boundary_map':
        report['audit']['requirement_assessments'][1].pop('limitation_indices')
    else:
        report['citation_bindings']['original_missing_aspects'] = ['A requested detail is absent.']
    assert citation_revision_plan(report, evidence, question=report['question']) is None
