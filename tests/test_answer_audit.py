import json

import pytest

from paperpilot_agentic.answer_audit import validate_draft, validate_audit, answer_then_audit, evidence_spans, audit_source_view, audit_facts_view
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.answer_audit import audited_presentation


EVIDENCE = [{'evidence_id': 'e1', 'content': 'The baseline uses a linear classifier.'},
            {'evidence_id': 'e2', 'content': 'The baseline uses a neural classifier.'}]
DRAFT = {'facts': [{'id': 'f1', 'text': 'Uses a linear classifier.', 'evidence_ids': ['e1']}],
         'direct_answer_id': 'f1'}


def audit(eid='e1', quote='The baseline uses a linear classifier.'):
    return {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                            'quotes': [{'evidence_id': eid, 'quote': quote}], 'span_ids': ['s0']}],
            'coverage': 'COMPLETE', 'missing_aspects': []}


def test_audit_accepts_grounded_quote_without_claiming_semantic_truth():
    assert validate_audit(audit(), DRAFT, EVIDENCE)['coverage'] == 'COMPLETE'


@pytest.mark.parametrize('value', [audit('e2'), audit(quote='Invented result'),
                                 {'assessments': [], 'coverage': 'COMPLETE'}])
def test_rejects_wrong_source_fabricated_quote_and_missing_fact(value):
    with pytest.raises(ValueError):
        validate_audit(value, DRAFT, EVIDENCE)


def test_draft_rejects_unknown_citation():
    with pytest.raises(ValueError):
        validate_draft({'facts': [{'id': 'f1', 'text': 'X', 'evidence_ids': ['unknown']}]}, EVIDENCE)


def test_abstention_does_not_require_inventing_fact():
    assert validate_draft({'facts': [], 'limitations': ['No usable evidence']}, EVIDENCE)
    with pytest.raises(ValueError):
        validate_draft({'facts': []}, EVIDENCE)
    with pytest.raises(ValueError):
        validate_draft({'facts': [], 'limitations': {'fake': 'reason'}}, EVIDENCE)


def test_empty_draft_is_not_vacuously_all_supported():
    result = answer_then_audit('Question?', EVIDENCE, Model([
        {'facts': [], 'limitations': ['No answer found']},
        {'assessments': [], 'coverage': 'UNANSWERABLE'}]))
    assert result['limitations'] == ['No answer found']
    assert not result['all_facts_supported']


def test_empty_evidence_does_not_spend():
    report = answer_then_audit('Question', [], None)
    assert report['status'] == 'NO_EVIDENCE'
    assert report['provider_attempts'] == 0


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def complete(self, messages, **kwargs):
        kwargs['before_attempt']()
        return LLMResponse(text=json.dumps(next(self.outputs)), finish_reason='stop')


def test_original_task_requirements_reach_writer_and_auditor_without_rewrite():
    captured = []
    class Capture(Model):
        def complete(self, messages, **kwargs):
            captured.append(messages)
            return super().complete(messages, **kwargs)
    history = [{'user_request': '请全程用中文，区分事实和假设。', 'role': 'initial_request'},
               {'user_request': '改为只讨论分类器，并说明局限。', 'role': 'pinned_request'}]
    draft = {**DRAFT, 'facts': [{**DRAFT['facts'][0], 'text': '该方法使用线性分类器。'}]}
    answer_then_audit('What classifier?', EVIDENCE, Capture([draft, audit()]),
                     original_question='继续', request_history=history)
    assert len(captured) == 2
    for messages in captured:
        payloads = [json.loads(m['content']) for m in messages if m['role'] == 'user']
        assert any(p.get('request_history') == history for p in payloads)
        assert 'not scientific evidence' in messages[0]['content']


def test_host_layout_runs_after_audit_without_extra_provider_attempt():
    captured = []
    draft = {**DRAFT, 'facts':[{**DRAFT['facts'][0], 'text':'该方法使用线性分类器。'}],
             'limitations':['当前证据无法确定运行时。', '当前证据无法确定内存需求。']}
    class Capture(Model):
        def complete(self, messages, **kwargs):
            captured.append(messages)
            return super().complete(messages, **kwargs)
    result = answer_then_audit('请用两点说明分类器和缺口', EVIDENCE,
        Capture([draft, audit(), {'groups':[['u1'],['u2','u3']]}]))
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['provider_attempts'] == 2
    assert result['presentation_contract']['status'] == 'SATISFIED'
    assert result['limitations_in_answer']
    assert result['answer'].startswith('1. 该方法使用线性分类器。 [e1]')
    assert '2. 当前证据无法确定运行时。' in result['answer']
    assert len(captured) == 2
    assert result['presentation_contract']['model_calls'] == 0
    assert result['presentation_contract']['point_count'] == 2


def test_changed_task_requirements_change_report_input_fingerprint():
    def report(text):
        return answer_then_audit('Q', [], None, request_history=[{'user_request': text}])
    assert report('使用中文')['input_sha256'] != report('使用英文')['input_sha256']


def test_unsupported_fact_is_retained_in_draft_not_published():
    verdict = {'assessments': [{'fact_id': 'f1', 'verdict': 'UNSUPPORTED', 'relevance': 'DIRECT'}],
               'coverage': 'PARTIAL'}
    result = answer_then_audit('What classifier?', EVIDENCE, Model([DRAFT, verdict]))
    assert result['draft'] == DRAFT
    assert result['answer'] == ''
    assert not result['all_facts_supported']
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['provider_attempts'] == 2


def test_model_audit_never_promotes_to_answered():
    result = answer_then_audit('What classifier?', EVIDENCE, Model([DRAFT, audit()]))
    assert result['all_facts_supported']
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'


def test_writer_designation_cannot_override_auditor_background_classification():
    source = [{'evidence_id': 'e1', 'content': 'A demonstration image achieved IoU 0.9.'}]
    draft = {'facts': [{'id': 'f1', 'text': 'A demonstration image achieved IoU 0.9.',
                       'evidence_ids': ['e1']}], 'direct_answer_id': 'f1'}
    review = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
               'relevance': 'NECESSARY_CONTEXT', 'span_ids': ['s0']}],
              'coverage': 'PARTIAL', 'missing_aspects': ['The benchmark score remains unknown.']}
    result = answer_then_audit('What is the benchmark IoU?', source, Model([draft, review]))
    assert result['draft']['direct_answer_id'] == 'f1'
    assert result['audit']['assessments'][0]['relevance'] == 'NECESSARY_CONTEXT'
    assert result['direct_answer_supported'] is False
    assert result['answer_scope'] == 'BACKGROUND_ONLY'


def test_reviewed_direct_answer_precedes_writer_nominated_background():
    draft = {'facts': [
        {'id': 'background', 'text': 'The method uses a linear classifier.', 'evidence_ids': ['e1']},
        {'id': 'score', 'text': 'Test F1 is 0.7.', 'evidence_ids': ['e2']}],
        'direct_answer_id': 'background'}
    review = {'assessments': [
        {'fact_id': 'background', 'verdict': 'SUPPORTED', 'relevance': 'NECESSARY_CONTEXT'},
        {'fact_id': 'score', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}], 'coverage': 'COMPLETE'}
    result = audited_presentation(draft, review)
    assert result['answer'].startswith('Test F1 is 0.7.')
    assert result['direct_answer_supported'] is True
    assert draft['direct_answer_id'] == 'background'


def test_question_scope_policy_is_recorded_and_changes_only_audit_instructions():
    from paperpilot_agentic.answer_audit import QUESTION_SCOPE_POLICY
    calls = []
    class Capture(Model):
        def complete(self, messages, **kwargs):
            calls.append(messages)
            return super().complete(messages, **kwargs)
    reports = [answer_then_audit('What classifier?', EVIDENCE, Capture([DRAFT, audit()]),
                               relation_policy=True, question_scope_policy=flag) for flag in (False, True)]
    assert calls[0] == calls[2]  # No writer change and no extra model call.
    assert calls[1][-1] == calls[3][-1]
    assert calls[3][0]['content'].replace('\n' + QUESTION_SCOPE_POLICY, '') == calls[1][0]['content']
    assert reports[0]['input_sha256'] == reports[1]['input_sha256']
    assert reports[0]['experiment_sha256'] != reports[1]['experiment_sha256']
    assert reports[1]['question_scope_policy'] is True


def test_bad_json_preserves_raw_response_and_attempt_count():
    class Broken:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text='{broken', finish_reason='stop')
    result = answer_then_audit('Question?', EVIDENCE, Broken())
    assert result['status'] == 'FAILED'
    assert result['provider_attempts'] == 2
    assert result['calls'][0]['text'] == '{broken'


def test_missing_direct_answer_field_is_repaired_by_model_not_guessed():
    broken = {'facts': DRAFT['facts']}
    result = answer_then_audit('What classifier?', EVIDENCE, Model([broken, DRAFT, audit()]))
    assert result['provider_attempts'] == 3
    assert result['draft_repair']['attempted']
    assert result['raw_draft'] == broken
    assert result['repaired_draft'] == DRAFT
    assert result['direct_answer_supported']


def test_spans_are_exact_source_offsets_and_never_cross_sources():
    sources = [{'evidence_id': 'a', 'content': 'Title\nFirst sentence. Second sentence!\nTail'},
               {'evidence_id': 'b', 'content': 'Other source.'}]
    spans = evidence_spans(sources)
    lookup = {e['evidence_id']: e['content'] for e in sources}
    assert [s['text'] for s in spans] == ['Title\nFirst sentence.', 'Second sentence!', 'Tail', 'Other source.']
    assert all(lookup[s['evidence_id']][s['start']:s['end']] == s['text'] for s in spans)
    assert len({s['span_id'] for s in spans}) == len(spans)


def test_pdf_line_wraps_do_not_expand_audit_metadata_or_lose_text():
    content = '\n'.join(['wrapped content'] * 100) + '.\n\nNext paragraph.'
    spans = evidence_spans([{'evidence_id': 'pdf', 'content': content}])
    assert len(spans) == 2
    assert all(content[s['start']:s['end']] == s['text'] for s in spans)
    assert ''.join(content.split()) == ''.join(''.join(s['text'].split()) for s in spans)


@pytest.mark.parametrize('sid', ['invented', 's1'])
def test_audit_span_binding_rejects_wrong_source_or_unknown(sid):
    value = audit()
    value['assessments'][0]['span_ids'] = [sid]
    with pytest.raises(ValueError):
        validate_audit(value, DRAFT, EVIDENCE, spans=evidence_spans(EVIDENCE))


def test_fact_list_without_designated_answer_is_rejected():
    with pytest.raises(ValueError, match='direct answer'):
        validate_draft({'facts': DRAFT['facts']}, EVIDENCE)


def test_compact_audit_view_preserves_every_source_span_exactly():
    spans = evidence_spans(EVIDENCE)
    compact = audit_source_view(spans)
    recovered = [(g['evidence_id'], sid, text) for g in compact for sid, text in g['spans'].items()]
    assert recovered == [(s['evidence_id'], s['span_id'], s['text']) for s in spans]
    assert all(isinstance(text, str) for g in compact for text in g['spans'].values())


def test_fact_span_allowlist_excludes_other_evidence():
    view = audit_facts_view(DRAFT, evidence_spans(EVIDENCE))
    assert view[0]['allowed_span_ids'] == ['s0']
    assert 'allowed_span_ids' not in DRAFT['facts'][0]


def test_auditor_real_but_uncited_source_is_quarantined_without_changing_citations():
    wrong = audit()
    wrong['assessments'][0]['span_ids'] = ['s1']
    result = answer_then_audit('Which baseline?', EVIDENCE, Model([DRAFT, wrong]))
    assert result['provider_attempts'] == 2
    assert not result.get('audit_repair')
    assert result['raw_audit']['assessments'][0]['span_ids'] == ['s1']
    assert result['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert result['draft']['facts'][0]['evidence_ids'] == ['e1']
    assert not result['answer']


def test_auditor_second_invalid_citation_fails_closed():
    wrong = audit()
    wrong['assessments'][0]['span_ids'] = ['unknown_span']
    result = answer_then_audit('Which baseline?', EVIDENCE, Model([DRAFT, wrong, wrong]))
    assert result['provider_attempts'] == 3
    assert result['status'] == 'FAILED'
    assert not result.get('answer')


@pytest.mark.parametrize('missing', ['not-a-list', {'a': 'gap'}, [42]])
def test_malformed_audit_gaps_are_not_silently_lost(missing):
    value = {**audit(), 'missing_aspects': missing}
    with pytest.raises(ValueError, match='Missing aspects'):
        validate_audit(value, DRAFT, EVIDENCE)


def test_auditor_gaps_survive_rendering_even_when_coverage_says_complete():
    draft = {**DRAFT, 'limitations': ['Missing character-level baseline']}
    verdict = {**audit(), 'missing_aspects': [' Missing character-level baseline ', 'No seed details']}
    result = answer_then_audit('Which baselines?', EVIDENCE, Model([draft, verdict]))
    assert result['limitations'][0] == 'Missing character-level baseline'
    assert 'No seed details' not in result['limitations']
    assert result['internal_gap_diagnostics']['missing_aspects'] == verdict['missing_aspects']
    assert result['limitations'][1].startswith('Not all requested work is complete')
    assert result['answer_scope'] == 'PARTIAL'
    assert result['raw_audit']['coverage'] == 'COMPLETE'


def test_rejected_direct_answer_retains_background_but_marks_unresolved():
    draft = {'facts': DRAFT['facts'] + [
        {'id': 'f2', 'text': 'Background context.', 'evidence_ids': ['e2']}],
        'direct_answer_id': 'f1'}
    verdict = {'assessments': [{'fact_id': 'f1', 'verdict': 'UNSUPPORTED'},
                               {'fact_id': 'f2', 'verdict': 'SUPPORTED'}], 'coverage': 'PARTIAL'}
    result = audited_presentation(draft, verdict)
    assert result['answer'] == 'Background context. [e2]'
    assert result['answer_scope'] == 'BACKGROUND_ONLY'
    from paperpilot_agentic.durable_reader import reader_completion_state
    assert reader_completion_state({**result, 'status': 'REVIEW_REQUIRED'})[1].value == 'UNRESOLVED'


def test_direct_answer_is_presented_before_background_without_mutating_draft():
    draft = {'facts': [{'id': 'background', 'text': 'Context.', 'evidence_ids': ['e2']}] + DRAFT['facts'],
             'direct_answer_id': 'f1'}
    verdict = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED'} for f in draft['facts']],
               'coverage': 'COMPLETE'}
    assert audited_presentation(draft, verdict)['answer'].startswith('Uses a linear classifier.')
    assert draft['facts'][0]['id'] == 'background'


def test_grounded_but_tangential_fact_is_not_published():
    verdict = {**audit(), 'assessments': [{**audit()['assessments'][0], 'relevance': 'TANGENTIAL'}]}
    result = answer_then_audit('What dataset?', EVIDENCE, Model([DRAFT, verdict]))
    assert result['answer'] == ''
    assert result['all_facts_supported']
    assert not result['direct_answer_supported']
    assert result['omitted_tangential_fact_ids'] == ['f1']
    assert result['draft'] == DRAFT


def test_current_audit_contract_does_not_default_missing_relevance_to_relevant():
    value = audit()
    del value['assessments'][0]['relevance']
    with pytest.raises(ValueError, match='relevance'):
        validate_audit(value, DRAFT, EVIDENCE, require_relevance=True)
    # Historical reports remain readable without pretending relevance was assessed.
    assert validate_audit(value, DRAFT, EVIDENCE)['coverage'] == 'COMPLETE'


def test_requested_numeric_result_survives_when_relevant():
    draft = {'facts': [{'id': 'f1', 'text': 'Accuracy is 80%.', 'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1'}
    result = audited_presentation(draft, audit())
    assert '80%' in result['answer']
    assert result['omitted_tangential_fact_ids'] == []


def test_inference_is_visibly_distinguished_without_rewriting_content():
    draft = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': '甲方法使用对齐。', 'evidence_ids': ['e1']},
        {'id': 'f2', 'kind': 'INFERENCE', 'text': '两种方法侧重点不同。',
         'evidence_ids': ['e1', 'e2'], 'depends_on': ['f1']}],
        'direct_answer_id': 'f2'}
    verdict = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED'} for f in draft['facts']],
               'coverage': 'COMPLETE'}
    validate_draft(draft, EVIDENCE)
    result = audited_presentation(draft, verdict)
    assert result['answer'].startswith('比较推断：两种方法侧重点不同。 [e1] [e2]')
    assert '原文事实：甲方法使用对齐。 [e1]' in result['answer']
    assert draft['facts'][0]['text'] == '甲方法使用对齐。'
    draft['facts'][1]['kind'] = 'CERTIFIED_TRUTH'
    with pytest.raises(ValueError, match='kind'):
        validate_draft(draft, EVIDENCE)


def test_rejected_comparison_does_not_hide_supported_direct_subanswers():
    draft = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'A uses alignment.', 'evidence_ids': ['e1']},
        {'id': 'f2', 'kind': 'INFERENCE', 'text': 'B requires aligned inputs.',
         'evidence_ids': ['e2'], 'depends_on': ['f1']}],
        'direct_answer_id': 'f2'}
    verdict = {'assessments': [
        {'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'},
        {'fact_id': 'f2', 'verdict': 'UNSUPPORTED', 'relevance': 'DIRECT'}],
        'coverage': 'PARTIAL', 'missing_aspects': ['Comparison premise not established.']}
    result = audited_presentation(draft, verdict)
    assert result['answer_scope'] == 'PARTIAL'
    assert result['direct_answer_supported']
    assert result['answer'] == 'Source fact: A uses alignment. [e1]'
    assert result['limitations'] == verdict['missing_aspects']
    verdict['assessments'][0]['relevance'] = 'NECESSARY_CONTEXT'
    assert audited_presentation(draft, verdict)['answer_scope'] == 'BACKGROUND_ONLY'


def test_quantity_policy_applies_to_writer_and_auditor_and_can_be_ablated():
    from paperpilot_agentic.answer_audit import QUANTITY_POLICY
    class CapturingModel(Model):
        def __init__(self):
            super().__init__([DRAFT, audit()])
            self.prompts = []
        def complete(self, messages, **kwargs):
            self.prompts.append(messages[0]['content'])
            return super().complete(messages, **kwargs)
    enabled = CapturingModel()
    answer_then_audit('How many?', EVIDENCE, enabled, quantity_policy=True)
    assert len(enabled.prompts) == 2
    assert all(QUANTITY_POLICY in p for p in enabled.prompts)
    baseline = CapturingModel()
    answer_then_audit('How many?', EVIDENCE, baseline, quantity_policy=False)
    assert all(QUANTITY_POLICY not in p for p in baseline.prompts)


def test_question_intent_and_user_limitations_are_requested_without_bypassing_audit():
    class CapturingModel(Model):
        prompts = None
        def complete(self, messages, **kwargs):
            if self.prompts is None:
                self.prompts = []
            self.prompts.append(messages[0]['content'])
            return super().complete(messages, **kwargs)
    verdict = audit()
    verdict['assessments'][0]['verdict'] = 'UNSUPPORTED'
    model = CapturingModel([DRAFT, verdict])
    report = answer_then_audit('What is the classifier?', EVIDENCE, model)
    assert 'definition rather than an exhaustive experimental list' in model.prompts[0]
    assert 'not fact IDs' in model.prompts[1]
    assert 'do not relax factual support' in model.prompts[1]
    assert report['answer'] == ''
    assert not report['direct_answer_supported']


def test_cross_fact_span_error_names_exact_failed_field_and_allowed_choices():
    invalid = audit()
    invalid['assessments'][0]['span_ids'] = ['s1']
    with pytest.raises(ValueError, match="fact_id=f1, invalid_span=s1, allowed_span_ids=\\['s0'\\]"):
        validate_audit(invalid, DRAFT, EVIDENCE, spans=evidence_spans(EVIDENCE), require_relevance=True)


def test_original_followup_wording_reaches_both_writer_and_auditor():
    class CapturingModel(Model):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            assert payload['question'] == 'What classifier does this model use?'
            assert payload['original_question'] == '它用什么分类器？'
            assert 'Use its language' in messages[0]['content']
            return super().complete(messages, **kwargs)
    result = answer_then_audit('What classifier does this model use?', EVIDENCE,
        CapturingModel([DRAFT, audit()]), original_question='它用什么分类器？')
    assert result['original_question'] == '它用什么分类器？'


def test_uncited_limitations_produce_actionable_repair_not_fabricated_citations():
    bad = {'facts': [{'id': 'boundary', 'text': 'Performance cannot be established.',
                     'evidence_ids': []}], 'direct_answer_id': 'boundary'}
    class RepairModel(Model):
        count = 0
        def complete(self, messages, **kwargs):
            self.count += 1
            if self.count == 2:
                assert 'fact_id=boundary' in messages[-1]['content']
                assert 'move it to limitations' in messages[-1]['content']
            return super().complete(messages, **kwargs)
    result = answer_then_audit('Can we infer performance?', EVIDENCE, RepairModel([
        bad, {'facts': [], 'limitations': ['Performance cannot be established.']},
        {'assessments': [], 'coverage': 'UNANSWERABLE', 'missing_aspects': []}]))
    assert result['answer'] == ''
    assert result['limitations'] == ['Performance cannot be established.']
    assert 'fact_id=boundary' in result['draft_repair']['validation_error']


def test_auditor_receives_limitations_for_full_answer_coverage():
    draft = {**DRAFT, 'limitations': ['No comparative performance evidence is supplied.']}
    class CapturingModel(Model):
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                payload = json.loads(messages[-1]['content'])
                assert payload['limitations'] == draft['limitations']
                assert 'Limitations remain untrusted' in messages[0]['content']
            return super().complete(messages, **kwargs)
    report = answer_then_audit('Method and limits?', EVIDENCE, CapturingModel([draft, audit()]))
    assert report['limitations'] == draft['limitations']


def test_audit_requests_reason_before_verdict_without_overriding_rejection():
    rejected = audit()
    rejected['assessments'][0]['verdict'] = 'UNSUPPORTED'
    class CapturingModel(Model):
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                schema = messages[0]['content'].split('Return JSON: ')[1]
                assert schema.index('"reason"') < schema.index('"verdict"')
                assert 'two concise sentences' in schema
            return super().complete(messages, **kwargs)
    report = answer_then_audit('Question?', EVIDENCE, CapturingModel([DRAFT, rejected]))
    assert report['answer'] == ''


def test_frozen_audit_uses_one_call_and_does_not_modify_candidate():
    class CapturingModel(Model):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            assert len(payload['sources']) == 1
            assert payload['sources'][0]['evidence_id'] == 'e1'
            return super().complete(messages, **kwargs)
    before = json.dumps(DRAFT, sort_keys=True)
    report = answer_then_audit('Question?', EVIDENCE, CapturingModel([audit()]),
                              frozen_draft=DRAFT, audit_cited_only=True)
    assert report['provider_attempts'] == 1
    assert report['draft'] == DRAFT
    assert json.dumps(DRAFT, sort_keys=True) == before
    assert report['audit_context']['omitted_evidence_ids'] == ['e2']
    assert report['writer_mode'] == 'frozen-draft-diagnostic'


def test_invalid_frozen_candidate_fails_before_any_request():
    with pytest.raises(ValueError, match='supplied evidence'):
        answer_then_audit('Question?', EVIDENCE, None,
                         frozen_draft={'facts': [{'id': 'bad', 'text': 'X', 'evidence_ids': []}]})


@pytest.mark.parametrize('supported', [False, True])
def test_exclusivity_review_preserves_evidence_and_obeys_auditor(supported):
    # Synthetic domain, not a replay of the paid QASPER question. The scripted
    # verdict verifies wiring/publication only, not real-model entailment ability.
    content = ('Every retained sample is sterile.' if supported else
               'Batches with predominantly contaminated samples were removed.')
    evidence = [{'evidence_id': 'e1', 'content': content}]
    draft = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                       'text': 'Every retained sample is sterile.', 'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1'}
    verdict = {'assessments': [{'fact_id': 'f1', 'relevance': 'DIRECT',
                               'verdict': 'SUPPORTED' if supported else 'UNSUPPORTED',
                               'span_ids': ['s0'] if supported else []}],
               'coverage': 'COMPLETE' if supported else 'PARTIAL',
               'missing_aspects': [] if supported else ['Sterility of all retained samples is unknown.']}

    class CapturingModel(Model):
        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            system = messages[0]['content']
            if self.calls == 1:
                assert 'normally use 1-3' in system
                assert 'not a hard cap' in system
                assert 'Preserve quantifier strength' in system
                assert json.loads(messages[-1]['content'])['evidence'][0]['content'] == content
            else:
                assert 'even if labeled INFERENCE' in system
                assert 'zero X items remain' in system
            return super().complete(messages, **kwargs)

    report = answer_then_audit('Are all retained samples sterile?', evidence,
                              CapturingModel([draft, verdict]))
    assert report['provider_attempts'] == 2
    assert report['draft']['facts'][0]['text'] == draft['facts'][0]['text']
    assert bool(report['answer']) is supported
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['mode'] == 'answer-first-shared-packet-v2-repair-and-gap-display'
