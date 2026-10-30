"""A revision cannot repeat a host-rejected absence claim unchanged."""
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_boundaries import enforce_answer_boundaries
from paperpilot_agentic.reader_recovery import revision_feedback, validate_boundary_revision


SOURCE = {'evidence_id': 'a', 'content': 'Alpha rearranges two feature sequences.',
          'source_uri': 'paper://sha/method'}
TEXT = 'Alpha rearranges sequences but does not provide geometric alignment.'


def rejected_report():
    draft = {'facts': [{'id': 'm', 'kind': 'SOURCE_FACT', 'text': TEXT,
                       'evidence_ids': ['a'], 'depends_on': []},
                      {'id': 'i', 'kind': 'INFERENCE', 'text': 'Beta is more direct.',
                       'evidence_ids': ['a'], 'depends_on': ['m']}],
             'direct_answer_id': 'i', 'limitations': []}
    model_audit = {'assessments': [{'fact_id': fid, 'verdict': 'SUPPORTED'}
                                  for fid in ['m', 'i']], 'coverage': 'COMPLETE'}
    audit, boundaries = enforce_answer_boundaries(draft, model_audit, [SOURCE])
    return {'draft': draft, 'audit': audit, 'answer_boundaries': boundaries}


def test_actual_host_rejection_requires_a_changed_claim_before_reaudit():
    report = rejected_report()
    assert report['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'
    feedback = revision_feedback(report, [SOURCE])
    candidate = deepcopy(report['draft'])
    # Changing the fact ID does not change the unsupported claim.
    candidate['facts'][0]['id'] = 'new'
    with pytest.raises(ValueError, match='Unchanged host-rejected absence claim'):
        validate_boundary_revision(candidate, feedback, [SOURCE])


def test_positive_rewrite_is_allowed_but_not_approved_by_progress_check():
    report = rejected_report()
    feedback = revision_feedback(report, [SOURCE])
    candidate = deepcopy(report['draft'])
    candidate['facts'][0]['text'] = 'Alpha rearranges two feature sequences.'
    before = deepcopy(candidate)
    validate_boundary_revision(candidate, feedback, [SOURCE])
    assert candidate == before
    assert feedback['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'


def test_new_exact_source_support_can_retain_the_previously_rejected_text():
    report = rejected_report()
    new_source = {**SOURCE, 'evidence_id': 'new', 'content': TEXT}
    candidate = deepcopy(report['draft'])
    candidate['facts'][0]['evidence_ids'] = ['new']
    validate_boundary_revision(candidate, revision_feedback(report, [SOURCE]), [new_source])


def test_ordinary_revision_and_missing_historical_feedback_are_compatible():
    candidate = rejected_report()['draft']
    validate_boundary_revision(candidate, None, [SOURCE])
    validate_boundary_revision(candidate, {'draft': candidate}, [SOURCE])


def test_writer_uses_its_existing_single_repair_before_audit():
    from paperpilot_agentic.answer_pipeline import AnswerOptions, initialize, generate_draft
    report = rejected_report()
    context = initialize('Compare the mechanisms.', [SOURCE], object(),
                         AnswerOptions(revision_feedback=revision_feedback(report, [SOURCE])), None)
    calls = []

    def call(messages, max_tokens, **kwargs):
        calls.append(kwargs)
        draft = deepcopy(report['draft'])
        if len(calls) == 2:
            assert kwargs['repair'] is True
            assert 'Unchanged host-rejected absence claim' in messages[-1]['content']
            draft['facts'][0]['text'] = 'Alpha rearranges two feature sequences.'
        return draft

    result = generate_draft(context, call)
    assert len(calls) == 2
    assert result['facts'][0]['text'] == SOURCE['content']
    assert 'audit' not in context.report, 'Repaired prose still requires independent audit'


def test_repeated_failed_repair_cannot_open_another_retry_loop():
    from paperpilot_agentic.answer_pipeline import AnswerOptions, initialize, generate_draft
    report = rejected_report()
    context = initialize('Compare the mechanisms.', [SOURCE], object(),
                         AnswerOptions(revision_feedback=revision_feedback(report, [SOURCE])), None)
    calls = []

    def call(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(report['draft'])

    with pytest.raises(ValueError, match='Unchanged host-rejected absence claim'):
        generate_draft(context, call)
    assert len(calls) == 2


def planned_report():
    report = rejected_report()
    report['model_audit'] = {'coverage': 'COMPLETE', 'assessments': [
        {'fact_id': fid, 'verdict': 'SUPPORTED'} for fid in ['m', 'i']]}
    report['audit']['limitation_assessments'] = []
    report['source_spans'] = [{'span_id': 's0', 'evidence_id': 'a', 'text': SOURCE['content']}]
    return report


def test_only_host_absence_rejection_uses_existing_revision_without_planning_call():
    from paperpilot_agentic.reader_recovery import plan_evidence_reload
    report = planned_report()
    before = deepcopy(report)
    # No model is available: invoking it would fail this test.
    plan = plan_evidence_reload('Compare.', report, [SOURCE], {'a'}, None)
    assert plan['action'] == 'REVISE' and plan['provider_attempts'] == 0
    assert plan['boundary_repair_fact_ids'] == ['i', 'm']
    assert plan['revision_validation'] == 'SOURCE_BOUND_NOT_FACT_VERIFIED'
    assert report == before


@pytest.mark.parametrize('failure', ['missing_content', 'missing_source', 'changed_source',
                                    'other_guard', 'request_gap'])
def test_real_content_or_source_gaps_cannot_use_the_host_absence_shortcut(failure):
    from paperpilot_agentic.reader_recovery import boundary_revision_plan
    report = planned_report()
    evidence = [SOURCE]
    if failure == 'missing_content':
        report['model_audit']['missing_aspects'] = ['Requested quantity missing']
    elif failure == 'missing_source':
        evidence = []
    elif failure == 'changed_source':
        evidence = [{**SOURCE, 'content': 'Changed original.'}]
    elif failure == 'other_guard':
        report['audit']['assessments'][0]['boundary_reason'] = 'invalid_numeric_binding'
    else:
        report['draft']['limitations'] = ['A requested mechanism is missing']
        report['audit']['limitation_assessments'] = [{'role': 'REQUEST_GAP'}]
    assert boundary_revision_plan(report, evidence) is None


@pytest.mark.parametrize('quote', [None, '', 'Alpha cannot handle displacement.'])
def test_reviewer_cannot_reject_an_inference_using_a_removed_assertion(quote):
    from paperpilot_agentic.inference_review import validate_disputed_claims
    draft = {'facts': [{'id': 'i', 'kind': 'INFERENCE',
                       'text': 'Beta estimates displacement; Alpha rearranges sequences.'}]}
    audit = {'assessments': [{'fact_id': 'i', 'verdict': 'UNCERTAIN', 'disputed_quote': quote}]}
    with pytest.raises(ValueError, match='CURRENT'):
        validate_disputed_claims(audit, draft)


def test_actual_unsupported_current_clause_can_still_be_rejected():
    from paperpilot_agentic.inference_review import validate_disputed_claims
    draft = {'facts': [{'id': 'i', 'kind': 'INFERENCE', 'text': 'Beta is always more accurate.'}]}
    audit = {'assessments': [{'fact_id': 'i', 'verdict': 'UNSUPPORTED',
                              'disputed_quote': 'always more accurate'}]}
    before = deepcopy(audit)
    validate_disputed_claims(audit, draft)
    assert audit == before


def test_author_ranking_requirement_is_invalid_but_does_not_promote_the_inference():
    from paperpilot_agentic.inference_review import validate_derivation_basis
    draft = {'facts': [{'id': 'i', 'kind': 'INFERENCE', 'text': 'Alpha addresses coordinates directly.'}]}
    audit = {'assessments': [{'fact_id': 'i', 'verdict': 'UNSUPPORTED',
        'reason': "The sources do not explicitly rank the designs by directness or define directness as a metric."}]}
    before = deepcopy(audit)
    with pytest.raises(ValueError, match='Invalid inference review criterion'):
        validate_derivation_basis(audit, draft)
    assert audit == before


def test_real_unsupported_causal_step_remains_rejected():
    from paperpilot_agentic.inference_review import validate_derivation_basis
    draft = {'facts': [{'id': 'i', 'kind': 'INFERENCE', 'text': 'Warping guarantees perfect accuracy.'}]}
    audit = {'assessments': [{'fact_id': 'i', 'verdict': 'UNSUPPORTED',
                              'reason': 'A geometric operation does not imply perfect measured accuracy.'}]}
    validate_derivation_basis(audit, draft)
    assert audit['assessments'][0]['verdict'] == 'UNSUPPORTED'


def test_exhausted_review_repair_keeps_supported_premises_instead_of_failing_whole_answer():
    from paperpilot_agentic.answer_pipeline import initialize, AnswerOptions, audit_draft
    context = initialize('Compare mechanisms.', [SOURCE], object(), AnswerOptions(), None)
    context.report['draft_repair'] = {'attempted': True}
    draft = {'facts': [
        {'id': 'p', 'kind': 'SOURCE_FACT', 'text': SOURCE['content'], 'evidence_ids': ['a'], 'depends_on': []},
        {'id': 'i', 'kind': 'INFERENCE', 'text': 'Alpha addresses sequence organization directly.',
         'evidence_ids': ['a'], 'depends_on': ['p']}], 'direct_answer_id': 'i', 'limitations': []}
    calls = []

    def call(*args, **kwargs):
        calls.append(kwargs)
        return {'assessments': [
            {'fact_id': 'p', 'verdict': 'SUPPORTED', 'span_ids': ['s0'], 'relevance': 'DIRECT', 'reason': 'Exact source.'},
            {'fact_id': 'i', 'verdict': 'UNCERTAIN', 'span_ids': ['s0'], 'relevance': 'DIRECT',
             'reason': 'The sources do not explicitly rank the designs by directness.'}],
            'coverage': 'PARTIAL', 'missing_aspects': ['Comparison remains unresolved'], 'omitted_answer_spans': []}

    audit, _, _ = audit_draft(context, draft, call)
    assert len(calls) == 1, 'No additional repair credit'
    assert audit['assessments'][0]['verdict'] == 'SUPPORTED'
    assert audit['assessments'][1]['verdict'] != 'SUPPORTED'
    assert audit['coverage'] == 'PARTIAL'
