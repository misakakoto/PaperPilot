"""Public answer/audit contract: revisions are not presumed monotonic or true."""
import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from test_answer_audit import Model


SOURCES = [{'evidence_id': 'a', 'content': 'Orion is evaluated on DeltaSet.'},
           {'evidence_id': 'b', 'content': 'Orion is evaluated on BetaSet.'}]


def fact(fid, text, eid):
    return {'id': fid, 'text': text, 'evidence_ids': [eid]}


def assessment(fid, sid, verdict='SUPPORTED'):
    return {'fact_id': fid, 'verdict': verdict, 'relevance': 'DIRECT', 'span_ids': [sid]}


def report(new, review, *, old=None):
    old = old or fact('old', 'Orion is evaluated on DeltaSet.', 'a')
    return answer_then_audit('List the datasets Orion evaluates on.', SOURCES, Model([review]),
        frozen_draft={'facts': new, 'direct_answer_id': new[0]['id']}, allow_repair=False,
        revision_feedback={'draft': {'facts': [old]}, 'audit': {
            'assessments': [assessment('old', 's0')], 'coverage': 'PARTIAL'}})


def test_faithful_paraphrase_is_accounted_for_only_after_fresh_source_audit():
    review = {'assessments': [assessment('rewritten', 's0'), assessment('added', 's1')],
        'coverage': 'COMPLETE', 'revision_resolutions': [{'previous_fact_id': 'old',
            'status': 'RETAINED', 'current_fact_ids': ['rewritten'], 'source_span_ids': ['s0'],
            'reason': 'Both statements identify the same evaluation dataset.'}]}
    result = report([fact('rewritten', 'DeltaSet is an evaluation dataset for Orion.', 'a'),
                     fact('added', 'Orion is evaluated on BetaSet.', 'b')], review)
    assert result['answer_scope'] == 'MODEL_REVIEWED'
    assert 'DeltaSet' in result['answer'] and 'BetaSet' in result['answer']
    assert not result['revision_accounting']['unresolved_previous_fact_ids']


def test_prior_positive_verdict_does_not_require_preserving_a_wrong_total():
    review = {'assessments': [assessment('delta', 's0'), assessment('beta', 's1')],
        'coverage': 'COMPLETE', 'revision_resolutions': [{'previous_fact_id': 'old',
            'status': 'CORRECTED', 'current_fact_ids': ['delta', 'beta'], 'source_span_ids': ['s0', 's1'],
            'reason': 'Two named evaluations contradict the old exclusive total.'}]}
    result = report([fact('delta', SOURCES[0]['content'], 'a'), fact('beta', SOURCES[1]['content'], 'b')], review,
                    old=fact('old', 'Orion is evaluated only on DeltaSet.', 'a'))
    assert result['answer_scope'] == 'MODEL_REVIEWED'
    assert 'only' not in result['answer']


@pytest.mark.parametrize('change', ['rejected_fact', 'wrong_original', 'missing_resolution'])
def test_resolution_cannot_override_current_audit_or_hide_unaccounted_content(change):
    review = {'assessments': [assessment('delta', 's0'), assessment('beta', 's1')],
        'coverage': 'COMPLETE', 'revision_resolutions': [{'previous_fact_id': 'old',
            'status': 'RETAINED', 'current_fact_ids': ['delta'], 'source_span_ids': ['s0'],
            'reason': 'The replacement covers the evaluation dataset.'}]}
    if change == 'rejected_fact':
        review['assessments'][0]['verdict'] = 'UNSUPPORTED'
    elif change == 'wrong_original':
        review['revision_resolutions'][0]['source_span_ids'] = ['s1']
    else:
        review.pop('revision_resolutions')
    result = report([fact('delta', 'Orion evaluated DeltaSet.', 'a'), fact('beta', SOURCES[1]['content'], 'b')], review)
    assert result['answer_scope'] == 'PARTIAL'
    assert result['revision_accounting']['unresolved_previous_fact_ids'] == ['old']
    assert result['limitations']


def test_source_bound_withdrawal_is_allowed_but_not_an_independent_truth_verdict():
    review = {'assessments': [assessment('beta', 's1')], 'coverage': 'COMPLETE',
        'revision_resolutions': [{'previous_fact_id': 'old', 'status': 'WITHDRAWN',
            'current_fact_ids': [], 'source_span_ids': ['s1'],
            'reason': 'The old exclusive statement conflicts with this original evaluation.'}]}
    result = report([fact('beta', SOURCES[1]['content'], 'b')], review,
                    old=fact('old', 'Orion is evaluated only on DeltaSet.', 'a'))
    assert result['revision_accounting']['resolutions'][0]['effective_status'] == 'WITHDRAWN'
    assert result['semantic_correctness'] == 'MODEL_AUDIT_ONLY_NOT_INDEPENDENT_GROUND_TRUTH'
    assert 'only' not in result['answer']


@pytest.mark.parametrize('field,value', [
    ('previous_fact_id', 'unknown'), ('current_fact_ids', ['unknown']),
    ('source_span_ids', ['unknown']), ('source_span_ids', []), ('reason', ''), ('status', 'VERIFIED')])
def test_invalid_revision_receipt_cannot_authorize_a_complete_answer(field, value):
    receipt = {'previous_fact_id': 'old', 'status': 'RETAINED', 'current_fact_ids': ['new'],
               'source_span_ids': ['s0'], 'reason': 'Same requested content.'}
    receipt[field] = value
    result = report([fact('new', SOURCES[0]['content'], 'a')], {
        'assessments': [assessment('new', 's0')], 'coverage': 'COMPLETE', 'revision_resolutions': [receipt]})
    assert result['status'] == 'FAILED'
    assert not result.get('answer')


@pytest.mark.parametrize('ids', [['new'], ['new', 'unknown'], ['new', 'new']])
def test_invalid_conflict_references_are_rejected_not_applied_to_arbitrary_facts(ids):
    result = report([fact('new', SOURCES[0]['content'], 'a')], {
        'assessments': [assessment('new', 's0')], 'coverage': 'COMPLETE',
        'answer_conflicts': [{'fact_ids': ids, 'reason': 'Conflict'}]})
    assert result['status'] == 'FAILED'


def test_conflict_downgrade_propagates_to_dependent_inference():
    facts = [fact('delta', 'Orion uses only DeltaSet.', 'a'), fact('beta', SOURCES[1]['content'], 'b'),
             {**fact('inference', 'Orion uses only one dataset.', 'a'), 'kind': 'INFERENCE',
              'depends_on': ['delta']}]
    result = answer_then_audit('List datasets.', SOURCES, Model([{
        'assessments': [assessment('delta', 's0'), assessment('beta', 's1'), assessment('inference', 's0')],
        'coverage': 'COMPLETE', 'answer_conflicts': [{'fact_ids': ['delta', 'beta'],
            'reason': 'An exclusive total contradicts another evaluation.'}]}]),
        frozen_draft={'facts': facts, 'direct_answer_id': 'delta'})
    assert not result['answer']
    assert not result['direct_answer_supported']


def test_missing_model_conflict_does_not_publish_closed_dataset_count_and_extra_use():
    from paperpilot_agentic.answer_integrity import quarantine_answer_conflicts

    draft = {'facts': [
        fact('total', 'ChangeMamba 论文使用了 SYSU-CD、LEVIR-CD +、WHU-CD 和 SECOND 四个数据集。', 'a'),
        fact('extra', 'ChangeMamba 论文还使用 xBD 数据集进行建筑物损伤评估。', 'b')],
        'direct_answer_id': 'total'}
    audit = {'assessments': [assessment('total', 's0'), assessment('extra', 's1')],
             'coverage': 'COMPLETE', 'missing_aspects': [], 'answer_conflicts': []}
    effective, diagnostic = quarantine_answer_conflicts(audit, draft, [], question='论文使用哪些数据集？')
    assert set(diagnostic['changed_fact_ids']) == {'total', 'extra'}
    assert {row['verdict'] for row in effective['assessments']} == {'UNCERTAIN'}
    assert effective['coverage'] == 'PARTIAL'


@pytest.mark.parametrize('count_text,extra_text', [
    ('ChangeMamba 论文例如使用了 SYSU-CD、LEVIR-CD 和 SECOND 三个数据集。',
     'ChangeMamba 论文还使用 xBD 数据集。'),
    ('ChangeMamba 论文使用了 SYSU-CD、LEVIR-CD 和 SECOND 三个光学数据集。',
     'ChangeMamba 论文还使用 xBD 数据集。'),
    ('ChangeMamba 论文使用了 SYSU-CD、LEVIR-CD 和 SECOND 三个数据集。',
     'FlowMamba 论文还使用 xBD 数据集。'),
    ('ChangeMamba 论文使用了 SYSU-CD、LEVIR-CD 和 SECOND 三个数据集。',
     'ChangeMamba 论文还使用 SECOND 数据集。'),
])
def test_closed_dataset_guard_avoids_examples_scopes_and_other_papers(count_text, extra_text):
    from paperpilot_agentic.answer_integrity import quarantine_answer_conflicts

    draft = {'facts': [fact('total', count_text, 'a'), fact('extra', extra_text, 'b')]}
    audit = {'assessments': [assessment('total', 's0'), assessment('extra', 's1')],
             'coverage': 'COMPLETE', 'missing_aspects': [], 'answer_conflicts': []}
    effective, diagnostic = quarantine_answer_conflicts(audit, draft, [], question='论文使用哪些数据集？')
    assert not diagnostic['deterministic_conflicts']
    assert effective['coverage'] == 'COMPLETE'
    assert all(row['verdict'] == 'SUPPORTED' for row in effective['assessments'])


def test_frozen_answer_pipeline_keeps_model_missed_count_conflict_unpublished():
    total = 'ChangeMamba 论文使用了 SYSU-CD、LEVIR-CD +、WHU-CD 和 SECOND 四个数据集。'
    extra = 'ChangeMamba 论文还使用 xBD 数据集进行建筑物损伤评估。'
    evidence = [{'evidence_id': 'a', 'content': total}, {'evidence_id': 'b', 'content': extra}]
    draft = {'facts': [fact('total', total, 'a'), fact('extra', extra, 'b')],
             'direct_answer_id': 'total'}
    review = {'assessments': [assessment('total', 's0'), assessment('extra', 's1')],
              'coverage': 'COMPLETE', 'missing_aspects': [], 'answer_conflicts': []}
    result = answer_then_audit('ChangeMamba 论文使用哪些数据集？', evidence, Model([review]),
                               frozen_draft=draft, allow_repair=False, numeric_scope_mode='off')
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['answer_consistency']['changed_fact_ids'] == ['total', 'extra']
    assert not result['answer']
    assert result['audit']['coverage'] == 'PARTIAL'
