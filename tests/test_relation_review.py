"""Qualifying relations need explicit review, not just a topic-matching verdict."""
import json
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


def test_default_audit_cannot_approve_stage_and_subject_without_review():
    source = [{'evidence_id': 'e', 'content':
        'Each stage of the reconstruction decoder contains an AlignGate module. '
        'Unlike conventional registration based on pixel similarity, AlignGate estimates flow. '
        'OffsetNet lacks an explicit alignment-quality constraint during training.'}]
    draft = {'facts': [
        {'id': 'stage', 'kind': 'SOURCE_FACT', 'text':
         'AlignGate 在解码前通过光流补偿错位。', 'evidence_ids': ['e']},
        {'id': 'subject', 'kind': 'SOURCE_FACT', 'text':
         '传统配准方法缺乏显式对齐质量约束。', 'evidence_ids': ['e']}],
        'direct_answer_id': 'stage', 'limitations': []}

    class BlindAudit:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            value = {'assessments': [
                {'fact_id': 'stage', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                 'span_ids': ['s0'], 'reason': 'The source describes flow alignment.'},
                {'fact_id': 'subject', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                 'span_ids': ['s2'], 'reason': 'Prior methods lack alignment constraints.'}],
                'coverage': 'COMPLETE', 'missing_aspects': []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop')

    report = answer_then_audit('说明该模块的位置和配准问题。', source, BlindAudit(),
        frozen_draft=draft, relation_policy=True, allow_repair=False)
    assert not report['all_facts_supported']
    assert not report['answer']
    assert report['provider_attempts'] == 1


@pytest.mark.parametrize('text', [
    'AlignGate 在重建解码器每个阶段对齐特征。',
    'AlignGate operates within each decoder stage.',
])
def test_correct_pipeline_relation_survives_actual_writer_and_auditor(text):
    captured = []
    source = [{'evidence_id': 'e', 'content':
               'Each stage of the reconstruction decoder contains an AlignGate module.'}]

    class CorrectModel:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            captured.append(messages)
            payload = json.loads(messages[-1]['content'])
            if len(captured) == 1:
                assert 'mechanism_relations' in payload['output_contract']
                value = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT', 'text': text,
                                   'evidence_ids': ['e']}], 'direct_answer_id': 'f', 'limitations': []}
            else:
                target = payload['relation_review_targets'][0]
                assert target['dimension'] == 'pipeline_position'
                assert 'relation_checks' in messages[0]['content']
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': ['s0']}], 'coverage': 'COMPLETE',
                    'relation_checks': [{'target_id': target['target_id'], 'span_ids': ['s0'],
                        'reason': 'Each decoder stage contains the module, matching the stated position.',
                        'verdict': 'SUPPORTED'}]}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')

    report = answer_then_audit('Describe its pipeline position.', source, CorrectModel(), relation_policy=True)
    assert text in report['answer']
    assert report['all_facts_supported']
    assert report['relation_review']['changed_fact_ids'] == []
    assert report['provider_attempts'] == 2


def relation_fixture():
    from paperpilot_agentic.answer_audit import evidence_spans
    source = [{'evidence_id': 'e', 'content': 'OffsetNet lacks an alignment-quality constraint.'},
              {'evidence_id': 'other', 'content': 'Other methods use a different objective.'}]
    draft = {'facts': [{'id': 'bad', 'kind': 'SOURCE_FACT',
                       'text': '传统配准方法缺乏显式质量约束。', 'evidence_ids': ['e']},
                      {'id': 'good', 'kind': 'SOURCE_FACT', 'text': 'OffsetNet 缺少质量约束。',
                       'evidence_ids': ['e']}]}
    audit = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED', 'span_ids': ['s0'],
                             'relevance': 'DIRECT'} for f in draft['facts']], 'coverage': 'COMPLETE'}
    return draft, audit, evidence_spans(source)


@pytest.mark.parametrize('defect', ['missing', 'foreign', 'unknown', 'duplicate', 'bad_shape', 'empty_reason'])
def test_invalid_relation_witness_withholds_only_the_affected_fact(defect):
    from paperpilot_agentic.relation_review import apply_relation_review
    draft, audit, spans = relation_fixture()
    check = {'target_id': 'bad:limitation_owner', 'verdict': 'SUPPORTED',
             'span_ids': ['s0'], 'reason': 'Explicitly reviewed the source owner.'}
    audit['relation_checks'] = [check]
    if defect == 'missing':
        audit['relation_checks'] = []
    elif defect == 'foreign':
        check['span_ids'] = ['s1']
    elif defect == 'unknown':
        check['span_ids'] = ['invented']
    elif defect == 'duplicate':
        audit['relation_checks'].append(deepcopy(check))
    elif defect == 'bad_shape':
        check['span_ids'] = [{'span_id': 's0'}]
    elif defect == 'empty_reason':
        check['reason'] = ''
    before = deepcopy((draft, audit))
    actual, record = apply_relation_review(draft, audit, spans, enabled=True)
    assert [a['verdict'] for a in actual['assessments']] == ['UNCERTAIN', 'SUPPORTED']
    assert actual['coverage'] == 'PARTIAL'
    assert record['changed_fact_ids'] == ['bad']
    assert (draft, audit) == before


def test_negative_subrelation_overrules_topic_support_and_positive_check_never_upgrades():
    from paperpilot_agentic.relation_review import apply_relation_review
    draft, audit, spans = relation_fixture()
    audit['relation_checks'] = [{'target_id': 'bad:limitation_owner', 'span_ids': ['s0'],
        'reason': 'Only OffsetNet has the limitation, not the whole class.', 'verdict': 'UNSUPPORTED'}]
    actual, _ = apply_relation_review(draft, audit, spans, enabled=True)
    assert actual['assessments'][0]['verdict'] == 'UNSUPPORTED'
    audit['assessments'][0]['verdict'] = 'UNCERTAIN'
    audit['relation_checks'][0]['verdict'] = 'SUPPORTED'
    actual, _ = apply_relation_review(draft, audit, spans, enabled=True)
    assert actual['assessments'][0]['verdict'] == 'UNCERTAIN'


@pytest.mark.parametrize('text', [
    '传统配准依赖像素相似性。',
    'OffsetNet 缺乏质量约束。',
    '不能证明传统方法缺乏显式约束。',
    'This comparison cannot establish that conventional methods lack constraints.',
    'The module estimates flow to align features.',
])
def test_unqualified_and_explicitly_bounded_claims_do_not_get_spurious_targets(text):
    from paperpilot_agentic.relation_review import review_targets
    assert review_targets({'facts': [{'id': 'f', 'text': text}]}) == []


def test_unrelated_caveat_cannot_disable_a_later_positive_limitation_assertion():
    from paperpilot_agentic.relation_review import review_targets
    text = '不能证明性能更好。传统配准方法缺乏质量约束。'
    assert review_targets({'facts': [{'id': 'f', 'text': text}]}) == [
        {'target_id': 'f:limitation_owner', 'fact_id': 'f', 'dimension': 'limitation_owner'}]
