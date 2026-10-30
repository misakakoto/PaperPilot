import copy
import json

import pytest

from paperpilot_agentic.experimental_scope import apply_experimental_scope, source_roles
from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


def observation(eid='evidence_abcdefghijklmnopqrstuvwxyz'):
    source = SourceRegion(paper_sha256='paper', object_id='table', label='Table I', page=1,
                          bbox=[1, 1, 10, 10], crop_sha256='crop')
    artifact = ParsedArtifact(kind='table', source=source, parser='fixture', parser_version='1',
        config_sha256='config', parse_status='PARSED', validation_status='VALIDATED',
        table_rows=[{'cells': {'F1': '81.11'}}])
    return {'evidence_id': eid, 'source_uri': 'paper://paper/layout/table', 'content': 'Table I. F1 81.11.',
            'provenance': {'structured_extraction': artifact.model_dump()}}


def assessment(text, roles=None):
    draft = {'facts': [{'id': 'f1', 'text': text, 'evidence_ids': ['t']}]}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED'}], 'coverage': 'COMPLETE'}
    before = copy.deepcopy(audit)
    result, diagnostics = apply_experimental_scope(draft, audit, roles or {'t': {'kind': 'table'}})
    assert audit == before
    return result, diagnostics


@pytest.mark.parametrize('text,reason', [
    ('Cross 将 token 交错排列。', 'mechanism_needs_method_passage'),
    ('A concatenates features across channels.', 'mechanism_needs_method_passage'),
    ('组合最大化联合建模效果。', 'conclusion_exceeds_tested_configurations'),
    ('该结果保证提升精度。', 'experimental_rows_do_not_establish_causality'),
    ('当前不能确定硬件需求；该方法保证提升精度。', 'experimental_rows_do_not_establish_causality'),
])
def test_table_only_overclaims_do_not_survive_supported_model_verdict(text, reason):
    audit, diagnostics = assessment(text)
    assert audit['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert audit['assessments'][0]['experimental_scope_reason'] == reason
    assert diagnostics['changed_fact_ids'] == ['f1']


@pytest.mark.parametrize('text', [
    'F1 在该表已测配置中最大，为 81.11。',
    '采用 token 交错配置时，该表 F1 为 81.11。',
    '当前结果不能保证提升精度。',
    'The reported value is 81.11.',
])
def test_valid_table_results_and_boundaries_are_preserved(text):
    audit, diagnostics = assessment(text)
    assert audit['assessments'][0]['verdict'] == 'SUPPORTED'
    assert diagnostics['changed_fact_ids'] == []


def test_method_passage_and_invalid_source_are_not_treated_as_table():
    assert assessment('Cross 将 token 交错排列。', {'t': {'kind': 'text'}})[1]['changed_fact_ids'] == []
    item = observation()
    assert source_roles([item])
    item['source_uri'] = 'paper://neighbor/layout/table'
    assert not source_roles([item])


def test_source_roles_follow_request_local_citation_aliases_in_both_stages():
    evidence = observation()
    eid = evidence['evidence_id']
    messages = []
    outputs = iter([
        {'facts': [{'id': 'f1', 'text': 'F1 为 81.11。', 'evidence_ids': ['ref1']}], 'direct_answer_id': 'f1'},
        {'assessments': [{'fact_id': 'f1', 'span_ids': ['s0'], 'reason': 'Value matches.',
                         'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}],
         'coverage': 'COMPLETE', 'missing_aspects': []}])
    class Model:
        def complete(self, sent, **kwargs):
            kwargs['before_attempt']()
            payload = json.loads(sent[-1]['content'])
            messages.append(payload)
            answer = next(outputs)
            if 'evidence' in payload:
                row = payload['table_rows'][0]
                answer['facts'] = [{'id': 'f1', 'kind': 'TABLE_FACT', 'operation': 'value',
                    'row_refs': [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'], 'metrics': ['F1']}]}]
            return LLMResponse(text=json.dumps(answer), finish_reason='stop')
    report = answer_then_audit('给出 F1', [evidence], Model())
    assert report['status'] == 'REVIEW_REQUIRED'
    assert eid in report['answer']
    assert all(m['source_roles'][0]['evidence_id'] == 'ref1' for m in messages)


@pytest.mark.parametrize('text,source,expected', [
    ('作者称该机制显著提升检测性能。', 'The authors report significantly improved performance.', 'SUPPORTED'),
    ('该机制带来统计显著的提升。', 'The authors report significantly improved performance.', 'UNCERTAIN'),
    ('该实验达到统计显著性。', 'The paired test reports p < 0.01.', 'SUPPORTED'),
    ('当前证据不能证明统计显著性。', 'Measured F1 increases.', 'SUPPORTED'),
])
def test_author_qualitative_description_is_distinct_from_statistical_assertion(text, source, expected):
    draft = {'facts': [{'id': 'f', 'text': text, 'evidence_ids': ['m']}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'span_ids': ['s0']}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, _ = apply_experimental_scope(draft, audit, {},
        spans=[{'span_id': 's0', 'evidence_id': 'm', 'text': source}])
    assert actual['assessments'][0]['verdict'] == expected


def test_statistical_phrase_in_an_unselected_source_span_cannot_support_claim():
    draft = {'facts': [{'id': 'f', 'text': '统计显著的提升。', 'evidence_ids': ['m']}]}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'span_ids': ['s0']}],
             'coverage': 'COMPLETE', 'missing_aspects': []}
    actual, _ = apply_experimental_scope(draft, audit, {}, spans=[
        {'span_id': 's0', 'evidence_id': 'm', 'text': 'Measured F1 increases.'},
        {'span_id': 's1', 'evidence_id': 'm', 'text': 'Another result has p < 0.01.'}])
    assert actual['assessments'][0]['verdict'] == 'UNCERTAIN'
