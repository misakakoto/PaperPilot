"""Explicit percent spelling must retain the same dataset/value safeguards."""
import pytest
from paperpilot_agentic.numeric_scope import apply_scope_guard


@pytest.mark.parametrize('spelling', ['%', ' percent', ' per cent'])
def test_explicit_percent_spelling_preserves_supported_score(spelling):
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT',
        'text': '在 CitySplit-v1 测试集上，RadarNet 的 F1 为 87.2%。', 'evidence_ids': ['e']}],
        'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                             'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': []}
    evidence = [{'evidence_id': 'e', 'content':
        f'RadarNet uses SAR image pairs. On CitySplit-v1 test, F1 is 87.2{spelling}.'}]
    effective, report = apply_scope_guard(draft, audit, evidence)
    assert effective['assessments'][0]['verdict'] == 'SUPPORTED', report


@pytest.mark.parametrize('source,expected', [
    ('On OtherSet test, RadarNet F1 is 87.2 percent.', 'UNSUPPORTED'),
    ('On CitySplit-v1 test, RadarNet F1 is 88.2 percent.', 'UNCERTAIN'),
    ('On CitySplit-v1 test, RadarNet mIoU is 87.2 percent.', 'UNSUPPORTED'),
    ('On CitySplit-v1 test, RadarNet F1 is 0.872.', 'UNCERTAIN'),
    ('On CitySplit-v1 test hardware, RadarNet F1 is 87.2 percent.', 'UNCERTAIN'),
])
def test_unit_recognition_does_not_relax_scope_value_or_metric(source, expected):
    draft = {'facts': [{'id': 'f', 'kind': 'SOURCE_FACT',
        'text': '在 CitySplit-v1 测试集上，RadarNet 的 F1 为 87.2%。', 'evidence_ids': ['e']}],
        'direct_answer_id': 'f', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                             'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': []}
    effective, _ = apply_scope_guard(draft, audit, [{'evidence_id': 'e', 'content': source}])
    assert effective['assessments'][0]['verdict'] == expected
