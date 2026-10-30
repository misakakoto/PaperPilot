"""A reading role cannot derive metric computability from a missing marker."""
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_pipeline import AnswerOptions, generate_draft, initialize
from paperpilot_agentic.literal_table_writer import inference_scope_errors
from test_table_claim_rows import table


@pytest.mark.parametrize('text', [
    '该标记不是实测零分，因此无法计算或报告 F1。',
    'The class has no observations, so F1 cannot be computed.',
    'The metric is undefined because the class is absent.',
])
def test_unsupported_computability_is_a_role_violation(text):
    raw = {'facts': [{'id': 'claim', 'kind': 'INFERENCE', 'text': text}]}
    assert set(inference_scope_errors(raw)) == {'claim'}
    raw['facts'][0]['kind'] = 'SOURCE_FACT'
    assert not inference_scope_errors(raw)  # author statement still needs ordinary audit


def test_zero_distinction_and_unknown_score_are_not_computability_claims():
    assert not inference_scope_errors({'facts': [{'id': 'f', 'kind': 'INFERENCE',
        'text': 'Missing observations are not a measured zero; the actual score is unknown.'}]})


def test_actual_writer_boundary_quarantines_only_extra_inference():
    source = table()
    raw = {'facts': [
        {'id': 'source', 'kind': 'SOURCE_FACT', 'text': 'The legend defines a missing observation.',
         'evidence_ids': ['t']},
        {'id': 'extra', 'kind': 'INFERENCE', 'text': 'F1 cannot be calculated.',
         'evidence_ids': ['t'], 'depends_on': ['source']}],
        'direct_answer_id': 'source', 'limitations': []}
    before = deepcopy(raw)
    context = initialize('Explain Table IV marker.', [source], object(), AnswerOptions(), None)
    result = generate_draft(context, lambda *args, **kwargs: raw)
    assert [f['id'] for f in result['facts']] == ['source']
    assert context.report['draft_isolation']['quarantined_fact_ids'] == ['extra']
    assert context.report['raw_draft'] == before
    assert not context.report.get('all_facts_supported')
