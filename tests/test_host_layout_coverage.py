"""Final formatting is a host obligation, not a new scientific fact."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from paperpilot_agentic.answer_coverage import apply_final_coverage, request_contract
from paperpilot_agentic.answer_pipeline import AnswerOptions, render_answer


QUESTION = '说明方法和结果。请将方法机制和实验结论分开回答。'


def fixture():
    draft = {'facts': [
        {'id': 'm', 'text': '模块融合两路特征。', 'evidence_ids': ['e']},
        {'id': 'n', 'text': '实验测得 F1=62.3%。', 'evidence_ids': ['e'],
         'row_refs': [{'row_id': 'r', 'evidence_id': 'e', 'metric': 'F1'}]}],
        'direct_answer_id': 'm', 'limitations': []}
    audit = {'assessments': [{'fact_id': fid, 'verdict': 'SUPPORTED', 'relevance': 'DIRECT'}
                            for fid in ('m', 'n')],
             'coverage': 'COMPLETE', 'missing_aspects': [],
             'requirement_assessments': [
                 {'requirement': '说明方法和结果', 'request_quote': '说明方法和结果。',
                  'fact_ids': ['m', 'n'], 'gap': ''},
                 {'requirement': '分开回答', 'request_quote': '请将方法机制和实验结论分开回答。',
                  'fact_ids': [], 'gap': ''}]}
    return draft, audit


def render(draft, audit):
    contract = request_contract(QUESTION, [])
    initial, coverage = apply_final_coverage(draft, audit, [], [], contract)
    report = {'read_packet': {'identity': 'test'}, 'request_contract': contract,
              'pre_coverage_audit': deepcopy(audit), 'audit': initial,
              'final_request_coverage': coverage}
    context = SimpleNamespace(question=QUESTION, options=AnswerOptions(), rows=[], report=report)
    render_answer(context, draft, initial,
                  lambda *args: pytest.fail('Host layout must not call a provider'))
    return report


def test_complete_host_sections_satisfy_pure_format_without_fact_ids():
    draft, audit = fixture()
    report = render(draft, audit)
    assert report['presentation_contract']['status'] == 'SATISFIED'
    assert '### 方法机制' in report['answer'] and '### 实验结论' in report['answer']
    assert report['audit']['coverage'] == 'COMPLETE'
    assert report['answer_scope'] != 'PARTIAL'
    assert report['final_request_coverage']['requirements'][-1]['status'] == 'COVERED_HOST_PRESENTATION'


@pytest.mark.parametrize('failure', ['rejected', 'model_partial', 'semantic_gap', 'foreign', 'mixed'])
def test_layout_cannot_waive_content_or_model_uncertainty(failure):
    draft, audit = fixture()
    if failure == 'rejected':
        audit['assessments'][1]['verdict'] = 'UNSUPPORTED'
    elif failure == 'model_partial':
        audit.update(coverage='PARTIAL', missing_aspects=['需要更多机制依据'])
    elif failure == 'semantic_gap':
        audit['requirement_assessments'][0]['gap'] = '尚未解释工作条件'
    elif failure == 'foreign':
        audit['requirement_assessments'][-1]['request_quote'] = '请将实验结论和机制方法分开回答。'
    else:
        audit['requirement_assessments'][-1]['request_quote'] = '请将方法机制和实验结论分开回答并说明参数。'
    report = render(draft, audit)
    assert report['final_request_coverage']['coverage'] == 'PARTIAL'
    assert report['answer_scope'] == 'PARTIAL'


def test_layout_does_not_mutate_original_audit():
    draft, audit = fixture()
    before = deepcopy(audit)
    render(draft, audit)
    assert audit == before
