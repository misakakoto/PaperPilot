"""Offline boundary regressions; recorded responses are NOT a live answer PASS."""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path

import pytest

from paperpilot_agentic.answer_contracts import (
    conservative_audit_contract, validate_audit, evidence_spans,
)
from paperpilot_agentic.answer_coverage import apply_final_coverage, request_contract
from paperpilot_agentic.answer_gap_display import validate_request_quotes, request_quote_units
from paperpilot_agentic.answer_isolation import isolate_draft
from paperpilot_agentic.source_purpose import process_part_errors
from paperpilot_agentic.table_claims import apply_row_guard

ROOT = Path(__file__).resolve().parents[1]


def saved_cases():
    # Live responses are ignored private artifacts, never checked-in fixtures.
    path = Path(os.environ.get('PAPERPILOT_TEST_HANDOFF_JSON',
                               ROOT / 'outputs/handoff/三题与诊断.json'))
    if not path.is_file():
        pytest.skip('Set PAPERPILOT_TEST_HANDOFF_JSON to the saved redacted live diagnostics')
    return json.loads(path.read_text(encoding='utf-8'))['cases']


def test_saved_full_configuration_comparison_is_not_single_factor_attribution():
    d = saved_cases()[0]['diagnostics']
    result, trace = apply_row_guard(d['draft'], d['model_audit'], d['table_rows'])
    assert next(a for a in result['assessments'] if a['fact_id'] == 'f10')['verdict'] == 'SUPPORTED'
    assert 'f10' not in trace['changed_fact_ids']


def group_fixture(text, verdict='SUPPORTED'):
    rows = []
    for i, enabled in enumerate([[], ['Ax'], ['By'], ['Ax', 'By']]):
        rows.append({'row_id': f'r{i}', 'evidence_id': 'e', 'paper_sha256': 'paper',
            'artifact_sha256': 'artifact', 'table': 'Table IV',
            'component_columns': ['Ax', 'By'], 'checked_columns': enabled,
            'metric_headers': ['F1', 'IoU'],
            'cells': {'Dataset': 'Dataset-Z', 'offset': 'w/o', 'F1': str(50+i), 'IoU': str(30+i)}})
    facts = [{'id': f'p{i}', 'kind': 'SOURCE_FACT', 'text': f'F1={50+i}; IoU={30+i}',
        'evidence_ids': ['e'], 'row_refs': [{'evidence_id': 'e', 'row_id': f'r{i}', 'metrics': ['F1', 'IoU']}]}
        for i in range(4)]
    facts.append({'id': 'rank', 'kind': 'INFERENCE', 'text': text,
        'evidence_ids': ['e'], 'depends_on': [f'p{i}' for i in range(4)]})
    audit = {'assessments': [{'fact_id': f['id'], 'verdict': verdict if f['id']=='rank' else 'SUPPORTED'}
        for f in facts], 'coverage': 'COMPLETE', 'missing_aspects': []}
    return {'facts': facts}, audit, rows


@pytest.mark.parametrize('text', [
    '同时启用 Ax 与 By 的完整组合在 F1 和 IoU 上均优于基线及各单机制。',
    '共同启用 Ax、By 的配置在 F1 上高于各单机制。',
    'The configuration jointly enabling Ax and By is better than either single configuration on F1.',
    '完整组合（同时启用 Ax 和 By）在 F1 上优于基线。',
])
@pytest.mark.parametrize('verdict', ['SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN'])
def test_group_ranking_preserves_independent_verdict(text, verdict):
    draft, audit, rows = group_fixture(text, verdict)
    original = deepcopy((draft, audit, rows))
    actual, _ = apply_row_guard(draft, audit, rows)
    assert actual['assessments'][-1]['verdict'] == verdict
    assert (draft, audit, rows) == original


@pytest.mark.parametrize('field,value', [
    ('artifact_sha256', 'other-experiment'), ('paper_sha256', 'other-paper'),
    ('Dataset', 'other-data'), ('offset', '5px'), ('metrics', ['IoU']),
    ('unit', '%'),
])
def test_group_ranking_does_not_bypass_experimental_ownership(field, value):
    draft, audit, rows = group_fixture('同时启用 Ax 与 By 的完整组合在 F1 上优于基线及各单机制。')
    if field in {'Dataset', 'offset'}:
        rows[-1]['cells'][field] = value
    elif field == 'metrics':
        draft['facts'][0]['row_refs'][0]['metrics'] = value
    elif field == 'unit':
        rows[-1]['metric_headers'] = ['F1 (%)', 'IoU']
        rows[-1]['cells']['F1 (%)'] = rows[-1]['cells'].pop('F1')
        draft['facts'][-2]['row_refs'][0]['metrics'] = ['F1 (%)', 'IoU']
    else:
        rows[-1][field] = value
    actual, _ = apply_row_guard(draft, audit, rows)
    assert actual['assessments'][-1]['verdict'] != 'SUPPORTED'


@pytest.mark.parametrize('text', [
    '同时启用 Ax 和 By 使 F1 提升，全部改善由 Ax 导致。',
    '同时启用 Ax 时 F1 从 50 提升到 53。',
    'Simultaneously enabling Ax and By causes improvements due only to Ax.',
])
def test_group_words_do_not_legalize_causal_attribution(text):
    draft, audit, rows = group_fixture(text)
    actual, _ = apply_row_guard(draft, audit, rows)
    assert actual['assessments'][-1]['verdict'] != 'SUPPORTED'


def audit_fixture(n=18):
    content = '\n\n'.join(f'Original source sentence {i}.' for i in range(n))
    evidence = [{'evidence_id': 'e', 'content': content}]
    spans = evidence_spans(evidence)
    draft = {'facts': [{'id': 'm', 'text': 'Original source sentence 0.',
                       'kind': 'SOURCE_FACT', 'evidence_ids': ['e']}],
             'direct_answer_id': 'm', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'm', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                              'span_ids': ['s0']}], 'coverage': 'COMPLETE', 'missing_aspects': [],
             'omitted_answer_spans': [f's{i}' for i in range(1, n)],
             'requirement_assessments': [], 'limitation_assessments': [], 'answer_conflicts': []}
    return draft, audit, evidence, spans


def test_omission_overflow_fallback_preserves_all_diagnostics_not_full_pass():
    draft, audit, evidence, spans = audit_fixture()
    original = deepcopy(audit)
    with pytest.raises(ValueError, match='at most 12'):
        validate_audit(audit, draft, evidence, spans=spans, require_relevance=True)
    actual = conservative_audit_contract(audit, draft, spans, question='Explain the method.')
    validated = validate_audit(actual, draft, evidence, spans=spans, require_relevance=True)
    assert validated['coverage'] == 'PARTIAL'
    assert validated['assessments'][0]['verdict'] == 'SUPPORTED'
    assert len(validated['omitted_answer_spans']) == 12
    assert actual['unvalidated_omitted_answer_spans'] == original['omitted_answer_spans']
    assert audit == original


def test_no_surviving_valid_support_still_fails_closed():
    draft, audit, _, spans = audit_fixture()
    audit['assessments'][0]['span_ids'] = [f's{i}' for i in range(18)]
    with pytest.raises(ValueError, match='All repaired audit support'):
        conservative_audit_contract(audit, draft, spans, question='Explain the method.')


def test_audit_repair_reports_both_overflows_in_one_turn():
    from paperpilot_agentic.answer_contracts import audit_repair_diagnostics
    draft, audit, _, spans = audit_fixture(37)
    audit['assessments'][0]['span_ids'] = [f's{i}' for i in range(36)]
    audit['omitted_answer_spans'] = [f's{i}' for i in range(17)]
    diagnostic = audit_repair_diagnostics(audit, draft, spans,
        question='Explain the method.', primary=ValueError('first violation'))
    assert 'assessments[0].span_ids' in diagnostic and '36' in diagnostic
    assert 'omitted_answer_spans' in diagnostic and '17' in diagnostic


def test_saved_second_question_keeps_both_shape_errors_for_local_repair():
    d = saved_cases()[1]['diagnostics']
    raw = deepcopy(d['raw_draft'])
    # Use the saved locally projected table facts; no source text or live audit is fabricated.
    by_id = {f['id']: f for f in d['draft']['facts']}
    raw['facts'] = [deepcopy(by_id.get(f['id'], f)) if f['kind']=='TABLE_FACT' else f for f in raw['facts']]
    evidence = [{'evidence_id': eid, 'content': 'Structural fixture only.'}
                for eid in sorted({e for f in raw['facts'] for e in f['evidence_ids']})]
    isolated, trace = isolate_draft(raw, evidence, question=saved_cases()[1]['question'],
        table_rows=d['table_rows'], rejected_facts=process_part_errors(raw))
    assert {f['id'] for f in isolated['facts']} == {'f4', 'f5', 'f6'}
    for fid in ['f1', 'f2', 'f3']:
        error = next(e['error'] for e in trace['errors'] if e['fact_id']==fid)
        assert 'Mixed process and purpose' in error
        assert 'Unrequested structural formula' in error


def test_saved_post_audit_rejected_mechanism_is_not_a_read_only_premise():
    from paperpilot_agentic.answer_isolation import extend_isolation_after_audit
    d = saved_cases()[1]['diagnostics']
    original = deepcopy(d['raw_draft'])
    actual = extend_isolation_after_audit(original, d['draft_isolation'], d['audit'])
    assert set(actual['quarantined_fact_ids']) == {'f1', 'f2', 'f3', 'f4', 'f7', 'f8'}
    assert set(actual['eligible_fact_ids']) == {'f5', 'f6'}
    assert actual['audit_rejected_fact_ids'] == ['f4']
    assert d['draft_isolation']['eligible_fact_ids'] == ['f4', 'f5', 'f6']


@pytest.mark.parametrize('lost', [[], ['b']])
def test_coverage_requires_all_mapped_content_after_filtering(lost):
    draft = {'facts': [{'id': f, 'text': f, 'evidence_ids': ['e']} for f in ['a', 'b']], 'limitations': []}
    audit = {'assessments': [{'fact_id': f, 'verdict': 'UNSUPPORTED' if f in lost else 'SUPPORTED'}
            for f in ['a', 'b']], 'coverage': 'COMPLETE', 'missing_aspects': [],
        'requirement_assessments': [{'requirement': 'Compare both mechanisms', 'fact_ids': ['a','b'], 'gap': ''}]}
    effective, report = apply_final_coverage(draft, audit, [], [], request_contract('Compare both mechanisms', []))
    assert effective['coverage'] == ('PARTIAL' if lost else 'COMPLETE')
    assert report['requirements'][0]['status'] == ('MISSING' if lost else 'COVERED_MODEL_ONLY')
    assert report['requirements'][0]['withheld_fact_ids'] == lost


def test_conditional_policy_is_not_an_unanswered_scientific_question():
    question = '说明机制；证据不足时明确说明。'
    audit = {'requirement_assessments': [{'requirement':'证据不足时明确说明',
              'request_quote':'证据不足时明确说明。', 'fact_ids': [], 'gap':'没有足够证据'}]}
    with pytest.raises(ValueError, match='conditional evidence policy'):
        validate_request_quotes(audit, question)


def test_satisfied_conditional_policy_does_not_need_a_fake_fact():
    from paperpilot_agentic.answer_contracts import audited_presentation
    question = '说明机制；证据不足时明确说明。'
    draft = {'facts': [{'id':'m', 'text':'Alpha 处理输入。', 'evidence_ids':['e']}],
             'direct_answer_id': 'm', 'limitations': []}
    audit = {'assessments': [{'fact_id':'m', 'verdict':'SUPPORTED', 'relevance':'DIRECT'}],
        'coverage':'COMPLETE', 'missing_aspects':[], 'limitation_assessments':[],
        'requirement_assessments':[
            {'requirement':'说明机制','request_quote':'说明机制；','fact_ids':['m'],'gap':''},
            {'requirement':'证据不足时明确说明','request_quote':'证据不足时明确说明。','fact_ids':[],'gap':''}]}
    effective, coverage = apply_final_coverage(draft,audit,[],[],request_contract(question,[]))
    assert coverage['coverage']=='COMPLETE'
    assert coverage['requirements'][1]['status']=='CONDITIONAL_POLICY'
    displayed=audited_presentation(draft,effective,question=question)
    assert displayed['answer_scope']=='MODEL_REVIEWED'
    assert not any('本轮未完成' in x for x in displayed['limitations'])


def test_real_missing_evidence_is_not_removed_by_policy_handling():
    question='说明机制和训练数据；证据不足时明确说明。'
    draft={'facts':[{'id':'m','text':'Alpha 处理输入。','evidence_ids':['e']}],'limitations':[]}
    audit={'assessments':[{'fact_id':'m','verdict':'SUPPORTED'}],'coverage':'COMPLETE','missing_aspects':[],
        'requirement_assessments':[{'requirement':'训练数据','request_quote':'说明机制和训练数据；',
                                   'fact_ids':[],'gap':'当前证据未覆盖训练数据。'}]}
    result,trace=apply_final_coverage(draft,audit,[],[],request_contract(question,[]))
    assert result['coverage']=='PARTIAL' and trace['requirements'][0]['status']=='UNRESOLVED'


def test_mechanism_followup_cannot_reactivate_previous_table_task():
    old='说明两种方法；分别引用 Table IV 和 Table IX 中一个消融结果。'
    current='基于上一问，只作机制层面的比较，并说明结论边界。'
    history=[{'role':'user','user_request':old}]
    quote='分别引用 Table IV 和 Table IX 中一个消融结果。'
    audit={'requirement_assessments':[{'requirement':'比较消融', 'request_quote':quote,
                                     'fact_ids':[],'gap':'没有表格数据'}]}
    with pytest.raises(ValueError, match='superseded'):
        validate_request_quotes(audit,current,history)
    assert quote not in [u['text'] for u in request_quote_units(current,history)]
    # An ordinary continuation may still carry the earlier requirement.
    validate_request_quotes(audit,'继续上一问未完成的分析。',history)


def test_followup_preserves_applicable_historical_format_and_source_constraints():
    current='基于上一问，只作机制层面的比较。'
    history=[{'user_request':'不得混用两篇论文证据；请用中文。'}]
    for quote in ['不得混用两篇论文证据；','请用中文。']:
        validate_request_quotes({'requirement_assessments':[{'request_quote':quote}]},current,history)


@dataclass(frozen=True)
class OfflineResponse:
    """Only a fixture transport receipt; no SDK or remote model is involved."""
    text: str
    finish_reason: str = 'stop'
    prompt_tokens: int = 0
    completion_tokens: int = 0


def fixture_response(value):
    return OfflineResponse(json.dumps(value, ensure_ascii=False))


def test_real_pipeline_repairs_both_audit_errors_then_safely_publishes_partial():
    from paperpilot_agentic.answer_audit import answer_then_audit
    draft, invalid, evidence, _ = audit_fixture(37)
    invalid['assessments'][0]['span_ids'] = [f's{i}' for i in range(36)]
    invalid['omitted_answer_spans'] = [f's{i}' for i in range(17)]
    repaired = deepcopy(invalid)
    repaired['assessments'][0]['span_ids'] = ['s0']
    calls = []
    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(deepcopy(messages))
            if len(calls)==3:
                text = messages[-1]['content']
                assert 'assessments[0].span_ids' in text and 'omitted_answer_spans' in text
            return fixture_response([draft, invalid, repaired][len(calls)-1])
    report=answer_then_audit('Explain the method.', evidence, Model())
    assert len(calls)==3 and report['provider_attempts']==3
    assert report['status']=='REVIEW_REQUIRED' and report['answer_scope']=='PARTIAL'
    assert draft['facts'][0]['text'] in report['answer']
    assert len(report['audit']['unvalidated_omitted_answer_spans'])==17
    assert report['audit']['coverage']=='PARTIAL'


def test_actual_local_patch_targets_guard_rejection_and_keeps_only_audited_parts():
    from paperpilot_agentic.answer_audit import answer_then_audit
    calls=[]
    correct_b='Beta 对齐输入。'
    correct_c='Gamma 使用参考特征。'
    false_goal='Beta 可以消除所有误差。'
    source='Alpha 拼接特征。'+correct_b+correct_c
    class Model:
        def ensure_capacity(self, requests):
            assert len(requests)>=1
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(deepcopy(messages))
            payload=json.loads(messages[-1]['content'])
            if len(calls)==1:
                value={'facts':[
                    {'id':'a','text':'Alpha 拼接特征。','evidence_ids':['e']},
                    {'id':'b','text':'Beta 拼接特征 F = concat_c(T1,T2)，旨在消除空间偏移。','evidence_ids':['e']},
                    {'id':'c','text':correct_c+'并保证准确率提升。','evidence_ids':['e']}],
                    'direct_answer_id':'a','limitations':[]}
            elif len(calls)==3:
                assert set(payload['repair_fact_ids'])=={'b','c'}
                assert [f['id'] for f in payload['read_only_source_premises']]==['a']
                errors=' '.join(e['error'] for e in payload['contract_errors'])
                assert 'Mixed process and purpose' in errors and 'Unrequested structural formula' in errors
                value={'replacements':{
                    'b':{'kind':'SOURCE_FACT','text':correct_b,'evidence_ids':['e'],
                         'purpose':{'text':false_goal,'evidence_ids':['e']}},
                    'c':{'kind':'SOURCE_FACT','text':correct_c,'evidence_ids':['e']}},'limitations':[]}
            else:
                value={'assessments':[
                    {'fact_id':f['id'],'verdict':'UNSUPPORTED' if (
                        f['id']=='c' and len(calls)==2 or f['id'].endswith('__purpose')) else 'SUPPORTED',
                     'relevance':'DIRECT','span_ids':['s0'],'reason':'Fixture clause-level assessment.'}
                    for f in payload['facts']], 'coverage':'PARTIAL','missing_aspects':[]}
            return fixture_response(value)
    report=answer_then_audit('说明 Alpha、Beta、Gamma 的过程与目的。',
                             [{'evidence_id':'e','content':source}], Model())
    assert len(calls)==4 and report['draft_isolation']['repair_status']=='PATCH_AUDITED'
    assert report['draft_isolation']['audit_rejected_fact_ids']==['c']
    assert correct_b in report['answer'] and correct_c in report['answer']
    assert false_goal not in report['answer'] and '并保证准确率提升' not in report['answer']
    assert report['answer_scope']=='PARTIAL'
    assert {f['id'] for f in report['draft']['facts']}=={'a','b','c','b__purpose'}
    assert report['draft']['facts'][0]['text']=='Alpha 拼接特征。'


def test_actual_scope_contract_repairs_old_table_task_without_erasing_history():
    from paperpilot_agentic.answer_audit import answer_then_audit
    current='基于上一问，只作机制层面的比较。'
    old='说明两种方法；分别引用 Table IV 和 Table IX 中一个消融结果。'
    draft={'facts':[{'id':'a','text':'Beta 对齐双时相特征。','evidence_ids':['e']}],
           'direct_answer_id':'a','limitations':[]}
    audit={'assessments':[{'fact_id':'a','verdict':'SUPPORTED','relevance':'DIRECT', 'span_ids':['s0']}],
           'coverage':'PARTIAL','missing_aspects':[],
           'requirement_assessments':[{'requirement':'报告旧消融',
               'request_quote':'分别引用 Table IV 和 Table IX 中一个消融结果。','fact_ids':[],'gap':'没有表格结果'}]}
    repaired=deepcopy(audit)
    repaired.update(coverage='COMPLETE', requirement_assessments=[{
        'requirement':'机制比较','request_quote':'只作机制层面的比较。','fact_ids':['a'],'gap':''}])
    calls=[]
    class Model:
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(deepcopy(messages))
            if len(calls)==1:
                payload=json.loads(messages[-1]['content'])
                assert payload['request_history'][0]['user_request']==old
                assert payload['request_type_contract']['current_mechanism_only'] is True
            if len(calls)==3:
                assert 'superseded' in messages[-1]['content']
            return fixture_response([draft,audit,repaired][len(calls)-1])
    report=answer_then_audit(current,[{'evidence_id':'e','content':'Beta 对齐双时相特征。'}],Model(),
                             original_question=current,request_history=[{'user_request':old}])
    assert len(calls)==3 and report['answer_scope']=='MODEL_REVIEWED'
    assert report['final_request_coverage']['coverage']=='COMPLETE'
    assert report['request_contract']['table_requirements']==[]


def test_failed_content_patch_cannot_replace_saved_audited_answer():
    from paperpilot_agentic.answer_audit import answer_then_audit
    calls=[]
    class Model:
        def ensure_capacity(self,requests): pass
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            if len(calls)==1:
                value={'facts':[{'id':'a','text':'Alpha 拼接特征。','evidence_ids':['e']},
                    {'id':'b','text':'Beta 处理特征，旨在消除偏移。','evidence_ids':['e']}],
                    'direct_answer_id':'a','limitations':[]}
            elif len(calls)==2:
                value={'assessments':[{'fact_id':'a','verdict':'SUPPORTED','relevance':'DIRECT','span_ids':['s0']}],
                       'coverage':'PARTIAL','missing_aspects':[]}
            else:
                value={'replacements':{'b':{'text':'Beta 处理特征，旨在消除偏移。','evidence_ids':['e']}},
                       'limitations':[]}
            return fixture_response(value)
    report=answer_then_audit('说明过程与目的。',[{'evidence_id':'e','content':'Alpha 拼接特征。'}],Model())
    assert len(calls)==3 and report['draft_isolation']['repair_status']=='FAILED_VALIDATION'
    assert 'Alpha 拼接特征。' in report['answer'] and '消除偏移' not in report['answer']
    assert report['answer_scope']=='PARTIAL'
