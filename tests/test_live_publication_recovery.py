"""Real failure classes, with unrelated source names and no answer goldens."""
from copy import deepcopy
import json
import pytest
from paperpilot_agentic.answer_integrity import quarantine_answer_conflicts
from paperpilot_agentic.answer_coverage import apply_final_coverage, request_contract
from paperpilot_agentic.answer_contracts import audited_presentation
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


def test_rejected_claim_does_not_conflict_with_independently_supported_value():
    draft={'facts':[{'id':'value','text':'Beta has F1 70.','evidence_ids':['e']},
                    {'id':'bad','text':'Beta is best everywhere.','evidence_ids':['e']} ]}
    audit={'assessments':[{'fact_id':'value','verdict':'SUPPORTED','span_ids':['s0']},
                          {'fact_id':'bad','verdict':'UNSUPPORTED'}],
           'coverage':'PARTIAL','missing_aspects':[],
           'answer_conflicts':[{'fact_ids':['value','bad'],'reason':'An inconsistent best claim.'}]}
    actual,trace=quarantine_answer_conflicts(audit,draft,[{'span_id':'s0','evidence_id':'e','text':'F1 70'}],question='Compare')
    assert actual['assessments'][0]['verdict']=='SUPPORTED'
    assert trace['changed_fact_ids']==[]


def test_whole_reviewed_evidence_relative_negative_clauses_display_and_cover_same_boundary():
    limit='当前证据无法确定两个编码器的误差差异；未提供同条件下的对比实验。'
    draft={'facts':[{'id':'m','text':'Beta 估计位移。','evidence_ids':['e']}],
           'direct_answer_id':'m','limitations':[limit]}
    audit={'assessments':[{'fact_id':'m','verdict':'SUPPORTED','relevance':'DIRECT'}],
        'coverage':'COMPLETE','missing_aspects':[],
        'limitation_assessments':[{'limitation_index':0,'role':'SCOPE_BOUNDARY','reason':'Bounds extrapolation.'}],
        'requirement_assessments':[{'requirement':'说明过程','fact_ids':['m'],'gap':''},
            {'requirement':'说明边界','fact_ids':[],'limitation_indices':[0],'gap':''}]}
    original=deepcopy(draft)
    actual,coverage=apply_final_coverage(draft,audit,[],[],request_contract('说明过程和边界。',[]))
    shown=audited_presentation(draft,actual,question='说明过程和边界。')
    assert shown['limitations']==[limit]
    assert coverage['coverage']=='COMPLETE'
    assert shown['answer_scope']=='MODEL_REVIEWED'
    assert draft==original


def test_actual_local_patch_contract_requires_explicit_parts_when_process_contains_purpose():
    calls=[]
    class Model:
        def ensure_capacity(self,requests):pass
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            payload=json.loads(messages[-1]['content'])
            if len(calls)==1:
                value={'facts':[{'id':'a','text':'Alpha 拼接特征。','evidence_ids':['e']},
                    {'id':'b','text':'Beta 处理特征，旨在对齐空间位置。','evidence_ids':['e']}],
                    'direct_answer_id':'a','limitations':[]}
            elif len(calls)==3:
                assert 'LOCAL PATCH CONTRACT' in messages[0]['content']
                value={'replacements':{'b':{'kind':'SOURCE_FACT','text':'Beta 处理特征。','evidence_ids':['e'],
                        'purpose':{'text':'作者旨在对齐空间位置。','evidence_ids':['e']}}},'limitations':[]}
            else:
                value={'assessments':[{'fact_id':f['id'],'verdict':'SUPPORTED','relevance':'DIRECT',
                                      'span_ids':['s0']} for f in payload['facts']],
                       'coverage':'COMPLETE','missing_aspects':[]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    result=answer_then_audit('说明Alpha与Beta的操作和目的。',
        [{'evidence_id':'e','content':'Alpha 拼接特征。Beta 处理特征。作者旨在对齐空间位置。'}],Model())
    assert len(calls)==4
    assert result['draft_isolation']['repair_status']=='PATCH_AUDITED'
    assert {f['id'] for f in result['draft']['facts']}=={'a','b','b__purpose'}


@pytest.mark.parametrize('tail',[
    '模型优于所有方法。','论文未报告实验。','未提供数据，但是模型可以提升精度。',
    '未提供数据，因此Beta更准确。','未提供数据，并证明方法有效。'])
def test_unsafe_negative_sequence_cannot_cover_an_approved_boundary(tail):
    from paperpilot_agentic.answer_completion import retained_boundary_indices
    value='当前证据无法确定误差；'+tail
    assert retained_boundary_indices({'limitations':[value]},
        {'limitation_assessments':[{'limitation_index':0,'role':'SCOPE_BOUNDARY'}]})==set()


def test_unreviewed_or_withheld_negative_sequence_remains_uncountable():
    from paperpilot_agentic.answer_completion import retained_boundary_indices
    value='当前证据无法确定误差；未提供测试数据。'
    assert retained_boundary_indices({'limitations':[value]}, {})==set()
    assert retained_boundary_indices({'limitations':[value]},
        {'limitation_assessments':[{'limitation_index':0,'role':'SCOPE_BOUNDARY'}],
         'withheld_limitation_indices':[0]})==set()


def test_two_simultaneously_supported_conflicting_claims_still_withheld():
    draft={'facts':[{'id':fid,'text':fid,'evidence_ids':['e']} for fid in ('a','b')]}
    audit={'assessments':[{'fact_id':fid,'verdict':'SUPPORTED'} for fid in ('a','b')],
           'coverage':'COMPLETE','missing_aspects':[],
           'answer_conflicts':[{'fact_ids':['a','b'],'reason':'Contradiction'}]}
    actual,trace=quarantine_answer_conflicts(audit,draft,[],question='Compare')
    assert set(trace['changed_fact_ids'])=={'a','b'}
    assert actual['coverage']=='PARTIAL'


@pytest.mark.parametrize('text', ['作者旨在降低估计误差。','图像处理器旨在对齐输入。',
                                'The processing module aims to align features.'])
def test_atomic_goal_or_method_name_is_not_mixed_processing(text):
    from paperpilot_agentic.source_purpose import process_part_errors
    assert not process_part_errors({'facts':[{'id':'a','text':text,'evidence_ids':['e']}]})


def test_host_comparison_context_is_bound_to_current_projection_and_selected_rows():
    from paperpilot_agentic.table_projection import project_table_facts,projection_audit_context
    from test_table_fact_projection import rows
    catalog=rows()
    choice={'facts':[{'id':'best','kind':'TABLE_FACT','operation':'max',
                     'row_refs':[{'evidence_id':'t','row_id':r['row_id'],'metrics':['F1 (%)']} for r in catalog[1:]]}]}
    projected,trace=project_table_facts(choice,catalog,chinese=True)
    context=projection_audit_context(projected,trace)
    assert context[0]['domain']=='explicit_selected_rows_only'
    assert context[0]['rankings'][0]['selected_row_ids']==['r1','r2']
    altered=deepcopy(projected);altered['facts'][0]['text']+=' everywhere'
    assert projection_audit_context(altered,trace)==[]


def test_failed_format_repair_cannot_silently_accept_mixed_process():
    calls=[]
    mixed={'facts':[{'id':'a','text':'Beta 处理特征，旨在对齐空间位置。','evidence_ids':['e']}],
           'direct_answer_id':'a','limitations':[]}
    class Model:
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            return LLMResponse(json.dumps(mixed,ensure_ascii=False),finish_reason='stop')
    result=answer_then_audit('说明操作与目的。',[{'evidence_id':'e','content':'Beta 处理特征。'}],Model())
    assert len(calls)==2
    assert not result.get('answer') and result['status']=='FAILED'
