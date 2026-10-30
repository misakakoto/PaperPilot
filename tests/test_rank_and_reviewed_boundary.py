"""Lock down two real publication failures with unrelated source examples."""
from copy import deepcopy
import json
import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.table_projection import project_table_facts
from test_table_fact_projection import rows


def ranked_selection(operation='max'):
    catalog=rows()
    for index,row in enumerate(catalog):
        row['metric_headers']=['F1 (%)','Precision (%)']
        row['cells']['Precision (%)']=str(71+index)
    value={'facts':[{'id':'winner','kind':'TABLE_FACT','operation':operation,
        'row_refs':[{'evidence_id':'t','row_id':r['row_id'],
                     'metrics':['F1 (%)','Precision (%)']} for r in catalog]}]}
    return value,catalog


@pytest.mark.parametrize('operation,winner',[('max','r2'),('min','r0')])
def test_common_winner_is_computed_per_metric_without_cross_column_arithmetic(operation,winner):
    value,catalog=ranked_selection(operation)
    originals=deepcopy((value,catalog))
    result,trace=project_table_facts(value,catalog,chinese=True)
    fact=result['facts'][0]
    assert fact['id']=='winner' and fact['kind']=='SOURCE_FACT'
    assert fact['row_refs'][0]['row_id']==winner
    assert set(trace[0]['rankings'][i]['metric'] for i in range(2))=={'F1 (%)','Precision (%)'}
    assert all(item['winner_row_ids']==[winner] for item in trace[0]['rankings'])
    assert all(item['selected_row_ids']==['r0','r1','r2'] for item in trace[0]['rankings'])
    assert '所选表内配置' in fact['text'] and '分别' in fact['text']
    assert (value,catalog)==originals


@pytest.mark.parametrize('defect,match',[
    ('different_winners','Different column winners'),('tie','Tied extrema'),
    ('dataset','Other source conditions'),('numeric_condition','Other source conditions'),
    ('artifact','one source table'),('metric_order','shared columns'),
    ('unknown_row','Unowned source'),('repeated_row','Repeated table row'),
    ('multi_difference','differences require one metric')])
def test_multiple_columns_do_not_relax_source_conditions_or_invent_overall_winner(defect,match):
    value,catalog=ranked_selection()
    if defect=='different_winners':catalog[0]['cells']['Precision (%)']='99'
    elif defect=='tie':catalog[1]['cells']['Precision (%)']=catalog[2]['cells']['Precision (%)']
    elif defect=='dataset':catalog[1]['cells']['dataset']='Other'
    elif defect=='numeric_condition':
        for row in catalog:row['cells']['batch_size']='4'
        catalog[1]['cells']['batch_size']='8'
    elif defect=='artifact':catalog[1]['artifact_sha256']='other'
    elif defect=='metric_order':value['facts'][0]['row_refs'][1]['metrics'].reverse()
    elif defect=='unknown_row':value['facts'][0]['row_refs'][1]['row_id']='forged'
    elif defect=='repeated_row':value['facts'][0]['row_refs'][1]=deepcopy(value['facts'][0]['row_refs'][0])
    else:value['facts'][0]['operation']='difference'
    original=deepcopy((value,catalog))
    with pytest.raises(ValueError,match=match):project_table_facts(value,catalog,chinese=True)
    assert (value,catalog)==original


def test_mixed_column_units_are_never_averaged_or_rescaled_to_find_a_winner():
    value,catalog=ranked_selection()
    for row in catalog:
        row['cells']['Score']=row['cells'].pop('Precision (%)')
        row['metric_headers']=['F1 (%)','Score']
    for ref in value['facts'][0]['row_refs']:ref['metrics']=['F1 (%)','Score']
    result,trace=project_table_facts(value,catalog,chinese=True)
    assert 'F1 (%)=63.40' in result['facts'][0]['text'] and 'Score=73' in result['facts'][0]['text']
    assert [entry['extreme'] for entry in trace[0]['rankings']]==['63.40','73']
    assert 'overall' not in result['facts'][0]['text']


def test_common_winner_local_replacement_preserves_identity_and_reaches_final_audit():
    from test_shared_packet_cost import fixture
    source=fixture()
    item=source['provenance']['structured_extraction']
    item['column_headers'].append('Precision (%)')
    for index,row in enumerate(item['table_rows']):row['cells']['Precision (%)']=str(71+index)
    source['content']='Table IV\n'+json.dumps({'column_headers':item['column_headers'],
                                               'rows':item['table_rows']},ensure_ascii=False)
    calls=[]
    class Model:
        def ensure_capacity(self,requests):pass
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            payload=json.loads(messages[-1]['content'])
            if len(calls)==1:
                value={'facts':[
                    {'id':'good','kind':'SOURCE_FACT','text':'实验使用Delta数据集。','evidence_ids':['t']},
                    {'id':'winner','kind':'INFERENCE','text':'表内最佳配置的F1=4.50。',
                     'evidence_ids':['t'],'depends_on':['good']}],
                    'direct_answer_id':'good','limitations':[]}
            elif len(calls)==3:
                value={'replacements':{'winner':{'kind':'TABLE_FACT','operation':'max',
                    'row_refs':[{'evidence_id':r['evidence_id'],'row_id':r['row_id'],
                                 'metrics':['F1','Precision (%)']} for r in payload['table_rows']]}},
                    'limitations':[]}
            else:
                if len(calls)==4:
                    contexts=payload['host_calculations']
                    assert [entry['fact_id'] for entry in contexts]==['winner']
                    assert contexts[0]['domain']=='explicit_selected_rows_only'
                    assert len(contexts[0]['rankings'])==2
                value={'assessments':[{'fact_id':f['id'],'span_ids':['s0'],
                    'verdict':'SUPPORTED','relevance':'DIRECT'} for f in payload['facts']],
                    'coverage':'COMPLETE','missing_aspects':[]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    source['evidence_id']='t'
    report=answer_then_audit('说明数据集并比较Table IV的F1和Precision最高配置。',[source],Model())
    assert report['draft_isolation']['repair_status']=='PATCH_AUDITED',report.get('failure_detail')
    assert len(calls)==4 and report['answer_scope']=='MODEL_REVIEWED'
    assert '分别' in report['answer'] and '4.50' in report['answer']
    assert report['local_table_projection'][0]['rankings'][0]['winner_row_ids']
    assert next(f for f in report['draft']['facts'] if f['id']=='winner')['kind']=='SOURCE_FACT'


def test_reviewed_evidence_relative_boundary_survives_default_coverage_and_display():
    boundary='当前证据未提供两个方法在同一数据集上的直接对照，因此不能证明 Beta 在性能上优于 Alpha。'
    calls=[]
    class Model:
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            if len(calls)==1:
                value={'facts':[{'id':'f','text':'Beta 使用特征变形。','evidence_ids':['e']}],
                       'direct_answer_id':'f','limitations':[boundary]}
            else:
                value={'assessments':[{'fact_id':'f','span_ids':['s0'],'verdict':'SUPPORTED',
                                      'relevance':'DIRECT'}], 'coverage':'COMPLETE','missing_aspects':[],
                    'requirement_assessments':[
                        {'requirement':'说明机制','request_quote':'说明机制，','fact_ids':['f'],'gap':''},
                        {'requirement':'说明比较边界','request_quote':'并说明比较不能证明什么。',
                         'fact_ids':[],'limitation_indices':[0],'gap':''}],
                    'limitation_assessments':[{'limitation_index':0,'role':'SCOPE_BOUNDARY',
                        'reason':'This evidence-relative caveat limits performance extrapolation, not the requested mechanism.'}]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    report=answer_then_audit('说明机制，并说明比较不能证明什么。',
                            [{'evidence_id':'e','content':'Beta 使用特征变形。'}],Model())
    assert boundary in report['limitations']
    assert report['answer_scope']=='MODEL_REVIEWED'
    assert len(calls)==2
    assert report['final_request_coverage']['requirements'][1]['retained_limitation_indices']==[0]
    assert not report['internal_gap_diagnostics']['request_gaps']


@pytest.mark.parametrize('defect',[
    'unreviewed','request_gap','paper_absence','positive_tail','causal_tail','withheld'])
def test_positive_claims_or_unreviewed_boundaries_do_not_complete_requested_work(defect):
    from paperpilot_agentic.answer_coverage import apply_final_coverage, request_contract
    from paperpilot_agentic.answer_contracts import audited_presentation
    from test_reviewed_boundary_recovery import boundary_sample
    draft,audit=boundary_sample()
    draft['facts'][0]['text']='Beta 使用特征变形。'
    boundary='当前证据未提供两个方法在同一数据集上的直接对照，因此不能证明 Beta 在性能上优于 Alpha。'
    draft['limitations']=[boundary]
    if defect=='unreviewed':audit.pop('limitation_assessments')
    elif defect=='request_gap':
        audit['limitation_assessments'][0].update(role='REQUEST_GAP',requirement='State the comparison boundary')
        audit['requirement_assessments'][1]['gap']='Unresolved boundary.'
        audit['coverage']='PARTIAL'
    elif defect=='paper_absence':draft['limitations']=[boundary.replace('当前证据','论文')]
    elif defect=='positive_tail':draft['limitations']=[boundary[:-1]+'，但 Beta 肯定适用于所有任务。']
    elif defect=='causal_tail':draft['limitations']=[boundary[:-1]+'因为它保证绝对安全。']
    else:audit['withheld_limitation_indices']=[0]
    actual,coverage=apply_final_coverage(draft,audit,[],[],request_contract('请说明机制和结论边界。',[]))
    assert actual['coverage']=='PARTIAL'
    assert coverage['requirements'][1]['retained_limitation_indices']==[]
    shown=audited_presentation(draft,actual,question='请说明机制和结论边界。')
    assert shown['answer_scope']=='PARTIAL'
    assert not any(value==draft['limitations'][0] for value in shown['limitations'])


def test_later_withheld_boundary_stays_withheld_after_an_earlier_visible_boundary():
    from paperpilot_agentic.answer_coverage import apply_final_coverage,request_contract
    from test_reviewed_boundary_recovery import boundary_sample
    draft,audit=boundary_sample()
    draft['facts'][0]['text']='Beta 使用特征变形。'
    draft['limitations']=['当前证据无法确定其他数据集上的结果。',
        '当前证据未提供两个方法的直接对照，因此不能证明 Beta 性能优于 Alpha。']
    audit['limitation_assessments'].append({'limitation_index':1,'role':'SCOPE_BOUNDARY',
                                          'reason':'This boundary has been explicitly withheld.'})
    audit['requirement_assessments'][1]['limitation_indices']=[1]
    audit['withheld_limitation_indices']=[1]
    result,coverage=apply_final_coverage(draft,audit,[],[],request_contract('请说明机制和结论边界。',[]))
    assert result['coverage']=='PARTIAL'
    assert coverage['requirements'][1]['retained_limitation_indices']==[]
