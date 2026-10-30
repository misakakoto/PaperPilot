"""A rejected purpose must not swallow a separately audited operation."""
import json
from copy import deepcopy
import pytest
from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.source_purpose import project_source_purposes
from paperpilot_agentic.answer_contracts import validate_fact_shape
from paperpilot_agentic.answer_isolation import merge_repair, isolate_repair_targets


def test_independently_audited_operation_survives_unsupported_nested_purpose():
    calls=[]
    operation='Beta 将特征变形到参考特征的坐标系。'
    purpose='Beta 的目的在于提升分类准确率。'
    class Model:
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            if len(calls)==1:
                value={'facts':[{'id':'method','kind':'SOURCE_FACT','text':operation,
                    'evidence_ids':['e'],'purpose':{'text':purpose,'evidence_ids':['e']}}],
                    'direct_answer_id':'method','limitations':[]}
            else:
                payload=json.loads(messages[-1]['content'])
                value={'assessments':[{'fact_id':f['id'],'verdict':'SUPPORTED',
                    'relevance':'DIRECT','span_ids':['s0']} for f in payload['facts']],
                    'coverage':'COMPLETE','missing_aspects':[],
                    'requirement_assessments':[
                        {'requirement':'说明过程','fact_ids':['method'],'gap':''},
                        {'requirement':'说明目的','fact_ids':['method__purpose'],'gap':''}]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    report=answer_then_audit('说明 Beta 的处理过程和目的。',
                            [{'evidence_id':'e','content':operation}],Model())
    assert operation in report.get('answer','')
    assert purpose not in report['answer']
    assert report['answer_scope']=='PARTIAL' and len(calls)==2
    assert [f['id'] for f in json.loads(calls[1][-1]['content'])['facts']]==['method','method__purpose']
    assert report['source_purpose_projection'][0]['purpose_fact_id']=='method__purpose'
    assert next(a for a in report['audit']['assessments'] if a['fact_id']=='method')['verdict']=='SUPPORTED'
    assert next(a for a in report['audit']['assessments'] if a['fact_id']=='method__purpose')['verdict']=='UNCERTAIN'


def test_host_keyed_local_repair_keeps_operation_and_audits_purpose_separately():
    calls=[]
    operation='Beta 将特征变形到参考特征的坐标系。'
    class Model:
        def ensure_capacity(self,requests):pass
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            payload=json.loads(messages[-1]['content'])
            if len(calls)==1:
                value={'facts':[
                    {'id':'a','text':'Alpha 拼接双时相特征。','evidence_ids':['e']},
                    {'id':'b','text':'Beta 使用 F = concat_c(T1,T2)。','evidence_ids':['e']}],
                    'direct_answer_id':'a','limitations':[]}
            elif len(calls)==3:
                value={'replacements':{'b':{'kind':'SOURCE_FACT','text':operation,
                    'evidence_ids':['e'],'purpose':{'text':'Beta 能提升分类精度。','evidence_ids':['e']}}},
                    'limitations':[]}
            else:
                value={'assessments':[{'fact_id':f['id'],'verdict':'SUPPORTED',
                    'relevance':'DIRECT','span_ids':['s0']} for f in payload['facts']],
                    'coverage':'PARTIAL' if len(calls)==2 else 'COMPLETE','missing_aspects':[]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    report=answer_then_audit('用文字说明 Alpha 和 Beta 的处理过程与目的。',
        [{'evidence_id':'e','content':'Alpha 拼接双时相特征。'+operation}],Model())
    assert report['draft_isolation']['repair_status']=='PATCH_AUDITED'
    assert operation in report['answer'] and '能提升分类精度' not in report['answer']
    assert report['answer_scope']=='PARTIAL' and len(calls)==4
    assert {f['id'] for f in report['draft']['facts']}=={'a','b','b__purpose'}
    assert report['local_source_purpose_projection'][0]['purpose_fact_id']=='b__purpose'


def parts():
    return {'facts':[{'id':'m','kind':'SOURCE_FACT','text':'Method warps the features.',
        'evidence_ids':['process'],'purpose':{'text':'Authors aim to improve alignment.',
        'evidence_ids':['goal']}}], 'direct_answer_id':'m','limitations':[]}


def test_projection_is_lossless_and_requires_separate_source_audit():
    raw=parts(); original=deepcopy(raw)
    value,trace=project_source_purposes(raw)
    assert raw==original
    assert value['direct_answer_id']=='m'
    assert value['facts'][0]['evidence_ids']==['process']
    assert value['facts'][1]['evidence_ids']==['goal']
    assert trace[0]['original_parts']==original['facts'][0]
    assert trace[0]['support_status']=='REQUIRES_INDEPENDENT_AUDIT'


@pytest.mark.parametrize('purpose', [None, {}, {'text':'x','evidence_ids':[]},
    {'text':'','evidence_ids':['e']}, {'text':'x','evidence_ids':['e'],'id':'forged'},
    {'text':'x','evidence_ids':'e'}])
def test_malformed_parts_cannot_be_ignored(purpose):
    value=parts();value['facts'][0]['purpose']=purpose
    with pytest.raises(ValueError):project_source_purposes(value)
    with pytest.raises(ValueError):validate_fact_shape(value['facts'][0],{'process'})


@pytest.mark.parametrize('kind',['TABLE_FACT','INFERENCE'])
def test_purpose_not_allowed_on_other_fact_kinds(kind):
    value=parts();value['facts'][0]['kind']=kind
    with pytest.raises(ValueError):project_source_purposes(value)


def test_reserved_identity_and_expanded_cap_cannot_be_bypassed():
    with pytest.raises(ValueError):project_source_purposes(parts(),reserved_ids=['m__purpose'])
    value=parts()
    value['facts'] += [{'id':str(i),'text':'x','evidence_ids':['e']} for i in range(11)]
    with pytest.raises(ValueError,match='12-fact'):project_source_purposes(value)


def test_legacy_compound_prose_is_not_split_or_approved():
    value={'facts':[{'id':'m','text':'Method warps features and improves accuracy.',
                    'evidence_ids':['e']}],'direct_answer_id':'m','limitations':[]}
    projected,trace=project_source_purposes(value)
    assert projected==value and trace==[]


def test_raw_model_child_addition_never_becomes_host_authorized():
    original=parts();original['facts'][0].pop('purpose')
    patch={'facts':deepcopy(original['facts'])+[
        {'id':'m__purpose','text':'Invented benefit','evidence_ids':['process']}],'limitations':[]}
    with pytest.raises(ValueError,match='exactly'):
        merge_repair(original,patch,{'quarantined_fact_ids':['m']},
            [{'evidence_id':'process','content':'Method warps the features.'}],
            question='Explain method',request_history=None,table_rows=[])
    retained,additions=isolate_repair_targets(patch,['m'])
    assert len(retained['facts'])==1 and additions[0]['id']=='m__purpose'


def test_valid_author_purpose_with_nested_alias_reaches_auditor_and_final_answer():
    calls=[]; source_id='evidence_'+('a'*40)
    operation='Beta 将特征变形到参考坐标系。'
    purpose='作者以此改善空间对齐。'
    class Model:
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']();calls.append(messages)
            if len(calls)==1:
                value={'facts':[{'id':'m','text':operation,'evidence_ids':['ref1'],
                    'purpose':{'text':purpose,'evidence_ids':['ref1']}}],
                    'direct_answer_id':'m','limitations':[]}
            else:
                payload=json.loads(messages[-1]['content'])
                assert all(f['evidence_ids']==['ref1'] for f in payload['facts'])
                value={'assessments':[{'fact_id':f['id'],'verdict':'SUPPORTED',
                    'relevance':'DIRECT','span_ids':['s0']} for f in payload['facts']],
                    'coverage':'COMPLETE','missing_aspects':[]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    report=answer_then_audit('说明 Beta 的过程和作者目的。',
        [{'evidence_id':source_id,'content':operation+purpose}],Model())
    assert len(calls)==2 and operation in report['answer'] and purpose in report['answer']
    assert report['answer_scope']=='MODEL_REVIEWED'
    assert all(f['evidence_ids']==[source_id] for f in report['draft']['facts'])


def test_forged_purpose_source_fails_existing_source_validator():
    projected,_=project_source_purposes(parts())
    with pytest.raises(ValueError):validate_fact_shape(projected['facts'][1],{'process'})
