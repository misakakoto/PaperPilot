"""A returned audit can be downgraded when format repair cannot be dispatched."""
import json
import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.providers import LLMResponse


def test_no_repair_budget_keeps_only_independently_audited_legal_fact():
    calls = []
    class Model:
        def ensure_capacity(self, requests):
            if len(calls) >= 2:
                raise LLMBudgetExhausted('No physical requests remain')
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) == 1:
                value = {'facts': [
                    {'id':'good', 'text':'该方法使用线性分类器。', 'evidence_ids':['e1']},
                    {'id':'bad', 'text':'该方法使用随机森林。', 'evidence_ids':['e2']}],
                    'direct_answer_id':'good','limitations':[]}
            else:
                value = {'assessments': [
                    {'fact_id':'good','verdict':'SUPPORTED','relevance':'DIRECT','span_ids':['s0']},
                    {'fact_id':'bad','verdict':'SUPPORTED','relevance':'DIRECT','span_ids':['s1']*13}],
                    'coverage':'COMPLETE','missing_aspects':[]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    r=answer_then_audit('说明分类器。',[
        {'evidence_id':'e1','content':'The method uses a linear classifier.'},
        {'evidence_id':'e2','content':'The method uses a multilayer perceptron.'}],Model())
    assert '该方法使用线性分类器。' in r.get('answer','')
    assert '随机森林' not in r['answer']
    assert r['answer_scope']=='PARTIAL'
    assert r['provider_attempts']==len(calls)==2
    assert r['audit_contract_fallback']['reason']=='REPAIR_BUDGET_UNAVAILABLE'


@pytest.mark.parametrize('defect',['all_support_invalid','truncated','unknown'])
def test_no_repair_budget_never_publishes_unreviewed_or_unknown_output(defect):
    calls=[]
    class Model:
        def ensure_capacity(self, requests):
            if len(calls)>=2:raise LLMBudgetExhausted('No physical requests remain')
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls)==1:
                value={'facts':[{'id':'f','text':'该方法使用线性分类器。','evidence_ids':['e']}],
                       'direct_answer_id':'f','limitations':[]}
            elif defect=='unknown':raise OSError('Outcome unknown; do not resend')
            elif defect=='truncated':return LLMResponse('{"assessments":',finish_reason='length')
            else:value={'assessments':[{'fact_id':'f','verdict':'SUPPORTED',
                       'relevance':'DIRECT','span_ids':['s0']*13}],'coverage':'COMPLETE'}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    r=answer_then_audit('说明分类器。',[{'evidence_id':'e',
        'content':'The method uses a linear classifier.'}],Model())
    assert not r.get('answer')
    assert r['status']=='FAILED'
    assert len(calls)==r['provider_attempts']==2
