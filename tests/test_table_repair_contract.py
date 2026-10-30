"""A language repair must not invalidate a legal host-owned table selection."""
from copy import deepcopy
import json
import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse
from test_shared_packet_cost import fixture


def test_actual_writer_language_repair_preserves_table_selection_contract():
    calls = []
    selection = None

    class Model:
        def complete(self, messages, **kwargs):
            nonlocal selection
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) == 1:
                payload = json.loads(messages[-1]['content'])
                row = payload['table_rows'][0]
                selection = {'id': 'f_table', 'kind': 'TABLE_FACT', 'operation': 'value',
                    'row_refs': [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'], 'metrics': ['F1']}]}
                value = {'facts': [selection], 'direct_answer_id': 'f_table',
                         'limitations': ['The supplied evidence does not establish generalization.']}
            elif len(calls) == 2:
                value = {'facts': [deepcopy(selection)], 'direct_answer_id': 'f_table',
                         'limitations': ['当前证据无法确定其他数据集的结果。']}
            else:
                value = {'assessments': [{'fact_id': 'f_table', 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': ['s0']}], 'coverage': 'COMPLETE',
                    'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY',
                                               'reason': 'Limited evidence scope.'}]}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')

    report = answer_then_audit('请用中文报告 Table IV 的一个 F1 结果。', [fixture()], Model())
    assert len(calls) == 3
    instruction = calls[1][-1]['content']
    assert 'Each fact kind is exactly SOURCE_FACT or INFERENCE' not in instruction
    assert 'kind must be SOURCE_FACT or INFERENCE' not in instruction
    assert 'facts require citations' not in instruction
    assert 'TABLE_FACT' in instruction
    assert report['draft_repair']['attempted']
    assert report['repaired_draft']['facts'][0] == report['raw_draft']['facts'][0]
    assert report['draft']['facts'][0]['kind'] == 'SOURCE_FACT'
    assert '3.25' in report['answer']
    assert report['provider_attempts'] == 3


def test_host_keyed_table_replacement_decodes_aliases_and_projects_values_before_audit():
    calls=[]
    source=fixture()
    source['evidence_id']='evidence_'+'a'*32
    class Model:
        def ensure_capacity(self,requests):pass
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            payload=json.loads(messages[-1]['content'])
            if len(calls)==1:
                value={'facts':[
                    {'id':'good','kind':'SOURCE_FACT','text':'实验使用Delta数据集。','evidence_ids':['ref1']},
                    {'id':'bad','kind':'SOURCE_FACT','text':'Table IV的F1=4.50。','evidence_ids':['ref1']}],
                    'direct_answer_id':'good','limitations':[]}
            elif len(calls)==3:
                assert payload['replacement_contract']['target_ids']==['bad']
                assert '"id":"f4"' not in messages[0]['content']
                row=payload['table_rows'][1]
                value={'replacements':{'bad':{'kind':'TABLE_FACT','operation':'value',
                    'row_refs':[{'evidence_id':row['evidence_id'],'row_id':row['row_id'],
                                 'metrics':['F1']}]}},'limitations':[]}
            else:
                if len(calls)==4:
                    measured=next(f for f in payload['facts'] if f['id']=='bad')
                    assert measured['evidence_ids']==['ref1']
                    assert '4.50' in measured['text']
                    assert measured['kind']=='SOURCE_FACT'
                value={'assessments':[{'fact_id':f['id'],'verdict':'SUPPORTED',
                    'relevance':'DIRECT','span_ids':['s0']} for f in payload['facts']],
                    'coverage':'COMPLETE','missing_aspects':[]}
            return LLMResponse(json.dumps(value,ensure_ascii=False),finish_reason='stop')
    report=answer_then_audit('说明数据集，并报告Table IV的一个F1结果。',[source],Model())
    assert report['draft_isolation']['repair_status']=='PATCH_AUDITED',report.get('failure_detail')
    assert len(calls)==4
    assert '4.50' in report['answer']
    assert report['draft']['facts'][1]['id']=='bad'
    assert report['draft']['facts'][1]['evidence_ids']==[source['evidence_id']]
    assert 'id' not in report['raw_local_repair']['replacements']['bad']


@pytest.mark.parametrize('defect', ['forged_row', 'raw_number'])
def test_correcting_the_repair_schema_does_not_approve_unbound_values(defect):
    calls = []
    selection = None

    class Model:
        def complete(self, messages, **kwargs):
            nonlocal selection
            kwargs['before_attempt']()
            calls.append(messages)
            if selection is None:
                row = json.loads(messages[-1]['content'])['table_rows'][0]
                selection = {'id': 'f_table', 'kind': 'TABLE_FACT', 'operation': 'value',
                    'row_refs': [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'], 'metrics': ['F1']}]}
            fact = deepcopy(selection)
            if defect == 'forged_row':
                fact['row_refs'][0]['row_id'] = 'invented_row'
            elif len(calls) == 2:
                fact = {'id': 'f_table', 'kind': 'SOURCE_FACT', 'text': 'Table IV 的 F1=3.25。',
                        'evidence_ids': [selection['row_refs'][0]['evidence_id']], 'depends_on': []}
            value = {'facts': [fact], 'direct_answer_id': 'f_table',
                     'limitations': ['English caveat.'] if len(calls) == 1 else []}
            return LLMResponse(text=json.dumps(value), finish_reason='stop')

    report = answer_then_audit('请用中文报告 Table IV 的一个 F1 结果。', [fixture()], Model())
    assert report['status'] == 'FAILED'
    assert not report.get('answer')
    assert len(calls) == report['provider_attempts'] == 2
    assert report['table_projection_errors']


def test_table_diagnostics_are_enabled_only_for_the_owned_selection_schema():
    from paperpilot_agentic.answer_contracts import draft_repair_diagnostics
    draft = {'facts': [{'id': 'f', 'kind': 'TABLE_FACT', 'operation': 'value', 'row_refs': []}],
             'limitations': ['English caveat.']}
    legacy = draft_repair_diagnostics(draft, question='请用中文回答。', primary=ValueError('Language'))
    owned = draft_repair_diagnostics(draft, question='请用中文回答。', primary=ValueError('Language'),
                                     allow_table_facts=True)
    assert 'kind must be SOURCE_FACT or INFERENCE' in legacy
    assert 'kind must be SOURCE_FACT or INFERENCE' not in owned
    assert 'facts require citations' not in owned
    assert 'limitation_index=0' in owned


@pytest.mark.parametrize('ending', ['.', ')', '%.'])
def test_rejected_numeric_inference_reports_how_to_preserve_the_comparison(ending):
    from paperpilot_agentic.table_projection import project_table_facts
    from test_table_fact_projection import rows
    draft = {'facts': [{'id': 'ranking', 'kind': 'INFERENCE',
                       'text': 'Beta is best on Delta with F1=63.40' + ending,
                       'evidence_ids': ['t'], 'depends_on': ['measured']}],
             'direct_answer_id': 'ranking', 'limitations': []}
    before = deepcopy(draft)
    with pytest.raises(ValueError) as error:
        project_table_facts(draft, rows())
    assert 'Catalogued table numbers require TABLE_FACT projection' in str(error.value)
    assert 'retain the requested winner/comparison' in str(error.value)
    assert 'remove repeated measurement literals' in str(error.value)
    assert draft == before
