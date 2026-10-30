import copy
import json

import pytest

from test_active_evidence import setup
from paperpilot_agentic import active_evidence_ledger as ledger
from paperpilot_agentic.paper_method_identity import method_hints, check_claim_identity


def located(tmp_path):
    runtime,h,rid=setup(tmp_path)
    s=h._load(rid)
    ledger.requirements(s,[{'dimension':'fusion','kind':'method','paper_ids':['P1']}])
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    return runtime,h,rid,s


def test_safe_parameter_normalization_preserves_exact_read_and_receipt(tmp_path):
    _,h,_,s=located(tmp_path)
    request={'source_ids':['S1'],'neighbors':True,'cursor':None}
    original=copy.deepcopy(request)
    result=h.sources.invoke(s,'paper.read_sources',request)
    assert request==original
    assert result['passages'][0]['content']==h.sources.units[s['source_handles']['S1']]['text']
    assert result['argument_normalization']['effective_arguments']=={'source_ids':['S1'],'neighbors':1}
    with pytest.raises(ValueError,match='INVALID_READ_RANGE'):
        h.sources.invoke(s,'paper.read_sources',{'source_ids':['S1'],'neighbors':2})
    with pytest.raises(ValueError,match='UNKNOWN_SOURCE_HANDLE'):
        h.sources.invoke(s,'paper.read_sources',{'source_ids':['S99'],'neighbors':True})


def test_retrieval_aliases_and_requirement_queries_cannot_change_scope(tmp_path):
    _,h,rid=setup(tmp_path)
    s=h._load(rid)
    ledger.requirements(s,[{'dimension':'fusion','kind':'method','paper_ids':['P1']}])
    with pytest.raises(ValueError,match='Conflicting'):
        h.sources.invoke(s,'paper.retrieve_evidence',{'query':'Alpha','question':'Beta','paper_id':'P1'})
    with pytest.raises(ValueError,match='existing requirement_ids'):
        h.sources.invoke(s,'paper.retrieve_evidence',{'requirement_ids':['D9']})
    assert s['searches']==0
    result=h.sources.invoke(s,'paper.retrieve_evidence',{'query':'Alpha','paper_id':'P1'})
    assert result['candidates'][0]['paper_id']=='P1'
    next_result=h.sources.invoke(s,'paper.retrieve_evidence',{'requirement_ids':['D1']})
    assert next_result['argument_normalization']['effective_arguments']['question']=='fusion'
    with pytest.raises(ValueError,match='one targeted supplement'):
        h.sources.invoke(s,'paper.retrieve_evidence',{'question':'again','paper_ids':['P1']})


def test_bad_requirement_amendment_does_not_block_read_or_apply_assessments(tmp_path):
    _,h,rid,s=located(tmp_path)
    old=copy.deepcopy(s['requirements'])
    action={'action':'explore','arguments':{
        'requirements':[{'dimension':'easier','kind':'method','paper_ids':['P1']}],
        'assessments':[{'invalid':'must not be applied against rejected needs'}],
        'operations':[{'tool':'paper.read_sources','arguments':{'source_ids':['S1'],'neighbors':True}}]}}
    result=h._apply(rid,s,action,1)
    assert result['status']=='ACCEPTED' and result['warnings'][0]['kind']=='REQUIREMENT_UPDATE_REJECTED'
    assert s['read_handles'] and s['requirements']==old and s['ledger']=={}
    assert not ledger.ready(s)
    assert result['new_progress']
    again=h._apply(rid,s,action,2)
    assert not again['new_progress'] and s['no_progress']==1


def test_identity_hints_require_self_naming_not_related_work():
    units={'u1':{'paper_sha':'a','page_start':1,'text':'We propose a temporal transformer (BIT) to model context.'},
           'u2':{'paper_sha':'b','page_start':1,'text':'Prior work proposes BIT. We compare BIT and Transformer (BIT).'},
           'u3':{'paper_sha':'b','page_start':8,'text':'We introduce a temporal transformer (BIT).'}}
    hints=method_hints(units)
    assert list(hints)==['a'] and hints['a'][0]['name']=='BIT'
    r=hints['a'][0]['range']
    assert units['u1']['text'][r[0]:r[1]]==hints['a'][0]['quote']


def test_wrong_paper_handle_and_method_subject_are_rejected_but_related_work_is_allowed(tmp_path):
    _,h,_,s=located(tmp_path)
    h.sources.read_sources(s,['S1'])
    s['method_identity_hints']={'P2':[{'name':'BIT'}]}
    with pytest.raises(ValueError,match='CLAIM_PAPER_MISMATCH'):
        ledger.canonical_claims(s,[{'fact':'BIT(P2) uses fusion','source_id':'R1','quote_id':'Q0'}])
    with pytest.raises(ValueError,match='CLAIM_METHOD_MISMATCH'):
        ledger.canonical_claims(s,[{'fact':'BIT uses fusion','source_id':'R1','quote_id':'Q0'}])
    check_claim_identity(s,{'fact':'BIT is discussed as related work','source_id':'R1','quote':'Prior work BIT uses tokens.'})
    s['method_identity_hints']['P1']=[{'name':'BIT'}]  # Ambiguous hints cannot prove a mismatch.
    check_claim_identity(s,{'fact':'BIT uses fusion','source_id':'R1','quote':'fusion'})


def test_final_view_retains_host_read_inventory(tmp_path):
    _,h,rid,s=located(tmp_path)
    h.sources.read_sources(s,['S1'])
    s['answer_mode']=True
    _,messages=h._context(rid,s)
    payload=json.loads(messages[-1]['content'])
    assert payload['papers'][0]['read_source_ids']==['R1']
    assert 'tools' not in payload
