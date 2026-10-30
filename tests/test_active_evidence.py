import copy
import json

import pytest

from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.models import PaperDocument, PaperChunk, PaperPage
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.active_evidence import ActiveEvidenceHarness
from paperpilot_agentic import active_evidence_ledger as ledger
from paperpilot_agentic.providers import LLMResponse

A, B = 'a'*64, 'b'*64


class Reader:
    calls = 0
    def complete(self,messages,*,before_attempt,**kwargs):
        before_attempt()
        self.calls += 1
        p = json.loads(next(m['content'] for m in messages if m['role']=='user'))
        if p['read_passages'] and ('finish' == p['available_actions'][0] and len(p['available_actions'])==1
                or p['evidence_policy']!='active'):
            r = p['read_passages'][0]
            value = {'action':'finish','arguments':{'answer':'采用特征融合。','verdict':'ANSWERED',
                'claims':[{'fact':'采用特征融合。','source_id':r['source_id'],'quote_id':r['quote_spans'][0]['quote_id']}],'gaps':[]}}
        elif p['read_passages']:
            r = p['read_passages'][0]
            value = {'action':'explore','arguments':{'assessments':[{'requirement':'D1','paper_id':r['paper_id'],
                'status':'SUPPORT_PROPOSED','reason':'原文描述融合。','claims':[{
                    'fact':'采用特征融合。','source_id':r['source_id'],'quote_id':r['quote_spans'][0]['quote_id']}]}]}}
        elif self.calls==1:
            value = {'action':'explore','arguments':{'requirements':[{'dimension':'fusion mechanism','kind':'method','paper_ids':['P1']}],
                'operations':[{'tool':'paper.retrieve_evidence','arguments':{'question':'Alpha fusion','paper_ids':['P1']}}]}}
        else:
            value = {'action':'explore','arguments':{'operations':[{'tool':'paper.read_sources','arguments':{'source_ids':['S1']}}]}}
        value['reason'] = 'Read actual evidence before answering'
        return LLMResponse(text=json.dumps(value),prompt_tokens=40,completion_tokens=80,finish_reason='stop',model='test')


def setup(tmp_path, *, model=None, text=None):
    store = PaperKnowledgeStore(tmp_path/'papers.sqlite3',namespace='test')
    documents = [PaperDocument(id='P1',file_name='alpha.pdf',sha256=A,title='Alpha fusion',page_count=1,processed_pages=1,
        pages=[PaperPage(number=1,text='Alpha')]),
        PaperDocument(id='P2',file_name='beta.pdf',sha256=B,title='Beta attention',page_count=1,processed_pages=1,
        pages=[PaperPage(number=1,text='Beta')])]
    chunks = [PaperChunk(id='P1:p1:c1',paper_id='P1',page_start=1,page_end=1,section='Methods',
        text=text or 'Alpha combines aligned temporal feature maps using learned feature fusion.'),
        PaperChunk(id='P2:p1:c1',paper_id='P2',page_start=1,page_end=1,section='Methods',text='Beta uses attention.')]
    store.sync_documents(documents,chunks)
    adapter = PaperPilotKnowledgeSearchAdapter(store,rerank_stage='final')
    runtime = ResearchAgentRuntime(workspace=tmp_path,db_path=tmp_path/'runs.db',search_adapter=adapter,llm=model or Reader())
    h = ActiveEvidenceHarness(runtime)
    rid = h.start('Alpha fusion mechanism','active-request-01',paper_ids=[A])['run']['run_id']
    return runtime,h,rid


def test_actual_episode_roundtrip_uses_only_read_sources_and_keeps_default_opt_in(tmp_path):
    runtime,h,rid = setup(tmp_path)
    result = runtime.resume(rid)
    assert result['run']['status']=='COMPLETED', result.get('active_evidence')
    assert result['final_answer']['verdict']=='ANSWERED', result['final_answer']
    assert result['run']['used_llm_calls']==4
    assert result['run']['used_tool_calls']==3
    assert result['active_evidence']['read_ranges']==1
    assert result['final_answer']['scientific_validity']=='NOT_INDEPENDENTLY_VERIFIED'
    assert result['final_answer']['citation_ids'] and runtime.db.list_evidence(rid)
    calls = runtime.llm_runtime.model.calls
    assert runtime.resume(rid)['final_answer']==result['final_answer']
    assert runtime.llm_runtime.model.calls==calls
    legacy = runtime.enqueue(question='ordinary question')['run']
    assert legacy['metadata'].get('retrieval_profile','legacy')=='legacy'


def test_navigation_preview_cannot_become_a_citation(tmp_path):
    _,h,rid = setup(tmp_path)
    s = h._load(rid)
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    with pytest.raises(ValueError,match='Citation quote absent'):
        ledger.admit_final(s,{'answer':'fusion','verdict':'ANSWERED','claims':[{
            'fact':'fusion','source_id':'S1','quote':'Alpha'}],'gaps':[]})
    with pytest.raises(ValueError,match='unknown|outside'):
        h.sources.retrieve_evidence(s,'Beta',['P2'])


def test_tail_cursor_is_exact_bounded_and_neighbor_does_not_escape_scope(tmp_path):
    text = 'Background. '*180 + 'TAIL_NEEDLE uses adaptive fusion.'
    _,h,rid = setup(tmp_path,text=text)
    s = h._load(rid)
    result = h.sources.retrieve_evidence(s,'TAIL_NEEDLE',['P1'],mode='literal')
    assert result['candidates'][0]['source_id']=='S1'
    first = h.sources.read_sources(s,['S1'],neighbors=1)
    assert not first['passages'][0]['unit_complete']
    second = h.sources.read_sources(s,**first['continuations'][0])
    assert 'TAIL_NEEDLE' in second['passages'][0]['content']
    for r in first['passages']+second['passages']:
        assert text[r['range'][0]:r['range'][1]]==r['content']
        assert r['paper_sha256']==A
    before = len(s['read_handles'])
    h.sources.read_sources(s,['S1'])
    assert len(s['read_handles'])==before
    with pytest.raises(ValueError,match='Cursor'):
        h.sources.read_sources(s,['S1'],cursor=len(text))


def test_source_version_changes_and_final_numeric_invention_are_rejected(tmp_path):
    runtime,h,rid = setup(tmp_path)
    s = h._load(rid)
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    h.sources.read_sources(s,['S1'])
    with pytest.raises(ValueError,match='Numeric claim'):
        ledger.admit_final(s,{'answer':'93.5','verdict':'INSUFFICIENT','claims':[{
            'fact':'F1 is 93.5','source_id':'R1','quote':s['read_handles']['R1']['content']}],'gaps':['missing']})
    with runtime.search_adapter.store._connect() as c:
        c.execute("UPDATE chunks SET section='Changed'")
    with pytest.raises(ValueError,match='version changed'):
        h._guard(rid,s)


def test_one_supplement_and_no_progress_finishes_exploration(tmp_path):
    _,h,rid = setup(tmp_path)
    s = h._load(rid)
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    h.sources.retrieve_evidence(s,'missing details',['P1'])
    with pytest.raises(ValueError,match='one targeted supplement'):
        h.sources.retrieve_evidence(s,'again',['P1'])
    s['no_progress']=2
    h._after_action(rid,s,'lead',{'status':'ACCEPTED'})
    assert h.allowed_actions(s)=={'finish'} and s['stop_reason']=='NO_PROGRESS'


def test_saved_response_restores_without_a_second_external_request(tmp_path,monkeypatch):
    runtime,h,rid = setup(tmp_path)
    import paperpilot_agentic.episode_runner as runner
    original = runner.immutable
    def crash(folder,name,content):
        if name=='turn-01-response.json':
            raise SystemExit('after provider response was saved')
        return original(folder,name,content)
    monkeypatch.setattr(runner,'immutable',crash)
    with pytest.raises(SystemExit):
        h.advance(rid,max_turns=1)
    assert runtime.db.get_run(rid)['used_llm_calls']==1
    monkeypatch.setattr(runner,'immutable',original)
    recovered = h.advance(rid,max_turns=1)
    assert recovered['run']['used_llm_calls']==1
    assert h._load(rid)['turn']==1
    assert runtime.llm_runtime.model.calls==1


def test_known_tool_receipt_reuses_handles_after_local_commit_interruption(tmp_path,monkeypatch):
    runtime,h,rid = setup(tmp_path)
    old = h._save
    def crash(rid,state,**kw):
        if kw.get('applied_turn')==1:
            raise SystemExit('local action returned, task not committed')
        return old(rid,state,**kw)
    monkeypatch.setattr(h,'_save',crash)
    with pytest.raises(SystemExit):
        h.advance(rid,max_turns=1)
    charged = runtime.db.get_run(rid)['used_tool_calls']
    monkeypatch.setattr(h,'_save',old)
    h.advance(rid,max_turns=1)
    assert runtime.db.get_run(rid)['used_tool_calls']==charged
    assert runtime.db.get_run(rid)['used_llm_calls']==1
    assert list(h._load(rid)['source_handles'])==['S1']


def test_unknown_response_never_resends_or_resets_budget(tmp_path):
    class Lost(Reader):
        def complete(self,messages,*,before_attempt,**kwargs):
            before_attempt()
            self.calls+=1
            raise TimeoutError('unknown response')
    runtime,h,rid = setup(tmp_path,model=Lost())
    h.advance(rid)
    assert h._load(rid)['phase']=='UNKNOWN'
    used = runtime.db.get_run(rid)['used_llm_calls']
    h.advance(rid)
    assert runtime.llm_runtime.model.calls==1 and runtime.db.get_run(rid)['used_llm_calls']==used


def test_ledger_reading_alone_is_not_support_and_wrong_paper_is_rejected(tmp_path):
    _,h,rid = setup(tmp_path)
    s = h._load(rid)
    ledger.requirements(s,[{'dimension':'fusion','kind':'method','paper_ids':['P1']}])
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    h.sources.read_sources(s,['S1'])
    assert ledger.coverage(s)[0]['status']=='READ' and not ledger.ready(s)
    with pytest.raises(ValueError,match='requires a read source'):
        ledger.assess(s,[{'requirement':'D1','paper_id':'P1','status':'SUPPORT_PROPOSED','reason':'trust me','claims':[]}])
    before = copy.deepcopy(s)
    with pytest.raises(ValueError,match='registered requirement'):
        ledger.assess(s,[{'requirement':'D1','paper_id':'P9','status':'NOT_FOUND','reason':'wrong','claims':[]}])
    assert s==before


def test_scope_repair_preserves_requirements_and_failed_tables_cannot_be_laundered(tmp_path):
    _,h,rid = setup(tmp_path)
    s = h._load(rid)
    ledger.requirements(s,[{'dimension':'fusion','kind':'method','paper_ids':['P1']}])
    with pytest.raises(ValueError,match='dimensions are frozen'):
        ledger.requirements(s,[{'dimension':'easier question','kind':'method','paper_ids':['P1']}])
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    h.sources.read_sources(s,['S1'])
    s['read_handles']['R1']['source_quality']['failed_table_transcription']=True
    s['read_handles']['R1']['content']='F1 93.5'
    with pytest.raises(ValueError,match='laundered'):
        ledger.admit_final(s,{'answer':'F1 93.5','verdict':'INSUFFICIENT','claims':[{
            'fact':'F1 93.5','source_id':'R1','quote':'F1 93.5'}],'gaps':['Unverified grid']})
    with pytest.raises(ValueError,match='immutable original'):
        h._guard(rid,s)


def test_predispatch_budget_block_publishes_partial_without_calling_model(tmp_path):
    runtime,h,rid = setup(tmp_path)
    with runtime.db.transaction() as c:
        c.execute('UPDATE runs SET used_tokens=59500 WHERE run_id=?',(rid,))
    result = h.advance(rid)
    assert result['final_answer']['verdict']=='INSUFFICIENT'
    assert result['run']['used_llm_calls']==0 and runtime.llm_runtime.model.calls==0
    assert result['active_evidence']['stop_reason']=='REQUIRED_CONTEXT_EXCEEDS_BUDGET'


def test_workbench_opt_in_and_publication_recovery(tmp_path,monkeypatch):
    runtime,h,_ = setup(tmp_path)
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background
    wb = ResearchWorkbench(runtime,Background())
    result = wb.submit({'request_id':'paper-active-web-01','question':'Alpha fusion mechanism',
        'action':'paper_active','paper_ids':[A]})
    rid = result['run_id']
    assert result['start'] and result['conversation_id']
    h.advance(rid)
    calls = runtime.llm_runtime.model.calls
    runtime.db.update_run_metadata(rid,{'final_answer':None})
    h.advance(rid)
    assert runtime.status(rid)['final_answer'] and runtime.llm_runtime.model.calls==calls


def test_quote_identifier_preserves_pdf_ligatures_and_line_breaks(tmp_path):
    text = 'The ﬁnal output\nuses aligned fusion. More details follow.'
    _,h,rid = setup(tmp_path,text=text)
    s = h._load(rid)
    h.sources.retrieve_evidence(s,'fusion',['P1'])
    h.sources.read_sources(s,['S1'])
    claim = ledger.canonical_claims(s,[{'fact':'使用融合。','source_id':'R1','quote_id':'Q0'}])[0]
    assert claim['quote'] in text and 'ﬁnal' in claim['quote'] and '\n' in claim['quote']
    with pytest.raises(ValueError,match='actually read'):
        ledger.canonical_claims(s,[{'fact':'invented','source_id':'S1','quote_id':'Q0'}])
    with pytest.raises(ValueError,match='actually read'):
        ledger.canonical_claims(s,[{'fact':'invented','source_id':'R1','quote_id':'Q99'}])


def test_echoed_requirement_id_is_accepted_but_cannot_rebind(tmp_path):
    _,h,rid = setup(tmp_path)
    s = h._load(rid)
    ledger.requirements(s,[{'dimension':'fusion','kind':'method','paper_ids':['P1']}])
    ledger.requirements(s,copy.deepcopy(s['requirements']))
    bad = copy.deepcopy(s['requirements'])
    bad[0]['id']='D9'
    with pytest.raises(ValueError,match='host-assigned'):
        ledger.requirements(s,bad)


def test_static_is_one_pass_even_when_model_requests_a_tool(tmp_path):
    class Invalid(Reader):
        def complete(self,messages,*,before_attempt,**kwargs):
            before_attempt()
            self.calls+=1
            return LLMResponse(text='{"action":"explore","arguments":{}}',prompt_tokens=40,completion_tokens=20,finish_reason='stop')
    runtime,h,_ = setup(tmp_path,model=Invalid())
    rid=h.start('Alpha fusion mechanism','static-onepass-case',paper_ids=[A],mode='static')['run']['run_id']
    result=h.advance(rid)
    assert result['run']['used_llm_calls']==1 and result['final_answer']['verdict']=='INSUFFICIENT'


def test_read_before_locating_recovers_through_explicit_navigation(tmp_path):
    class EarlyReader(Reader):
        def complete(self,messages,*,before_attempt,**kwargs):
            if self.calls >= 2:
                return super().complete(messages,before_attempt=before_attempt,**kwargs)
            before_attempt()
            self.calls += 1
            p = json.loads(next(m['content'] for m in messages if m['role']=='user'))
            assert 'paper.read_sources' not in p['tools']
            assert p['navigation']['next_step']=='RETRIEVE_EVIDENCE_FOR_P_HANDLES'
            if self.calls == 1:
                value={'action':'explore','arguments':{
                    'requirements':[{'dimension':'fusion mechanism','kind':'method','paper_ids':['P1']}],
                    'operations':[{'tool':'paper.read_sources','arguments':{'source_ids':['P1_S1']}}]}}
            else:
                assert 'SOURCE_NOT_LOCATED' in p['last_observation']['error']
                value={'action':'explore','arguments':{'operations':[
                    {'tool':'paper.retrieve_evidence','arguments':{'question':'Alpha fusion','paper_ids':['P1']}}]}}
            return LLMResponse(text=json.dumps(value),prompt_tokens=40,completion_tokens=80,finish_reason='stop',model='test')

    runtime,h,rid = setup(tmp_path,model=EarlyReader())
    result=runtime.resume(rid)
    assert result['final_answer']['verdict']=='ANSWERED'
    assert result['run']['used_llm_calls']==5
    assert result['active_evidence']['read_ranges']==1
    assert result['active_evidence']['action_trace'][0]['status']=='REJECTED'


def test_navigation_survives_lost_chat_preview_and_separates_read_errors(tmp_path):
    _,h,rid=setup(tmp_path)
    s=h._load(rid)
    h.sources.retrieve_evidence(s,'Alpha',['P1'])
    s['last_observation']={'status':'REJECTED','error':'An unrelated action failed'}
    s['contexts']={}
    _,messages=h._context(rid,s)
    p=json.loads(messages[-1]['content'])
    assert p['navigation']['known_source_ids']==['S1']
    assert p['navigation']['located_sources'][0]['paper_id']=='P1'
    assert 'paper.read_sources' in p['tools']
    with pytest.raises(ValueError,match='UNKNOWN_SOURCE_HANDLE'):
        h.sources.read_sources(s,['P1_S1'])
    with pytest.raises(ValueError,match='INVALID_READ_RANGE'):
        h.sources.read_sources(s,['S1'],neighbors=None)
    assert not s['read_handles']  # Navigation and rejected actions cannot become evidence.


def test_read_evidence_is_discoverable_and_handed_to_prototype_without_model_calls(tmp_path):
    runtime,h,rid=setup(tmp_path)
    runtime.resume(rid)
    from paperpilot_agentic.workbench import ResearchWorkbench
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from test_workbench import Background
    wb=ResearchWorkbench(runtime,Background())
    materials=wb.materials(rid)
    reader=next(r for r in materials['readers'] if r['run_id']==rid)
    selected=reader['sources'][0]
    assert selected['assurance']=='READ_SOURCE_ORIGIN_ONLY_NOT_ENTAILMENT'
    assert selected['paper_sha256']==A and selected['source_uri'].startswith('paper://')
    assert selected['range']==h._load(rid)['read_handles']['R1']['range']
    parent=runtime.db.get_run(rid)
    calls=runtime.llm_runtime.model.calls
    payload={'action':'research_prototype','request_id':'active-prototype-handoff',
        'parent_run_id':rid,'conversation_id':parent['metadata']['conversation_id'],
        'source_ids':[selected['id']],'question':'Design a temporal feature fusion model',
        'prototype_mode':'simple_loop'}
    submission=wb.submit(payload)
    state=PrototypeHarness(runtime)._load(submission['run_id'])
    source=state['sources'][selected['id']]
    assert source['content']==h._load(rid)['read_handles']['R1']['content']
    assert source['provenance']['index_identity']==selected['index_identity']
    assert state['required_evidence_ids']==[selected['id']]
    assert state['work']['tasks']['design']['depends_on']==['read0']
    assert runtime.llm_runtime.model.calls==calls  # Submission schedules, never fabricates an executed prototype.
    assert wb.submit(payload)['run_id']==submission['run_id']
    assert wb.background.submitted==[submission['run_id']]
