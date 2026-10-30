import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.context_window import ContextWindowExceeded, enforce_window, split_text
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget, LLMCallPolicy, RunStatus, ToolCallStatus
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.research_notebook import ResearchNotebook, resume_notebook


def notebook(tmp_path):
    db = RuntimeDatabase(tmp_path/'runtime.db')
    rid = db.create_run('统计数据集、方法、模块和论文报告结果', Budget(max_tokens=2000000,max_llm_calls=500), metadata={'conversation_id':'research'})
    return ResearchNotebook(db,rid)


class Extractor:
    def complete(self,messages,**kwargs):
        payload = json.loads(messages[-1]['content'])
        text = payload['source_slice']
        return LLMResponse(text=json.dumps({'notes':[{'kind':'method','title':'source note','statement':'Unreviewed method',
            'quote':text[:100]}]}),finish_reason='stop')


def test_unicode_splitting_loses_no_content():
    text = '变化检测中文🙂abc\n' * 3000
    parts = list(split_text(text,511))
    assert ''.join(x[2] for x in parts)==text
    assert all(len(x[2].encode())<=511 and text[x[0]:x[1]]==x[2] for x in parts)


def test_gate_accounts_for_output_and_does_not_bill(tmp_path):
    book = notebook(tmp_path)
    class Never:
        context_window_tokens = 4096
        def complete(self,*args,**kwargs):
            pytest.fail('Oversized request reached provider')
    rt = PersistentLLMRuntime(book.db,Never())
    with pytest.raises(ContextWindowExceeded):
        rt.complete(book.run_id,[{'role':'system','content':'规则'*3000}],policy=LLMCallPolicy('test',1000,False))
    assert book.db.get_run(book.run_id)['used_llm_calls']==0
    assert book.db.list_context_snapshots(book.run_id)
    with pytest.raises(ContextWindowExceeded):
        enforce_window([{'role':'user','content':'short'}],4000,limit=4096)


def test_many_papers_persist_and_batches_do_not_grow_with_history(tmp_path):
    book = notebook(tmp_path)
    for i in range(50):
        book.add_source(f'paper://sha{i}/chunk',f'paper{i}: module source text. '*60,max_bytes=600)
    before = book.overview()['batches']['PENDING']
    model = Extractor()
    seen = []
    class Recording:
        def complete(self,messages,**kwargs):
            seen.append(len(json.dumps(messages).encode()))
            return model.complete(messages,**kwargs)
    while book.process_next(Recording()):
        pass
    restored = ResearchNotebook(RuntimeDatabase(book.db.path),book.run_id)
    assert restored.overview()['batches']['COMPLETED']==before
    assert len(restored.notes(limit=100))==min(100,before)
    assert max(seen)<6000
    assert restored.process_next(Recording()) is False
    assert len(seen)==before
    note = restored.notes(query='paper49')[0]
    assert note['quote'] in restored.source(note['batch_id'])['source_text']
    assert not note['is_scientific_evidence']


def test_overflow_splits_and_restarts_with_smaller_inputs(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://sha/chunk','a'*6000)
    class Limited(Extractor):
        def complete(self,messages,**kwargs):
            if len(json.loads(messages[-1]['content'])['source_slice'])>1500:
                raise ContextWindowExceeded('Serving limit')
            return super().complete(messages,**kwargs)
    while book.process_next(Limited()):
        pass
    assert book.overview()['batches']=={'COMPLETED':4,'SPLIT':3}
    with book.db.connect() as c:
        leaves = c.execute("SELECT source_text FROM notebook_batches WHERE run_id=? AND state='COMPLETED' ORDER BY start_char",(book.run_id,)).fetchall()
    assert ''.join(r['source_text'] for r in leaves)=='a'*6000


def test_uncertain_call_cannot_be_replayed(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://sha/chunk','source')
    class Lost:
        def complete(self,*args,**kwargs):
            raise TimeoutError('response unknown')
    with pytest.raises(TimeoutError):
        book.process_next(Lost())
    with pytest.raises(RuntimeError,match='uncertain'):
        ResearchNotebook(book.db,book.run_id).process_next(Extractor())


def test_invalid_quote_keeps_raw_output_for_review(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://sha/chunk','source')
    class Invented:
        def complete(self,*args,**kwargs):
            return LLMResponse(text=json.dumps({'notes':[{'kind':'reported_result','title':'fake','statement':'99%', 'quote':'NOT IN SOURCE'}]}),finish_reason='stop')
    assert book.process_next(Invented())
    assert book.overview()['batches']=={'REVIEW':1}
    assert not book.notes()
    with book.db.connect() as c:
        assert 'NOT IN SOURCE' in c.execute('SELECT response FROM notebook_batches').fetchone()[0]


def test_batch_and_experiment_read_scope(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://sha/chunk','private')
    book.process_next(Extractor())
    other_id = book.db.create_run('other',Budget(),metadata={'conversation_id':'other'})
    other = ResearchNotebook(book.db,other_id)
    with pytest.raises(KeyError):
        other.source(book.notes()[0]['batch_id'])
    with pytest.raises(ValueError,match='outside'):
        book.import_experiment_run(other_id)


def test_actual_runtime_mode_reads_every_selected_paper(tmp_path):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    class Store:
        def paper_outline(self,sha):
            return [{'chunk_ids':[sha+'_chunk']}]
        def get_chunks(self,ids):
            return [SimpleNamespace(paper_sha256=cid.split('_')[0],text='original source',stable_id=cid) for cid in ids]
    class Model(Extractor):
        def complete(self,messages,**kwargs):
            kwargs['before_attempt']()
            return super().complete(messages,**kwargs)
    rt = ResearchAgentRuntime(db_path=tmp_path/'run.db',workspace=tmp_path/'work',llm=Model())
    rt.search_adapter = SimpleNamespace(store=Store())
    queued = rt.enqueue(question='分析这些论文',budget=Budget(max_tokens=200000,max_llm_calls=10),
                        metadata={'engine':'research_notebook','paper_sha256s':['sha1','sha2']})
    result = rt.resume(queued['run']['run_id'])
    assert result['run']['status']=='COMPLETED'
    assert result['run']['metadata']['notebook_overview']['batches']['COMPLETED']==2
    assert result['run']['used_llm_calls']==2
    followup = rt.enqueue_follow_up(parent_run_id=queued['run']['run_id'],question='比较方法模块')
    assert followup['run']['metadata']['engine']=='notebook_query'
    assert followup['run']['metadata']['notebook_root_run_id']==queued['run']['run_id']


def test_hundred_experiment_receipts_reload_without_reexecution(tmp_path):
    book = notebook(tmp_path)
    run_id = book.db.create_run('experiment fixtures, not actual training',Budget(),metadata={'conversation_id':'research'})
    step = book.db.create_step(run_id,'test',{})
    for i in range(120):
        cid,_ = book.db.create_tool_call(run_id=run_id,step_id=step,tool_name='experiment.evaluate',
            args={'experiment_id':f'exp{i}','metric':'F1','split':'test-v1','seed':i},risk_level='low',
            idempotency_key=f'fixture{i}',max_attempts=1)
        book.db.update_tool_call(cid,ToolCallStatus.FAILED if i%2 else ToolCallStatus.COMPLETED,
            result={'fixture_only':True,'score':None if i%2 else 0.8,'code_hash':'fixture','data_hash':'fixture'},
            error='synthetic failure' if i%2 else None)
    book.import_experiment_run(run_id)
    book.import_experiment_run(run_id)
    assert book.overview()['notes']['experiment']==120
    assert len(book.notes(kind='experiment',offset=100,limit=100))==20
    assert book.retrieve_sources('exp119')[0]['content'].find('exp119')>=0
    assert 'FAILED' in book.retrieve_sources('exp119')[0]['content']
    assert book.db.get_run(book.run_id)['used_llm_calls']==0


def test_budget_extension_preserves_usage_and_completed_batches(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://sha/chunk','source')
    book.process_next(Extractor())
    book.db.update_run_status(book.run_id,RunStatus.BUDGET_EXHAUSTED)
    budget = Budget.from_dict(book.db.get_run(book.run_id)['budget'])
    budget.max_llm_calls += 20
    book.extend_budget(budget.to_dict())
    assert book.db.get_run(book.run_id)['status']=='PENDING'
    assert book.process_next(Extractor()) is False


def test_query_reloads_source_not_model_statement(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://sha/chunk','Original module text.')
    book.process_next(Extractor())
    loaded = book.retrieve_sources('method')
    assert loaded[0]['content']=='Original module text.'
    assert 'Unreviewed method' not in loaded[0]['content']


def test_notebook_query_enters_audited_reader_with_raw_sources(tmp_path,monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    class Resolver:
        calls = 0
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(text=json.dumps({'question': 'method', 'use_prior_sources': True,
                'needs_clarification': False, 'reason': 'Same notebook research'}), finish_reason='stop')
    resolver = Resolver()
    rt = ResearchAgentRuntime(db_path=tmp_path/'query.db',workspace=tmp_path/'work',llm=resolver)
    rid = rt.db.create_run('research goal',Budget(),metadata={'conversation_id':'c','engine':'research_notebook'})
    book = ResearchNotebook(rt.db,rid)
    book.add_source('paper://sha/chunk','Original method evidence.')
    book.process_next(Extractor())
    rt.db.update_run_status(rid,RunStatus.COMPLETED)
    followup = rt.enqueue_follow_up(parent_run_id=rid,question='继续分析它')
    seen = []
    def reader(db,run_id,evidence,model,**kwargs):
        seen.extend(evidence)
        assert model.stage=='answer_first'
        assert kwargs['question'] == 'method'
        assert db.get_run(run_id)['metadata']['answer_request_history'][0]['user_request'] == 'research goal'
        db.update_run_status(run_id,RunStatus.COMPLETED)
    monkeypatch.setattr('paperpilot_agentic.durable_reader.run_durable_reader',reader)
    result = rt.resume(followup['run']['run_id'])
    assert result['run']['status']=='COMPLETED'
    assert seen[0]['content']=='Original method evidence.'
    assert resolver.calls == 1
    rt.resume(followup['run']['run_id'])
    assert resolver.calls == 1


def test_ambiguous_notebook_followup_clarifies_without_retrieval(tmp_path, monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    class Resolver:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps({'question': '哪一个方法？', 'use_prior_sources': True,
                'needs_clarification': True, 'reason': '请明确是前一种还是后一种方法。'}), finish_reason='stop')
    rt = ResearchAgentRuntime(db_path=tmp_path/'ambiguous.db', workspace=tmp_path/'work', llm=Resolver())
    root = rt.db.create_run('比较两种方法', Budget(), metadata={
        'conversation_id': 'owned', 'engine': 'research_notebook'})
    rt.db.update_run_status(root, RunStatus.COMPLETED)
    child = rt.enqueue_follow_up(parent_run_id=root, question='它怎么样？')['run']['run_id']
    monkeypatch.setattr(ResearchNotebook, 'retrieve_sources', lambda *a: pytest.fail('Ambiguous query retrieved'))
    result = rt.resume(child)
    assert result['run']['metadata']['clarification_required']
    assert result['run']['research_outcome'] == 'UNRESOLVED'
    assert result['final_answer']['answer'] == ''


@pytest.mark.parametrize('cancel', [False, True])
def test_notebook_query_tool_budget_and_cancel_prevent_generation(tmp_path, monkeypatch, cancel):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    rt = ResearchAgentRuntime(db_path=tmp_path/'guard.db', workspace=tmp_path/'work', llm=Extractor())
    root = rt.db.create_run('method', Budget(), metadata={'conversation_id':'c','engine':'research_notebook'})
    rid = rt.db.create_run('method', Budget(max_tool_calls=1 if cancel else 0), metadata={
        'conversation_id':'c', 'engine':'notebook_query', 'notebook_root_run_id':root})
    def retrieve(*args, **kwargs):
        assert cancel, 'Zero-budget query should not retrieve'
        rt.db.update_run_status(rid, RunStatus.CANCELLED)
        return []
    monkeypatch.setattr(ResearchNotebook, 'retrieve_sources', retrieve)
    monkeypatch.setattr('paperpilot_agentic.durable_reader.run_durable_reader',
                        lambda *a, **k: pytest.fail('Stopped query reached generation'))
    result = rt.resume(rid)
    assert result['run']['status'] == ('CANCELLED' if cancel else 'BUDGET_EXHAUSTED')
    assert result['run']['used_tool_calls'] == (1 if cancel else 0)


def test_notebook_scope_filters_sources_before_top_k(tmp_path):
    book = notebook(tmp_path)
    book.add_source('paper://excluded/chunk', 'Original method evidence excluded.')
    book.add_source('paper://allowed/chunk', 'Original method evidence allowed.')
    while book.process_next(Extractor()):
        pass
    rows = book.retrieve_sources('method', limit=1, paper_sha256s={'allowed'})
    assert len(rows) == 1
    assert rows[0]['source_uri'] == 'paper://allowed/chunk'


def test_notebook_exclusion_reaches_both_retrievers_and_next_turn(tmp_path, monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    scopes = []
    class Search:
        def resolve_explicit_title_scope(self, queries, **kwargs):
            return {'b'} if any('Beta' in q for q in queries) else set()
        def search(self, query, *, top_k, paper_sha256s):
            scopes.append(set(paper_sha256s))
            return [{'paper_sha256':'a', 'stable_id':'chunk', 'content':'Original method.'}]
    class Resolver:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps({'question':'method', 'use_prior_sources':True,
                'needs_clarification':False, 'reason':'Continue notebook comparison'}), finish_reason='stop')
    rt = ResearchAgentRuntime(db_path=tmp_path/'scope.db', workspace=tmp_path/'work',
                              llm=Resolver(), search_adapter=Search())
    root = rt.db.create_run('method', Budget(), metadata={'conversation_id':'c',
        'engine':'research_notebook','resolved_paper_sha256s':['a','b']})
    rid = rt.db.create_run('method，排除 Beta', Budget(), metadata={'conversation_id':'c',
        'engine':'notebook_query','notebook_root_run_id':root})
    def notes(self, question, **kwargs):
        assert kwargs['paper_sha256s'] == {'a'}
        return []
    def reader(db, run_id, evidence, model, **kwargs):
        assert all(e['source_uri'].startswith('paper://a/') for e in evidence)
        db.update_run_status(run_id, RunStatus.COMPLETED)
    monkeypatch.setattr(ResearchNotebook, 'retrieve_sources', notes)
    monkeypatch.setattr('paperpilot_agentic.durable_reader.run_durable_reader', reader)
    assert rt.resume(rid)['run']['status'] == 'COMPLETED'
    child = rt.enqueue_follow_up(parent_run_id=rid, question='继续比较')['run']['run_id']
    result = rt.resume(child)
    assert result['run']['status'] == 'COMPLETED'
    assert result['run']['metadata']['resolved_paper_sha256s'] == ['a']
    assert scopes == [{'a'}, {'a'}]
