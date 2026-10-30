import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.evidence_selection import select_evidence
from paperpilot_agentic.reader_evidence_selection import select_reader_evidence
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime, LLMBudgetExhausted
from paperpilot_agentic.models import Budget, RunStatus
from paperpilot_agentic.providers import LLMResponse


def candidates(n=24):
    return [dict(stable_id=str(i),paper_sha256='sha',content='A source fact.') for i in range(n)]


class Provider:
    def __init__(self, invalid=None):
        self.invalid = invalid
        self.calls = 0
    def complete(self, messages, *, before_attempt=lambda:None, **kwargs):
        before_attempt()
        self.calls += 1
        data = json.loads(next(m['content'] for m in messages if m['role']=='user'))
        ids = [p['id'] for p in data['passages']][:8]
        if self.invalid == 'duplicate':
            ids[-1] = ids[0]
        elif self.invalid == 'unknown':
            ids[-1] = 'injected'
        elif self.invalid == 'short':
            ids.pop()
        return LLMResponse(text=json.dumps({'ranking':ids}),prompt_tokens=100,completion_tokens=30,
                           finish_reason='length' if self.invalid=='truncated' else 'stop')


@pytest.mark.parametrize('count',[1,7,8,24])
def test_select_only_required_original_ids(count):
    rows = candidates(count)
    result = select_evidence('question',rows,Provider())
    assert len(result['hits']) == min(8,count)
    assert all(hit in rows for hit in result['hits'])


@pytest.mark.parametrize('invalid',['duplicate','unknown','short','truncated'])
def test_bad_selections_are_rejected_without_repair(invalid):
    model = Provider(invalid)
    with pytest.raises(ValueError):
        select_evidence('question',candidates(),model)
    assert model.calls == 1


def runtime(tmp_path, provider, budget=None):
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('question',budget or Budget())
    return SimpleNamespace(db=db,llm_runtime=PersistentLLMRuntime(db,provider)),rid


def test_explicit_fallback_preserves_original_order_and_failure_receipt(tmp_path):
    provider = Provider('unknown')
    rt,rid = runtime(tmp_path,provider)
    baseline = candidates()[8:16]
    result = select_reader_evidence(rt,rid,'question',candidates(),baseline,{'sha'})
    assert [r['stable_id'] for r in result] == [r['stable_id'] for r in baseline]
    trace = rt.db.get_run(rid)['metadata']['reader_evidence_selection']
    assert trace['status'] == 'DEGRADED' and trace['error_type'] == 'ValueError'
    assert trace['fallback'] == 'legacy_retrieval'
    assert rt.db.get_run(rid)['used_llm_calls'] == 1 and provider.calls == 1


def test_provider_internal_retry_cannot_bypass_single_attempt_limit(tmp_path):
    class RetryProvider:
        calls = 0
        def complete(self,messages,*,before_attempt,**kwargs):
            before_attempt()
            self.calls += 1
            # Simulate the provider attempting its built-in retry after a failure.
            before_attempt()
            self.calls += 1
            raise AssertionError('retry must never execute')
    provider = RetryProvider()
    rt,rid = runtime(tmp_path,provider)
    result = select_reader_evidence(rt,rid,'question',candidates(),candidates()[:8],{'sha'})
    assert len(result) == 8 and provider.calls == 1
    trace = rt.db.get_run(rid)['metadata']['reader_evidence_selection']
    assert trace['status'] == 'DEGRADED' and trace['provider_attempts'] == 1
    assert trace['error_type'] == 'SelectionRetryBlocked'
    assert rt.db.get_run(rid)['used_llm_calls'] == 1


@pytest.mark.parametrize('stop',['budget','cancelled','scope'])
def test_safety_stops_never_become_quality_fallback(tmp_path,stop):
    provider = Provider()
    rt,rid = runtime(tmp_path,provider,Budget(max_llm_calls=2) if stop=='budget' else None)
    rows = candidates()
    if stop == 'cancelled':
        rt.db.update_run_status(rid,RunStatus.CANCELLED)
    if stop == 'scope':
        rows[0]['paper_sha256'] = 'outside'
    error = {'budget':LLMBudgetExhausted,'cancelled':InterruptedError,'scope':ValueError}[stop]
    with pytest.raises(error):
        select_reader_evidence(rt,rid,'question',rows,candidates()[:8],{'sha'})
    assert provider.calls == 0


def test_runtime_candidate_api_matches_frozen_pre_diversity_probe(tmp_path):
    from evals.qasper.candidate_budget_comparison import ProbeAdapter
    from evals.qasper.agent_case import paper_records
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_retrieval import PaperKnowledgeStore
    class Ranker:
        def score(self,q,passages):
            return [float(len(p)) for p in passages]
    paper = dict(title='Candidate parity',abstract='experiment evidence',figures_and_tables=[],
        full_text=[dict(section_name='Results',paragraphs=[f'experiment evidence {i} '+('word '*i) for i in range(40)])])
    _,doc,chunks = paper_records(paper,source_id='parity')
    store = PaperKnowledgeStore(tmp_path/'index.sqlite3')
    store.sync_documents([doc],chunks)
    # This fixture has no dense vectors. Both callers use exactly the same
    # real lexical store and deterministic scoring function.
    class Marker:
        name = 'fixture-embedding'
        def embed_query(self,query):
            return [1.0,0.0]
    marker = Marker()
    expected = ProbeAdapter(store,embedder=marker,reranker=Ranker(),rerank_stage='final',scoped_candidate_k=64)
    expected.search('experiment evidence',top_k=8,paper_sha256s={doc.sha256})
    adapter = PaperPilotKnowledgeSearchAdapter(store,embedder=marker,reranker=Ranker(),rerank_stage='final')
    actual = adapter.evidence_candidates('experiment evidence',paper_sha256s={doc.sha256})
    assert [h['stable_id'] for h in actual] == expected.candidate_ids[:24]
    assert len(actual) == 24 and adapter.scoped_candidate_k is None


def test_reader_pipeline_opt_in_uses_stage_and_cached_report(tmp_path):
    from test_reader_pipeline import factory, Search
    from test_durable_reader import AnswerProvider
    class SearchWithCandidates(Search):
        def evidence_candidates(self,question,*,paper_sha256s):
            return self.search(question,paper_sha256s=paper_sha256s)
    class SelectAndAnswer(AnswerProvider):
        def complete(self,messages,**kwargs):
            data = json.loads(messages[-1]['content'])
            if 'passages' in data:
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text=json.dumps({'ranking':[p['id'] for p in data['passages']]}),
                    prompt_tokens=10,completion_tokens=5,finish_reason='stop')
            return super().complete(messages,**kwargs)
    provider = SelectAndAnswer()
    rt = factory(tmp_path,provider)
    rt.search_adapter = SearchWithCandidates()
    rid = rt.enqueue(question='What fact?',metadata={'engine':'answer_first_text',
        'paper_sha256s':['sha'],'evidence_selection':'select-v2'})['run']['run_id']
    result = rt.resume(rid)
    assert result['run']['status'] == 'COMPLETED',result['run']
    assert result['run']['metadata']['reader_evidence_selection']['status'] == 'selected'
    assert provider.calls == 3 and result['run']['used_llm_calls'] == 3
    assert rt.resume(rid)['final_answer'] == result['final_answer']
    assert provider.calls == 3
