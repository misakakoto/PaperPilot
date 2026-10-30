import time
import json
import pytest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.web_api import create_fastapi_router
from test_durable_reader import AnswerProvider
from paperpilot_agentic.providers import LLMResponse


class Search:
    def search(self, query, top_k=8, *, paper_sha256s=None):
        return [{'stable_id': 'chunk', 'content': 'A source fact.',
                 'paper_sha256': 'sha', 'source_uri': 'paper://sha/chunk', 'page': 1}]


def factory(tmp_path, provider):
    return ResearchAgentRuntime(db_path=tmp_path/'runtime.sqlite3', workspace=tmp_path,
        search_adapter=Search(), llm=provider, text_reader_protocol='test-reader-v3')


@pytest.mark.parametrize('question', ['What fact?', '这是什么事实？'])
@pytest.mark.parametrize('preflight_stop', [True, False])
def test_recovery_stop_is_distinct_from_attempt_failure(tmp_path, monkeypatch, question, preflight_stop):
    from paperpilot_agentic import reader_pipeline as pipeline
    from paperpilot_agentic.models import Budget
    real_reader = pipeline.run_durable_reader

    def partial(*args, **kwargs):
        report = real_reader(*args, **kwargs)
        return {**report, 'answer_scope': 'PARTIAL'}

    attempts = []

    def failing_plan(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('synthetic planning failure')

    monkeypatch.setattr(pipeline, 'run_durable_reader', partial)
    monkeypatch.setattr(pipeline, 'plan_evidence_reload', failing_plan)
    class LocalizedProvider(AnswerProvider):
        def complete(self, messages, **kwargs):
            from dataclasses import replace
            response = super().complete(messages, **kwargs)
            if question != 'What fact?':
                response = replace(response, text=response.text.replace('A source fact.', '一条来源事实。'))
            return response

    provider = LocalizedProvider()
    runtime = factory(tmp_path, provider)
    queued = runtime.enqueue(question=question, budget=Budget(max_llm_calls=5 if preflight_stop else 6),
                             metadata={'engine': 'answer_first_text'})
    result = runtime.resume(queued['run']['run_id'])
    assert provider.calls == 2
    assert bool(attempts) is not preflight_stop, result['run']['metadata'].get('reader_report')
    meta = result['run']['metadata']
    assert meta['reader_recovery_status'] == ('NOT_STARTED_BUDGET' if preflight_stop else 'FAILED')
    assert meta['reader_recovery_failure_stage'] == ('recovery_preflight' if preflight_stop else 'recovery_decision')
    text = ' '.join(result['final_answer']['limitations'])
    assert '已核验' not in text and 'LLMBudgetExhausted' not in text
    if question == 'What fact?':
        assert 'not started' in text if preflight_stop else 'could not be completed' in text
        assert not any('\u4e00' <= char <= '\u9fff' for char in text)
    else:
        assert '尚未启动' in text if preflight_stop else '未完成' in text
    assert result['final_answer']['review_required'] is True
    assert result['final_answer']['answer'].startswith('A source fact.' if question == 'What fact?' else '一条来源事实。')
    restored = runtime.resume(queued['run']['run_id'])
    assert restored['final_answer'] == result['final_answer']
    assert provider.calls == 2


def test_formal_runtime_uses_reader_and_restores_owned_answer(tmp_path):
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    queued = runtime.enqueue(question='What fact?', metadata={'engine': 'answer_first_text'})
    rid = queued['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert result['final_answer']['answer'].startswith('A source fact.')
    assert provider.calls == 2
    assert result['run']['used_tool_calls'] == 1
    assert set(result['final_answer']['citation_ids']) == {e['evidence_id'] for e in result['evidence']}
    restored = factory(tmp_path, provider).resume(rid)
    assert restored['final_answer'] == result['final_answer']
    assert provider.calls == 2
    child = runtime.enqueue_follow_up(parent_run_id=rid, question='And then?')
    assert child['run']['metadata']['engine'] == 'answer_first_text'
    assert child['run']['metadata']['conversation_context_is_evidence'] is False


@pytest.mark.parametrize('explicit_engine', [False, True])
def test_web_post_text_engine_returns_202_and_get_restores_result(tmp_path, explicit_engine):
    provider = AnswerProvider()
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: factory(tmp_path, provider),
        default_engine='answer_first_text', default_visual_policy='auto'))
    with TestClient(app) as client:
        payload = {'question': 'What fact?'}
        if explicit_engine:
            payload['engine'] = 'answer_first_text'
        response = client.post('/agentic/runs', json=payload)
        assert response.status_code == 202
        rid = response.json()['run']['run_id']
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = client.get(f'/agentic/runs/{rid}').json()
            if result['run']['status'] in {'FAILED', 'COMPLETED'}:
                break
            time.sleep(.01)
        assert result['run']['status'] == 'COMPLETED'
        assert result['run']['metadata']['engine'] == 'answer_first_text'
        assert result['run']['metadata']['visual_policy'] == 'auto'
        assert result['final_answer']['answer'].startswith('A source fact.')
        assert provider.calls == 2


def test_web_app_default_matches_browser_and_explicit_legacy_is_preserved(tmp_path, monkeypatch):
    from paperpilot_agentic import web_app
    runtime = factory(tmp_path, AnswerProvider())
    monkeypatch.setattr(web_app, 'build_runtime', lambda settings: runtime)
    # Nothing is dispatched during this selection test; no provider or live DB.
    monkeypatch.setattr('paperpilot_agentic.web_api._BackgroundRunPool.submit', lambda *args: True)
    with TestClient(web_app.create_app(SimpleNamespace())) as client:
        response = client.post('/api/agentic/runs', json={'question': 'What fact?'})
        assert response.status_code == 202
        metadata = response.json()['run']['metadata']
        assert metadata['engine'] == 'answer_first_text'
        assert metadata['visual_policy'] == 'auto'
        response = client.post('/api/agentic/runs', json={
            'question': 'What fact?', 'visual_policy': 'off'})
        assert response.json()['run']['metadata']['visual_policy'] == 'off'
        response = client.post('/api/agentic/runs', json={
            'question': 'What fact?', 'engine': 'legacy'})
        assert response.status_code == 202
        assert response.json()['run']['metadata'].get('engine', 'legacy') == 'legacy'


def test_default_reader_without_configuration_fails_before_creating_run(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path/'runtime.sqlite3', workspace=tmp_path)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime,
        default_engine='answer_first_text'))
    with TestClient(app) as client:
        response = client.post('/agentic/runs', json={'question': 'What fact?'})
        assert response.status_code == 400
        assert 'not configured' in response.json()['detail']
        assert runtime.list_runs() == []


@pytest.mark.parametrize('options', [
    {'default_engine': 'unknown'}, {'default_visual_policy': 'unknown'}])
def test_invalid_router_defaults_fail_at_construction(options):
    with pytest.raises(ValueError):
        create_fastapi_router(**options)


def test_tool_budget_stops_before_retrieval_or_provider(tmp_path):
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    rid = runtime.enqueue(question='What fact?', budget={'max_tool_calls': 0},
        metadata={'engine': 'answer_first_text'})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'BUDGET_EXHAUSTED'
    assert provider.calls == 0


def test_outline_navigation_is_used_by_formal_runtime(tmp_path):
    class Navigator(AnswerProvider):
        def complete(self, messages, **kwargs):
            if 'outline' in json.loads(messages[-1]['content']):
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text='{"section_ids":["section_0"],"reason":"Read definition"}',
                                   finish_reason='stop')
            return super().complete(messages, **kwargs)

    class Store:
        def paper_outline(self, sha):
            return [{'section_id': 'section_0', 'title': 'Definition', 'chars': 30, 'chunk_ids': ['chunk2']}]

        def get_chunks(self, ids):
            return [SimpleNamespace(paper_sha256='sha', title='Paper', section='Definition',
                text='A definition.', stable_id='chunk2', page_start=2)]

    provider = Navigator()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter.store = Store()
    rid = runtime.enqueue(question='What fact?', metadata={'engine': 'answer_first_text'})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert provider.calls == 3
    # Includes the source inventory and the new host-only object locator;
    # the locator must be accounted without adding a model request.
    assert result['run']['used_tool_calls'] == 5
    assert len(result['evidence']) == 2


def test_truncated_router_still_reads_bounded_local_definitions(tmp_path):
    class Truncated(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'outline' in payload:
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text='{"section_ids":', finish_reason='length')
            return super().complete(messages, **kwargs)

    class Store:
        def paper_outline(self, sha):
            return [{'section_id': 's', 'title': 'Method', 'chars': 60, 'chunk_ids': ['definition']}]
        def get_chunks(self, ids):
            return [SimpleNamespace(paper_sha256='sha', title='Paper', section='Method',
                text='A source fact. Grouping alternates the two input features.',
                stable_id='definition', page_start=2)]

    provider = Truncated()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter.store = Store()
    rid = runtime.enqueue(question='Explain grouping mechanism', metadata={
        'engine': 'answer_first_text'})['run']['run_id']
    result = runtime.resume(rid)
    routing = result['run']['metadata']['reader_routing_by_paper']['sha']
    assert routing['status'] == 'FAILED'
    assert routing['fallback']['provider_attempts'] == 0
    assert any('Grouping alternates' in row['content'] for row in result['evidence'])
    assert provider.calls == 3


def test_explicit_source_violation_fails_before_provider(tmp_path):
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    rid = runtime.enqueue(question='What fact?', metadata={
        'engine': 'answer_first_text', 'paper_sha256s': ['different-source']})['run']['run_id']
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0


def test_visual_targets_require_opt_in_before_any_model_request(tmp_path):
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    rid = runtime.enqueue(question='Read table', metadata={
        'engine': 'answer_first_text', 'reader_visual_targets': [
            {'paper_sha256': 'sha', 'label': 'Table XII'}]})['run']['run_id']
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0


def test_explicit_visual_transcription_reaches_answer_reader(tmp_path, monkeypatch):
    import paperpilot_agentic.reader_visual as visual_module
    seen = []
    class VisualAwareProvider(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'evidence' in payload:
                assert 'Vision-model transcription' in payload['evidence'][0]['content']
                seen.append(True)
            return super().complete(messages, **kwargs)
    def collect(*args):
        return [{'evidence_id': 'visual', 'evidence_type': 'TABLE',
                 'content': '[Vision-model transcription] A source fact.',
                 'source_uri': 'paper://sha/layout/table', 'page': 2,
                 'bbox': [1, 2, 3, 4], 'object_type': 'table'}]
    monkeypatch.setattr(visual_module, 'collect_visual_evidence', collect)
    runtime = factory(tmp_path, VisualAwareProvider())
    rid = runtime.enqueue(question='Table?', metadata={'engine': 'answer_first_text',
        'allow_visual': True, 'paper_sha256s': ['sha'],
        'reader_visual_targets': [{'paper_sha256': 'sha', 'label': 'Table XII'}]})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert seen
    assert any(e['evidence_type'] == 'TABLE' for e in result['evidence'])


def test_unavailable_target_table_does_not_fall_back_to_unverified_text_answer(tmp_path, monkeypatch):
    import paperpilot_agentic.reader_visual as visual_module
    monkeypatch.setattr(visual_module, 'collect_visual_evidence', lambda *args: [])
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    rid = runtime.enqueue(question='Table values?', metadata={'engine': 'answer_first_text',
        'allow_visual': True, 'paper_sha256s': ['sha'],
        'reader_visual_targets': [{'paper_sha256': 'sha', 'label': 'Table XII'}]})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['research_outcome'] == 'UNRESOLVED'
    assert result['final_answer']['answer'] == ''
    assert provider.calls == 0


def test_invalid_source_scope_finishes_failed_instead_of_leaving_pending(tmp_path):
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    rid = runtime.enqueue(question='What fact?', metadata={
        'engine': 'answer_first_text', 'paper_sha256s': 'not-a-list'})['run']['run_id']
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0


def test_cancel_during_provider_does_not_publish_or_call_auditor(tmp_path):
    class CancelProvider(AnswerProvider):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            runtime.cancel(rid)
            return response

    provider = CancelProvider()
    runtime = factory(tmp_path, provider)
    rid = runtime.enqueue(question='What fact?', metadata={'engine': 'answer_first_text'})['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'CANCELLED'
    assert result['final_answer'] is None
    assert provider.calls == 1


@pytest.mark.parametrize('continuing,reset', [(True, False), (False, False), (True, True)])
def test_answer_requirements_are_scoped_to_continued_task(tmp_path, monkeypatch, continuing, reset):
    captured = []
    class Conversational(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'current_question' in payload:
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text=json.dumps({'question': 'What fact?',
                    'use_prior_sources': continuing, 'needs_clarification': False,
                    'reason': 'Task selection'}), finish_reason='stop')
            captured.append(payload)
            return super().complete(messages, **kwargs)
    runtime = factory(tmp_path, Conversational())
    parent = runtime.enqueue(question='What fact? Always explain uncertainty.',
        metadata={'engine': 'answer_first_text', 'reader_memory_version': 'request-ledger-v1'})['run']['run_id']
    runtime.resume(parent)
    # A matching arithmetic helper must not bypass the requirements-aware
    # Writer/Auditor when the follow-up is rewritten, even after a task reset.
    import paperpilot_agentic.numeric_scope as numeric
    monkeypatch.setattr(numeric, 'deterministic_numeric_attribution',
        lambda question, evidence: {'answer': 'Arithmetic-only shortcut.'})
    child = runtime.enqueue_follow_up(parent_run_id=parent,
        question='reset constraints; What fact?' if reset else 'What supports it?')['run']['run_id']
    captured.clear()
    result = runtime.resume(child)
    assert result['run']['status'] == 'COMPLETED'
    assert result['final_answer']['answer'].startswith('A source fact.')
    dispatch = result['run']['metadata']['reader_calculation_dispatch']
    assert dispatch['mode'] == 'REQUIREMENTS_AWARE_READER'
    assert dispatch['question_rewritten'] is True
    assert result['run']['metadata']['reader_deterministic_calculation']['answer'] == 'Arithmetic-only shortcut.'
    history = result['run']['metadata']['answer_request_history']
    if continuing and not reset:
        assert history[0]['user_request'] == 'What fact? Always explain uncertainty.'
        assert len(captured) >= 2
        assert all(payload.get('request_history') == history for payload in captured)
    else:
        assert history == []
        assert all('request_history' not in payload for payload in captured)


def test_unchanged_followup_still_honors_history_instead_of_arithmetic_template(tmp_path, monkeypatch):
    captured = []
    class Conversational(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'current_question' in payload:
                kwargs['before_attempt']()
                self.calls += 1
                return LLMResponse(text=json.dumps({'question': payload['current_question'],
                    'use_prior_sources': True, 'needs_clarification': False,
                    'reason': 'Same subject'}), finish_reason='stop')
            captured.append(payload)
            return super().complete(messages, **kwargs)
    provider = Conversational()
    runtime = factory(tmp_path, provider)
    parent = runtime.enqueue(question='What fact? Always explain uncertainty.',
        metadata={'engine': 'answer_first_text', 'reader_memory_version': 'request-ledger-v1'})['run']['run_id']
    runtime.resume(parent)
    import paperpilot_agentic.numeric_scope as numeric
    monkeypatch.setattr(numeric, 'deterministic_numeric_attribution',
        lambda question, evidence: {'answer': 'Arithmetic-only shortcut.'})
    child = runtime.enqueue_follow_up(parent_run_id=parent, question='What fact?')['run']['run_id']
    captured.clear()
    result = runtime.resume(child)
    assert result['run']['status'] == 'COMPLETED'
    assert result['run']['metadata']['reader_calculation_dispatch']['question_rewritten'] is False
    assert len(captured) == 2
    assert all(p['request_history'][0]['user_request'].endswith('Always explain uncertainty.') for p in captured)
    assert all(p['original_question'] == 'What fact?' for p in captured)
    assert result['final_answer']['answer'].startswith('A source fact.')
    calls = provider.calls
    assert runtime.resume(child)['final_answer'] == result['final_answer']
    assert provider.calls == calls


@pytest.mark.parametrize('suffix', [
    ' 请用一点回答。',
    ' 请同时解释这个结果的适用边界，不要推断部署表现。',
    ' Answer in English and distinguish measurement from inference.',
])
def test_first_turn_real_numeric_match_uses_full_answer_contract(tmp_path, suffix):
    source = ('TABLE IV\nPERFORMANCES ON THE BRIGHT TEST SET.\n'
              'ChangeMamba attains an overall accuracy of 96.22% and a mIoU of 67.63%. '
              'The proposed FlowMamba further improves these results to 96.68% and 68.41%, respectively.')
    captured = []

    class SourceSearch(Search):
        def search(self, query, top_k=8, *, paper_sha256s=None):
            return [{**super().search(query)[0], 'content': source}]

    class RecordingProvider(AnswerProvider):
        def complete(self, messages, **kwargs):
            from dataclasses import replace
            captured.append(json.loads(messages[-1]['content']))
            response = super().complete(messages, **kwargs)
            return replace(response, text=response.text.replace('A source fact.', '这段材料报告了 BRIGHT 测试集结果。'))

    provider = RecordingProvider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter = SourceSearch()
    question = '96.68% 对应哪个数据集、哪项指标、哪张表？' + suffix
    rid = runtime.enqueue(question=question,
        metadata={'engine': 'answer_first_text'})['run']['run_id']
    result = runtime.resume(rid)
    meta = result['run']['metadata']
    assert meta['reader_deterministic_calculation']['dataset'] == 'BRIGHT'
    assert meta['reader_calculation_dispatch']['has_request_history'] is False
    assert meta['reader_calculation_dispatch']['question_rewritten'] is False
    assert provider.calls == 2
    assert all(p['original_question'] == question for p in captured)
    # This scripted provider verifies dispatch and preservation, not factual
    # correctness or whether the entire requested answer was generated.
    assert result['final_answer']['answer'] != meta['reader_deterministic_calculation']['answer']
    assert all(e['content'] == source for e in result['evidence'])
    assert runtime.resume(rid)['final_answer'] == result['final_answer']
    assert provider.calls == 2


def test_follow_up_resolves_question_but_retrieves_fresh_evidence(tmp_path):
    class Conversational(AnswerProvider):
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            if 'current_question' in payload:
                kwargs['before_attempt']()
                self.calls += 1
                assert payload['history']['prior_question'] == 'What fact?'
                return LLMResponse(text=json.dumps({'question': 'What supports the source fact?',
                    'use_prior_sources': True, 'needs_clarification': False, 'reason': 'Same topic'}),
                    finish_reason='stop')
            return super().complete(messages, **kwargs)

    class RecordingSearch(Search):
        queries = []
        def search(self, query, top_k=8, *, paper_sha256s=None):
            self.queries.append((query, paper_sha256s))
            return super().search(query, top_k, paper_sha256s=paper_sha256s)

    provider = Conversational()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter = RecordingSearch()
    parent = runtime.enqueue(question='What fact?', metadata={'engine': 'answer_first_text'})['run']['run_id']
    first = runtime.resume(parent)
    child = runtime.enqueue_follow_up(parent_run_id=parent, question='What supports it?')['run']['run_id']
    result = runtime.resume(child)
    assert result['run']['status'] == 'COMPLETED'
    assert result['run']['question'] == 'What supports it?'
    assert result['run']['metadata']['resolved_question'] == 'What supports the source fact?'
    assert runtime.search_adapter.queries[-1] == ('What supports the source fact?', {'sha'})
    assert provider.calls == 5  # Parent writer/audit; child resolution/writer/audit.
    assert {e['evidence_id'] for e in first['evidence']}.isdisjoint(
        {e['evidence_id'] for e in result['evidence']})
    assert result['run']['metadata']['conversation_id'] == first['run']['metadata']['conversation_id']


def test_ambiguous_follow_up_requests_clarification_without_search(tmp_path):
    class Ambiguous(AnswerProvider):
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(text=json.dumps({'question': 'Which one?', 'use_prior_sources': False,
                'needs_clarification': True, 'reason': 'Please specify which method.'}), finish_reason='stop')
    runtime = factory(tmp_path, AnswerProvider())
    parent = runtime.enqueue(question='What fact?', metadata={'engine': 'answer_first_text'})['run']['run_id']
    runtime.resume(parent)
    provider = Ambiguous()
    resumed = factory(tmp_path, provider)
    child = resumed.enqueue_follow_up(parent_run_id=parent, question='Which one?')['run']['run_id']
    result = resumed.resume(child)
    assert result['run']['research_outcome'] == 'UNRESOLVED'
    assert result['final_answer']['limitations'] == ['Please specify which method.']
    assert result['run']['used_tool_calls'] == 0
    assert result['evidence'] == []
    assert provider.calls == 1


@pytest.mark.parametrize('use_prior,expected', [(True, {'a', 'b'}), (False, {'new'})])
def test_followup_effective_scope_persists_after_title_resolution(tmp_path, use_prior, expected):
    from paperpilot_agentic.models import RunStatus
    class Resolver(AnswerProvider):
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(text=json.dumps({'question': 'Resolved topic',
                'use_prior_sources': use_prior, 'needs_clarification': False,
                'reason': 'Same papers' if use_prior else 'New subject'}), finish_reason='stop')
    class InspectSearch:
        seen = None
        def resolve_explicit_title_scope(self, queries, paper_sha256s=None):
            return {'new'} if 'Resolved topic' in queries else set()
        def search(self, query, top_k=8, *, paper_sha256s=None):
            self.seen = paper_sha256s
            raise RuntimeError('Deliberate stop after scope inspection, not an answer test')
    runtime = factory(tmp_path, Resolver())
    runtime.search_adapter = InspectSearch()
    parent = runtime.enqueue(question='Two papers', metadata={
        'engine': 'answer_first_text', 'paper_sha256s': ['a', 'b']})['run']['run_id']
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    child = runtime.enqueue_follow_up(parent_run_id=parent, question='Next?')['run']['run_id']
    result = runtime.resume(child)
    assert runtime.search_adapter.seen == expected
    assert result['run']['metadata']['resolved_paper_sha256s'] == sorted(expected)
    assert result['run']['status'] == 'FAILED'  # The deliberately stopped retrieval was not success.


@pytest.mark.parametrize('query,prior,cap,expected', [
    ('Exclude Beta; continue the analysis.', ['a','b'], None, {'a'}),
    ('Only analyze Beta.', ['a','b'], None, {'b'}),
    ('Also include Beta.', ['a'], None, {'a','b'}),
    ('Also include Beta.', ['a'], ['a'], None),
])
def test_original_source_edits_survive_lossy_rewrite(tmp_path, query, prior, cap, expected):
    from paperpilot_agentic.models import RunStatus
    class Resolver(AnswerProvider):
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(text=json.dumps({'question':'Continue analysis.',
                'use_prior_sources':True, 'needs_clarification':False,
                'reason':'Synthetic rewrite deliberately omits source edit'}), finish_reason='stop')
    class ScopeSearch:
        seen = None
        def resolve_explicit_title_scope(self, queries, paper_sha256s=None):
            text = ' '.join(queries)
            return {sha for name,sha in [('Alpha','a'),('Beta','b')] if name in text}
        def search(self, query, top_k=8, *, paper_sha256s=None):
            self.seen = paper_sha256s
            raise RuntimeError('Stop after actual retrieval scope capture; not answer validation')
    runtime = factory(tmp_path, Resolver())
    runtime.search_adapter = ScopeSearch()
    parent = runtime.enqueue(question='Analyze Alpha and Beta', metadata={
        'engine':'answer_first_text', 'paper_sha256s':prior})['run']['run_id']
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    child = runtime.enqueue_follow_up(parent_run_id=parent, question=query,
        metadata={'paper_sha256s':cap} if cap else {})['run']['run_id']
    result = runtime.resume(child)
    if expected is None:
        assert runtime.search_adapter.seen is None
        assert result['run']['metadata']['requested_sources_outside_selection'] == ['b']
        assert result['final_answer']['answer'] == ''
        assert result['run']['research_outcome'] == 'UNRESOLVED'
        return
    assert result['run']['status'] == 'FAILED'
    assert runtime.search_adapter.seen == expected
    assert result['run']['metadata']['resolved_paper_sha256s'] == sorted(expected)
    # Continuation must not silently restore the prior wider set.
    follow = runtime.enqueue_follow_up(parent_run_id=child, question='Continue.')['run']['run_id']
    runtime.resume(follow)
    assert runtime.search_adapter.seen == expected
