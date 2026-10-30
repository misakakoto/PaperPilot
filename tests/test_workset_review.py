"""Read-only scheduler integration; scripted models do not measure answer quality."""
import pytest
from dataclasses import replace

from paperpilot_agentic.models import Budget, RunStatus
from paperpilot_agentic.request_state_store import RequestStateStore
from paperpilot_agentic.workset_review import enqueue_review, resume_review
from test_durable_reader import AnswerProvider
from test_reader_pipeline import factory


def prepared(tmp_path, count=25, max_calls=30):
    class LocalizedProvider(AnswerProvider):
        def complete(self, messages, **kwargs):
            response = super().complete(messages, **kwargs)
            return replace(response, text=response.text.replace('A source fact.', '一条来源事实。'))
    provider = LocalizedProvider()
    runtime = factory(tmp_path, provider)
    text = 'Review source facts. Do not train. ' + ' '.join(f'Activity{i}' for i in range(count))
    parent = runtime.enqueue(question=text, metadata={'engine': 'answer_first_text'})['run']['run_id']
    runtime.db.update_run_metadata(parent, {'request_state_root': parent, 'resolved_paper_sha256s': ['sha']})
    store = RequestStateStore(runtime.db)
    ops = [
        {'kind': 'create_task', 'task_id': 'A', 'label': 'Source facts', 'quote': 'Review source facts'},
        {'kind': 'activate_task', 'task_id': 'A', 'quote': 'Review source facts'},
        {'kind': 'set_condition', 'task_id': 'A', 'key': 'no_train', 'text': 'Do not train',
         'quote': 'Do not train', 'applies_to': None},
    ]
    for i in range(count):
        ops.extend([
            {'kind': 'add_todo', 'task_id': 'A', 'key': f'a{i}', 'text': f'Activity{i}', 'quote': f'Activity{i}'},
            {'kind': 'set_condition', 'task_id': 'A', 'key': f'c{i}', 'text': 'Source fact ' * 35,
             'quote': 'source facts', 'applies_to': f'a{i}'},
        ])
    state = store.commit(parent, source_run_id=parent,
                         delta={'expected_revision': 'initial', 'operations': ops})
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    result = enqueue_review(runtime, parent, synthesize=False, budget=Budget(max_llm_calls=max_calls,
        max_tokens=1000000, synthesis_reserve_calls=0, repair_reserve_calls=0))
    return runtime, provider, result['run']['run_id'], parent, state


def test_pages_execute_once_reopen_resume_same_budget_and_leave_todos_pending(tmp_path):
    runtime, provider, rid, root, state = prepared(tmp_path)
    count = len(runtime.db.get_run(rid)['metadata']['workset_manifest']['pages'])
    assert count > 1
    first = resume_review(runtime, rid, max_batches=1)
    assert first['run']['status'] == 'RUNNING', first['run'].get('error')
    assert first['run']['metadata']['workset_progress']['reviewed_pages'] == 1
    calls = provider.calls
    assert calls == 2
    reopened = factory(tmp_path, provider)
    result = reopened.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    assert provider.calls == 2 * count
    assert result['run']['used_llm_calls'] == provider.calls
    assert result['run']['research_outcome'] != 'ANSWERED'
    assert result['run']['metadata']['workset_progress']['activities_executed'] == 0
    reopened.resume(rid)
    assert provider.calls == 2 * count
    assert RequestStateStore(runtime.db).load(root, requester_id=root) == state


def test_budget_not_multiplied_by_page_count_and_partial_results_survive(tmp_path):
    runtime, provider, rid, _, _ = prepared(tmp_path, max_calls=2)
    result = runtime.resume(rid)
    assert result['run']['status'] == 'BUDGET_EXHAUSTED'
    assert provider.calls == 2
    assert result['run']['metadata']['workset_progress']['reviewed_pages'] == 1
    assert result['final_answer']['answer']
    assert '1/' in result['final_answer']['limitations'][0]


def test_unknown_started_page_is_not_retried(tmp_path):
    runtime, provider, rid, _, _ = prepared(tmp_path)
    class Crash(BaseException):
        pass
    def crash(*args, **kwargs):
        raise Crash()
    runtime.search_adapter.search = crash
    with pytest.raises(Crash):
        resume_review(runtime, rid)
    reopened = factory(tmp_path, provider)
    result = reopened.resume(rid)
    assert result['run']['status'] == 'RUNNING'
    assert result['run']['used_tool_calls'] == 1
    assert provider.calls == 0


def test_cancel_between_pages_prevents_more_model_calls(tmp_path):
    runtime, provider, rid, _, _ = prepared(tmp_path)
    resume_review(runtime, rid, max_batches=1)
    runtime.db.update_run_status(rid, RunStatus.CANCELLED)
    assert runtime.resume(rid)['run']['status'] == 'CANCELLED'
    assert provider.calls == 2


def test_new_requirements_invalidate_old_review_instead_of_silently_replanning(tmp_path):
    runtime, provider, rid, root, state = prepared(tmp_path)
    resume_review(runtime, rid, max_batches=1)
    child = runtime.enqueue_follow_up(parent_run_id=root, question='New constraint')['run']['run_id']
    RequestStateStore(runtime.db).commit(root, source_run_id=child, delta={
        'expected_revision': state['revision'], 'operations': []})
    result = runtime.resume(rid)
    assert result['run']['status'] == 'FAILED'
    assert provider.calls == 2


def test_explicit_empty_paper_scope_cannot_become_global_search(tmp_path):
    runtime, _, _, root, _ = prepared(tmp_path)
    runtime.db.update_run_metadata(root, {'source_scope_explicitly_empty': True})
    with pytest.raises(ValueError, match='empty source scope'):
        enqueue_review(runtime, root)


def test_step_budget_and_no_cross_scope_evidence(tmp_path):
    runtime, provider, rid, _, _ = prepared(tmp_path)
    with runtime.db.transaction() as conn:
        import json
        limits = runtime.db.get_run(rid)['budget']
        limits['max_steps'] = 1
        conn.execute('UPDATE runs SET budget_json=? WHERE run_id=?', (json.dumps(limits), rid))
    assert runtime.resume(rid)['run']['status'] == 'BUDGET_EXHAUSTED'
    assert provider.calls == 2


def test_out_of_scope_result_fails_before_model_call(tmp_path):
    runtime, provider, rid, _, _ = prepared(tmp_path)
    runtime.search_adapter.search = lambda *a, **k: [
        {'stable_id': 'other', 'paper_sha256': 'wrong', 'content': 'Wrong source'}]
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0


def test_protocol_change_fails_without_leaving_pending_run_or_spending(tmp_path):
    runtime, provider, rid, _, _ = prepared(tmp_path)
    runtime.text_reader_protocol = 'different-model-protocol'
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert provider.calls == 0


def test_http_endpoint_uses_saved_plan_not_client_supplied_pages(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    from unittest.mock import patch
    runtime, _, _, root, _ = prepared(tmp_path)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with patch('paperpilot_agentic.web_api._BackgroundRunPool.submit', return_value=True), TestClient(app) as client:
        bad = client.post(f'/agentic/runs/{root}/workset-review', json={'pages': []})
        assert bad.status_code == 422
        response = client.post(f'/agentic/runs/{root}/workset-review', json={'budget': {'max_llm_calls': 10}})
        assert response.status_code == 202, response.text
        assert response.json()['run']['metadata']['engine'] == 'workset_review'


def test_followup_stays_read_only_and_pinned_to_review_sources(tmp_path):
    runtime, _, rid, _, _ = prepared(tmp_path)
    runtime.resume(rid)
    follow = runtime.enqueue_follow_up(parent_run_id=rid, question='继续比较')
    assert follow['run']['metadata']['engine'] == 'answer_first_text'
    assert follow['run']['metadata']['paper_sha256s'] == ['sha']


@pytest.mark.parametrize('verdict', ['ACCEPT', 'REJECT'])
def test_partitionable_state_still_requires_full_audit_before_commit(tmp_path, verdict):
    from paperpilot_agentic.request_state_compiler import compile_request_batch
    from test_request_compiler_view import Model
    _, _, _, _, state = prepared(tmp_path)
    model = Model([
        {'base_revision': state['revision'], 'frames': [{'source_id': 'next', 'operations': []}]},
        {'verdict': verdict, 'issues': [] if verdict == 'ACCEPT' else ['Wrong interpretation'],
         'missing_requirements': []},
    ])
    result = compile_request_batch(state, [{'id': 'next', 'text': 'Continue'}], model,
                                   allow_work_batches=True)
    assert len(model.messages) == 2
    if verdict == 'ACCEPT':
        assert result['status'] == 'MODEL_ACCEPTED_UNVERIFIED'
        assert result['work_view']['view'] is None  # Never a falsely complete first page.
        assert result['work_view']['work_plan']['status'] == 'NEEDS_WORK_BATCHES'
    else:
        assert result['status'] == 'REJECTED_OR_UNCERTAIN'
        assert 'candidate_state' not in result


def test_concurrent_resumers_cannot_run_same_page_twice(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    runtime, provider, rid, _, _ = prepared(tmp_path)
    entered, release = Event(), Event()
    search = runtime.search_adapter.search
    def slow_search(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return search(*args, **kwargs)
    runtime.search_adapter.search = slow_search
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(resume_review, runtime, rid, max_batches=1)
        assert entered.wait(10)
        try:
            other = factory(tmp_path, provider)
            result = resume_review(other, rid, max_batches=1)
            assert result['run']['used_tool_calls'] == 1
            assert provider.calls == 0
        finally:
            release.set()
        assert running.result(timeout=20)['run']['status'] == 'RUNNING'
    assert provider.calls == 2


def test_normal_followup_commits_audited_large_state_and_offers_review(tmp_path):
    import json
    from paperpilot_agentic.providers import LLMResponse
    runtime, provider, _, root, _ = prepared(tmp_path)
    original = provider.complete
    def complete(messages, *, before_attempt, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if 'state' in payload and 'sources' in payload:
            value = {'base_revision': payload['state']['revision'],
                     'frames': [{'source_id': s['id'], 'operations': []} for s in payload['sources']]}
        elif 'before' in payload and 'proposal' in payload:
            value = {'verdict': 'ACCEPT', 'issues': [], 'missing_requirements': []}
        else:
            return original(messages, before_attempt=before_attempt, **kwargs)
        before_attempt()
        provider.calls += 1
        return LLMResponse(text=json.dumps(value), prompt_tokens=10, completion_tokens=5, finish_reason='stop')
    provider.complete = complete
    child = runtime.enqueue_follow_up(parent_run_id=root, question='继续审阅活动')['run']['run_id']
    result = runtime.resume(child)
    assert result['run']['metadata']['request_memory_status'] == 'NEEDS_WORK_BATCHES'
    assert result['run']['metadata']['workset_review_available'] is True
    assert '分批审阅研究活动' in result['final_answer']['limitations'][0]
    assert provider.calls == 2
    review = enqueue_review(runtime, child)['run']['run_id']
    final = runtime.resume(review)
    assert final['run']['status'] == 'COMPLETED', final['run'].get('error')
    assert final['run']['research_outcome'] == 'PARTIAL'
