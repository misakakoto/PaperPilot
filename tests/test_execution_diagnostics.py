from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperpilot_agentic.execution_diagnostics import execution_fault
from paperpilot_agentic.models import RunStatus
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.web_api import create_fastapi_router
from paperpilot_agentic.workbench import ResearchWorkbench


@pytest.mark.parametrize('status,error,provider,snapshot,code', [
    ('FAILED', 'PermissionDeniedError', {'provider_code': 'AllocationQuota.FreeTierOnly'}, {}, 'MODEL_QUOTA_EXHAUSTED'),
    ('BUDGET_EXHAUSTED', 'LLMBudgetExhausted', {}, {}, 'LOCAL_BUDGET_ADMISSION'),
    ('FAILED', 'ContextWindowExceeded', {}, {}, 'CONTEXT_TOO_LARGE'),
    ('FAILED', 'APIConnectionError', {}, {'unknown_results': 1}, 'UNKNOWN_RESULT'),
    ('FAILED', 'PermissionDeniedError', {'provider_code': 'access_denied'}, {}, None),
    ('COMPLETED', '', {'provider_code': 'AllocationQuota.FreeTierOnly'}, {}, None),
])
def test_provider_budget_window_and_unknown_are_distinct(status, error, provider, snapshot, code):
    run = {'status': status, 'error': error, 'metadata': {'last_provider_failure': provider}}
    result = execution_fault(run, snapshot)
    assert (result or {}).get('code') == code


def test_api_profile_defaults_and_explicit_allowances(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path / 'runtime.sqlite3', workspace=tmp_path,
                                   text_reader_protocol='offline-fixture')
    runtime.resume = lambda rid: runtime.status(rid)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime, default_engine='answer_first_text'))
    with TestClient(app) as client:
        daily = client.post('/agentic/runs', json={'question': 'Read a mechanism.'}).json()['run']
        assert daily['budget']['max_tokens'] == 200000
        frozen = client.post('/agentic/runs', json={'question': 'Frozen question.',
            'execution_profile': 'frozen_acceptance', 'budget': {'max_tokens': 12345}}).json()['run']
        assert frozen['metadata']['execution_profile'] == 'frozen_acceptance'
        assert frozen['budget']['max_tokens'] == 12345
        assert client.post('/agentic/runs', json={'question': 'Q', 'execution_profile': 'invalid'}).status_code == 400
        runtime.db.update_run_status(frozen['run_id'], RunStatus.COMPLETED)
        child = client.post('/agentic/runs/' + frozen['run_id'] + '/follow-up',
                            json={'question': 'Follow-up.'}).json()['run']
        assert child['metadata']['execution_profile'] == 'frozen_acceptance'
        assert child['budget']['max_tokens'] == 100000


def test_actual_workbench_exposes_fault_without_publishing_a_draft(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path / 'runtime.sqlite3', workspace=tmp_path)
    run = runtime.enqueue(question='Q', metadata={'engine': 'answer_first_text'})['run']
    runtime.db.update_run_status(run['run_id'], RunStatus.FAILED, error='ModelQuotaExhausted')
    workbench = ResearchWorkbench(runtime, SimpleNamespace(active=lambda _: False))
    turn = workbench.turn(run['run_id'])
    assert turn['failure']['code'] == 'MODEL_QUOTA_EXHAUSTED'
    assert turn['message']['answer'] == ''
    assert turn['budget']['max_tokens'] == 200000


def test_workbench_explicit_allowance_and_duplicate_submission_are_preserved(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path / 'runtime.sqlite3', workspace=tmp_path,
                                   text_reader_protocol='offline-fixture')
    workbench = ResearchWorkbench(runtime, SimpleNamespace(active=lambda _: False, submit=lambda *args: True))
    request = {'request_id': 'explicit-allowance', 'action': 'answer',
               'question': 'Read a mechanism.', 'budget': {'max_tokens': 12345, 'max_llm_calls': 4}}
    first = workbench.submit(request)
    repeated = workbench.submit(request)
    assert repeated['run_id'] == first['run_id'] and repeated['reused']
    run = runtime.db.get_run(first['run_id'])
    assert run['budget']['max_tokens'] == 12345 and run['budget']['max_llm_calls'] == 4
    assert run['used_llm_calls'] == 0


def test_workbench_provider_404_explains_model_access_without_publishing_answer(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path / 'runtime.sqlite3', workspace=tmp_path)
    run = runtime.enqueue(question='Q', metadata={'engine': 'research_session'})['run']
    runtime.db.update_run_metadata(run['run_id'], {'last_provider_failure': {
        'error_type': 'NotFoundError', 'http_status': 404,
        'provider_code': 'model_not_found', 'category': 'MODEL_OR_ENDPOINT_UNAVAILABLE',
    }})
    runtime.db.update_run_status(run['run_id'], RunStatus.FAILED, error='NotFoundError')
    workbench = ResearchWorkbench(runtime, SimpleNamespace(active=lambda _: False))
    turn = workbench.turn(run['run_id'])
    assert turn['failure']['code'] == 'MODEL_OR_ENDPOINT_UNAVAILABLE'
    assert '404' in turn['failure']['title']
    assert '业务空间' in turn['failure']['action']
    assert turn['message']['answer'] == ''
    assert runtime.db.get_run(run['run_id'])['used_llm_calls'] == 0


def test_local_not_found_name_alone_is_not_a_provider_404():
    run = {'status': 'FAILED', 'error': 'NotFoundError', 'metadata': {}}
    assert execution_fault(run, {}) is None


def test_provider_404_cannot_override_unknown_result():
    run = {'status': 'FAILED', 'metadata': {'last_provider_failure': {
        'category': 'MODEL_OR_ENDPOINT_UNAVAILABLE', 'http_status': 404,
    }}}
    assert execution_fault(run, {'unknown_results': 1})['code'] == 'UNKNOWN_RESULT'
