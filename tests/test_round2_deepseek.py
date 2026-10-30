import json
import gzip

import httpx
import pytest

from evals.qasper.round2_deepseek import CappedTransport, run_summary


def test_summary_does_not_count_failed_or_unrun_cases_as_complete():
    rows = [{'status': 'COMPLETED', 'error_type': None},
            {'status': 'FAILED', 'error_type': 'ReadTimeout'},
            {'status': 'NOT_RUN_PROVIDER_STOP_OR_BUDGET'}]
    report = run_summary(rows, 4, 3)
    assert report['complete_case_count'] == 1
    assert report['recorded_case_count'] == report['planned_case_count'] == 3
    assert report['results'] == rows
    assert report['actual_requests'] == 4
    assert report['factual_review'] == 'PENDING'


@pytest.mark.parametrize('rows', [[], [{'status': 'COMPLETED', 'error_type': 'RuntimeError'}]])
def test_interrupted_summary_keeps_full_planned_denominator(rows):
    report = run_summary(rows, 1, 3)
    assert report['complete_case_count'] == 0
    assert report['planned_case_count'] == 3
    assert report['recorded_case_count'] == len(rows)
    assert report['actual_requests'] == 1


def test_compressed_response_is_decoded_once_and_saved_without_headers(tmp_path):
    payload = b'{"id":"compressed-fixture"}'
    def respond(request):
        return httpx.Response(200, content=gzip.compress(payload),
                              headers={'content-encoding': 'gzip'})
    transport = CappedTransport(tmp_path, httpx.MockTransport(respond))
    with httpx.Client(transport=transport) as client:
        response = client.post('https://api.deepseek.com/v1/chat/completions', json={})
    assert response.json() == {'id': 'compressed-fixture'}
    assert (tmp_path/'response-001.bin').read_bytes() == payload
    assert transport.count == 1


def test_cap_actual_dispatches_without_saving_key(tmp_path):
    sent = []
    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={'id': 'fixture'})
    transport = CappedTransport(tmp_path, httpx.MockTransport(respond))
    with httpx.Client(transport=transport) as client:
        for _ in range(20):
            client.post('https://api.deepseek.com/v1/chat/completions', json={'messages': []},
                        headers={'Authorization': 'secret-fixture'}).raise_for_status()
        with pytest.raises(httpx.RequestError):
            client.post('https://api.deepseek.com/v1/chat/completions', json={})
    assert len(sent) == transport.count == 20
    assert not any('secret-fixture' in p.read_text() for p in tmp_path.iterdir())


def test_other_provider_blocked_and_auth_error_stops_before_second_dispatch(tmp_path):
    transport = CappedTransport(tmp_path, httpx.MockTransport(lambda request: httpx.Response(403, text='secret-fixture')))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.RequestError):
            client.post('https://other.example/v1/chat/completions', json={})
        assert transport.count == 0
        assert client.post('https://api.deepseek.com/v1/chat/completions', json={}).status_code == 403
        with pytest.raises(httpx.RequestError):
            client.post('https://api.deepseek.com/v1/chat/completions', json={})
    assert transport.count == 1
    assert json.loads((tmp_path/'response-001-status.json').read_text()) == {'status': 403}
    assert not (tmp_path/'response-001.bin').exists()


def test_timeout_still_consumes_slot_and_does_not_log_exception_message(tmp_path):
    def fail(request):
        raise httpx.ReadTimeout('secret-fixture', request=request)
    transport = CappedTransport(tmp_path, httpx.MockTransport(fail))
    with httpx.Client(transport=transport) as client, pytest.raises(httpx.ReadTimeout):
        client.post('https://api.deepseek.com/v1/chat/completions', json={})
    assert transport.count == 1
    assert json.loads((tmp_path/'request-001-error.json').read_text()) == {'error_type': 'ReadTimeout'}


def test_reduced_budget_caps_dispatch_and_summary(tmp_path):
    transport = CappedTransport(tmp_path, httpx.MockTransport(lambda r: httpx.Response(200, json={})), limit=2)
    with httpx.Client(transport=transport) as client:
        for _ in range(2):
            client.post('https://api.deepseek.com/v1/chat/completions', json={})
        with pytest.raises(httpx.RequestError):
            client.post('https://api.deepseek.com/v1/chat/completions', json={})
    assert transport.count == 2
    assert run_summary([], transport.count, 3, 2)['limit'] == 2


@pytest.mark.parametrize('limit', [0, -1, 21, True, 1.5])
def test_invalid_budget_rejected_before_transport_creation(tmp_path, limit):
    with pytest.raises(ValueError):
        CappedTransport(tmp_path, limit=limit)
