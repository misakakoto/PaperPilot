import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest


spec = importlib.util.spec_from_file_location('prototype_comparison', Path(__file__).parents[1]/'scripts/compare_research_prototypes.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def case():
    return {'id': 'heldout', 'question': 'Modify the selected baseline',
            'conversation_id': 'owned', 'parent_run_id': 'parent', 'source_ids': ['source'],
            'prototype_profile_id': 'pinned-snunet'}


def test_both_modes_and_repeats_keep_exact_baseline_and_sources():
    jobs = comparison.requests([case()], 'campaign')
    assert len(jobs) == 4
    assert len({j['request_id'] for j in jobs}) == 4
    assert {j['prototype_mode'] for j in jobs} == {'adaptive', 'simple_loop'}
    assert all(j['prototype_profile_id'] == 'pinned-snunet' and j['source_ids'] == ['source'] for j in jobs)


def test_comparison_rejects_duplicate_case_ids():
    with pytest.raises(ValueError, match='Duplicate'):
        comparison.requests([case(), case()], 'campaign')


def test_comparison_preserves_same_constraints_in_both_modes():
    value = case()
    value['prototype_constraints'] = {'input_cases': [[1,48,80]], 'max_parameters': 100}
    jobs = comparison.requests([value], 'constraints')
    assert all(j['prototype_constraints'] == value['prototype_constraints'] for j in jobs)
    jobs[0]['prototype_constraints']['max_parameters'] = 1
    assert jobs[1]['prototype_constraints']['max_parameters'] == 100


def test_real_local_http_submission_uses_selected_service_and_presaves_intent(tmp_path):
    output = tmp_path/'receipts.json'
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert json.loads(output.read_text())[-1]['status'] == 'STARTED'
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, payload))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'run_id': 'fixture-'+payload['request_id']}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        comparison.submit_jobs(comparison.requests([case()], 'campaign'), output,
                               base_url=f'http://127.0.0.1:{server.server_port}')
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert len(seen) == 4
    assert all(path == '/api/agentic/turns' and payload['prototype_profile_id'] == 'pinned-snunet'
               for path, payload in seen)
    assert all(r['status'] == 'SUBMITTED' for r in json.loads(output.read_text()))


def test_unknown_submission_stops_without_retry_or_next_job(tmp_path, monkeypatch):
    output = tmp_path/'receipts.json'
    calls = []

    def lost_response(request, **kwargs):
        calls.append(request)
        raise TimeoutError('Response lost after possible acceptance')

    monkeypatch.setattr(comparison.urllib.request, 'urlopen', lost_response)
    with pytest.raises(TimeoutError):
        comparison.submit_jobs(comparison.requests([case()], 'campaign'), output,
                               base_url='http://127.0.0.1:12345')
    records = json.loads(output.read_text())
    assert len(calls) == len(records) == 1
    assert records[0]['status'] == 'UNKNOWN'
    with pytest.raises(FileExistsError):
        comparison.submit_jobs([], output)
    assert len(calls) == 1


def test_dry_run_does_not_make_http_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('HTTP in dry run'))
    output = tmp_path/'receipts.json'
    comparison.submit_jobs(comparison.requests([case()], 'campaign'), output)
    assert all(r['status'] == 'NOT_RUN' for r in json.loads(output.read_text()))
