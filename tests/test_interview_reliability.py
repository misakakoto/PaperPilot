"""Six faults through production persistence and HTTP, using local fake providers."""
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.durable_reader import run_durable_reader, ReaderAlreadyStarted
from paperpilot_agentic.models import Budget
from paperpilot_agentic.web_api import create_fastapi_router
from test_durable_reader import reader_setup
from test_reader_pipeline import factory


def child_process(folder, body):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        str(p) for p in (root/'src', root/'tests', root)))
    script = '''import os, sys
from pathlib import Path
from test_durable_reader import reader_setup
from paperpilot_agentic.durable_reader import run_durable_reader
folder = Path(sys.argv[1])
db, rid, provider, model, evidence = reader_setup(folder)
(folder/'run-id.txt').write_text(rid)
''' + body
    return subprocess.run([sys.executable, '-c', script, str(folder)], env=env,
                          capture_output=True, text=True, timeout=180)


def wait_complete(client, rid):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = client.get('/agentic/runs/' + rid).json()
        if result['run']['status'] in {'COMPLETED', 'FAILED', 'BUDGET_EXHAUSTED'}:
            return result
        time.sleep(.02)
    raise AssertionError('local fake provider did not finish')


def test_refresh_and_duplicate_http_submission_reuse_completed_response(tmp_path):
    from test_durable_reader import AnswerProvider
    provider = AnswerProvider()
    runtime = factory(tmp_path, provider)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    body = {'request_id': 'frozen-request', 'action': 'answer', 'question': 'What fact?',
            'visual_policy': 'off'}
    with TestClient(app) as client:
        first = client.post('/agentic/turns', json=body)
        assert first.status_code == 202
        rid = first.json()['run_id']
        completed = wait_complete(client, rid)
        assert completed['run']['status'] == 'COMPLETED'
        calls = provider.calls
        assert calls == 2
        for _ in range(3):
            refreshed = client.get('/agentic/runs/' + rid).json()
            assert refreshed['final_answer'] == completed['final_answer']
        duplicate = client.post('/agentic/turns', json=body).json()
        assert duplicate['reused'] is True and duplicate['run_id'] == rid
        assert provider.calls == calls
        assert factory(tmp_path, provider).status(rid)['run']['used_llm_calls'] == calls
        conflict = client.post('/agentic/turns', json={**body, 'question': 'changed'})
        assert conflict.status_code == 400
        assert 'different input' in conflict.json()['detail']
        assert provider.calls == calls


def test_budget_pause_survives_reopen_without_spending_or_resetting(tmp_path):
    db, rid, provider, model, evidence = reader_setup(tmp_path)
    with db.transaction() as conn:
        conn.execute('UPDATE runs SET budget_json=? WHERE run_id=?',
                     (__import__('json').dumps(Budget(max_llm_calls=1).to_dict()), rid))
    report = run_durable_reader(db, rid, evidence, model, protocol_id='fixed')
    assert report['error_type'] == 'LLMBudgetExhausted'
    reopened = RuntimeDatabase(db.path)
    assert reopened.get_run(rid)['status'] == 'BUDGET_EXHAUSTED'
    assert run_durable_reader(reopened, rid, evidence, model, protocol_id='fixed') == report
    assert provider.calls == reopened.get_run(rid)['used_llm_calls'] == 0


def test_real_process_exit_after_saved_response_can_republish_without_provider(tmp_path):
    child = child_process(tmp_path, '''import paperpilot_agentic.durable_reader as reader
reader.publish_reader_answer = lambda *a, **kw: os._exit(37)
run_durable_reader(db, rid, evidence, model, protocol_id='fixed')
''')
    assert child.returncode == 37, child.stderr
    db, _, provider, model, evidence = reader_setup(tmp_path)
    rid = (tmp_path/'run-id.txt').read_text()
    saved = db.get_run(rid)
    assert saved['metadata']['answer_first_checkpoint']['state'] == 'FINISHED'
    assert saved['used_llm_calls'] == 2
    result = run_durable_reader(db, rid, evidence, model, protocol_id='fixed')
    assert db.get_run(rid)['metadata']['final_answer']['answer'] == result['answer']
    assert provider.calls == 0
    assert db.get_run(rid)['used_llm_calls'] == 2


@pytest.mark.parametrize('during_request', [False, True], ids=['interruption', 'unknown-request'])
def test_real_process_interruption_never_steals_started_checkpoint(tmp_path, during_request):
    if during_request:
        body = '''def lost(messages, *, before_attempt, **kw):
    before_attempt()
    os._exit(38)
provider.complete = lost
run_durable_reader(db, rid, evidence, model, protocol_id='fixed')
'''
    else:
        body = '''run_durable_reader(db, rid, evidence, model, protocol_id='fixed',
    on_progress=lambda value: os._exit(38))
'''
    child = child_process(tmp_path, body)
    assert child.returncode == 38, child.stderr
    db, _, provider, model, evidence = reader_setup(tmp_path)
    rid = (tmp_path/'run-id.txt').read_text()
    assert db.get_run(rid)['metadata']['answer_first_checkpoint']['state'] == 'STARTED'
    with pytest.raises(ReaderAlreadyStarted):
        run_durable_reader(db, rid, evidence, model, protocol_id='fixed')
    assert db.get_run(rid)['used_llm_calls'] == int(during_request)
    assert provider.calls == 0
