"""UI contracts cross the same interface as the browser; no paid providers."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperpilot_agentic.models import Budget, RunStatus, ExperimentProtocol, SandboxPolicy
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.sandbox import SandboxExperimentRunner
from paperpilot_agentic.web_api import create_fastapi_router
from paperpilot_agentic.workbench import ResearchWorkbench, cursor_decode


class Background:
    def __init__(self): self.submitted = []
    def active(self, rid): return rid in self.submitted
    def submit(self, rid, operation): self.submitted.append(rid)


@pytest.fixture
def wb(tmp_path):
    runtime = ResearchAgentRuntime(db_path=tmp_path/'runtime.db',workspace=tmp_path,
                                   text_reader_protocol='test')
    return ResearchWorkbench(runtime,Background())


def completed(wb, cid='conv', question='Question'):
    rid = wb.db.create_run(question,Budget(),{'conversation_id':cid,'engine':'answer_first_text'})
    wb.db.update_run_status(rid,RunStatus.COMPLETED)
    return rid


def test_browser_projection_exposes_persisted_answer_sections(wb):
    from paperpilot_agentic.durable_reader import publish_reader_answer
    rid = completed(wb)
    sections = [{'kind': 'direct', 'text': 'Answer.'}, {'kind': 'context', 'text': 'Condition.'}]
    publish_reader_answer(wb.db, rid, {'status': 'REVIEW_REQUIRED',
        'answer': 'Answer.\n\nCondition.', 'answer_sections': sections})
    assert wb.turn(rid)['message']['answer_sections'] == sections


def test_conversation_paging_does_not_depend_on_recent_200_runs(wb):
    for i in range(211): completed(wb,f'conv-{i:03}')
    ids, cursor = [], None
    while True:
        page = wb.conversations(cursor=cursor,limit=37)
        ids.extend(r['conversation_id'] for r in page['items'])
        cursor = page['next_cursor']
        if not cursor: break
    assert len(ids) == len(set(ids)) == 211


def test_journal_failure_does_not_mask_execution_failure(wb, monkeypatch, caplog):
    from paperpilot_agentic.research_journal import ResearchJournal
    rid = completed(wb)
    def fail_resume(_):
        raise ValueError('original execution failure')
    def fail_journal(*_):
        raise RuntimeError('projection unavailable')
    monkeypatch.setattr(wb.runtime, '_resume', fail_resume)
    monkeypatch.setattr(ResearchJournal, 'record', fail_journal)
    with pytest.raises(ValueError, match='original execution failure'):
        wb.runtime.resume(rid)
    assert 'Research journal update failed' in caplog.text
    assert wb.db.get_run(rid)['status'] == 'COMPLETED'


def test_turn_pagination_keeps_whole_long_conversation(wb):
    expected = {completed(wb,question=f'Question {i}') for i in range(67)}
    seen, cursor = [], None
    while True:
        page = wb.runs('conv',cursor=cursor,limit=13)
        seen += [t['run_id'] for t in page['items']]
        cursor = page['next_cursor']
        if not cursor: break
    assert set(seen) == expected and len(seen) == 67


def test_idempotent_submission_is_persistent_and_checks_payload(wb):
    payload = {'request_id':'unique-request','question':'Explain the mechanism'}
    one = wb.submit(payload)
    fresh = ResearchWorkbench(wb.runtime,wb.background)
    assert fresh.submit(payload)['run_id'] == one['run_id']
    assert len(wb.background.submitted) == 1
    with pytest.raises(ValueError,match='different input'):
        fresh.submit({**payload,'question':'Another question'})


def test_concurrent_same_request_creates_at_most_one_run(wb):
    payload = {'request_id':'concurrent-key','question':'Question'}
    def send():
        try: return wb.submit(payload)
        except RuntimeError: return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _:send(),range(4)))
    ids = {r['run_id'] for r in results if r}
    assert len(ids) == 1 and len(wb.runtime.list_runs()) == 1
    assert len(wb.background.submitted) == 1


def test_uncertain_submission_does_not_launch_again(wb):
    payload={'request_id':'uncertain-key','question':'question'}
    import hashlib
    fp=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    with wb.db.transaction() as c:
        c.execute('INSERT INTO workbench_submissions(request_id,fingerprint,state,created_at) VALUES(?,?,?,?)',
                  (payload['request_id'],fp,'STARTED','now'))
    with pytest.raises(RuntimeError,match='uncertain'): wb.submit(payload)
    assert not wb.background.submitted


def test_foreign_parent_and_client_paths_are_rejected(wb):
    rid=completed(wb)
    with pytest.raises(ValueError,match='another conversation'):
        wb.submit({'request_id':'foreign-key','conversation_id':'other','parent_run_id':rid,'question':'Q'})
    with pytest.raises(ValueError,match='Unknown'):
        wb.submit({'request_id':'path-key','question':'Q','command':['python','evil.py']})


def test_cancel_status_distinguishes_request_from_quiescence(wb):
    rid=wb.runtime.enqueue(question='Q')['run']['run_id']
    wb.background.submitted.append(rid)
    wb.runtime.cancel(rid)
    assert wb.turn(rid)['status']=='STOPPING'
    assert not wb.turn(rid)['can_follow_up']
    wb.background.submitted.clear()
    assert wb.turn(rid)['status']=='CANCELLED'


def test_completed_run_cancel_does_not_rewrite_history(wb):
    rid=completed(wb)
    assert wb.runtime.cancel(rid)['run']['status']=='COMPLETED'


def test_partial_answer_not_presented_as_success(wb):
    rid=completed(wb)
    wb.db.update_run_metadata(rid,{'final_answer':{'answer':'A supported fragment','review_required':True}})
    value=wb.turn(rid)
    assert '核验' in value['state_text']
    assert value['message']['answer']=='A supported fragment'


def test_background_scope_reaches_browser_without_rewriting_original(wb):
    rid = completed(wb)
    final = {'answer': 'A supported example, not a benchmark score.',
             'answer_scope': 'BACKGROUND_ONLY', 'direct_answer_supported': False}
    wb.db.update_run_metadata(rid, {'final_answer': final})
    assert wb.turn(rid)['message']['answer_scope'] == 'BACKGROUND_ONLY'
    assert wb.db.get_run(rid)['metadata']['final_answer'] == final


@pytest.mark.parametrize('cursor',['??','a', 'WzFd'])
def test_bad_cursor_is_rejected(cursor):
    with pytest.raises(ValueError): cursor_decode(cursor)


def test_artifact_preview_checks_ownership_and_digest(wb,tmp_path):
    path=tmp_path/'draft.py';path.write_text('print(1)',encoding='utf-8')
    import hashlib
    rid=completed(wb)
    aid=wb.db.add_artifact(rid,None,{'path':str(path),'kind':'draft','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    assert wb.artifact(rid,aid)['content']=='print(1)'
    path.write_text('print(2)',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'): wb.artifact(rid,aid)
    with pytest.raises(KeyError): wb.artifact(rid,'unknown')


def test_real_owned_process_is_stopped_on_cancel(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'worker.py').write_text('import time\nprint("started",flush=True)\ntime.sleep(30)\n',encoding='utf-8')
    cancel=threading.Event()
    runner=SandboxExperimentRunner(cancel_requested=cancel.is_set)
    protocol=ExperimentProtocol(name='cancel-check',command=[sys.executable,'-u','worker.py'],
        source_dir=str(source),output_dir=str(tmp_path/'output'),metric_files=[],
        policy=SandboxPolicy(backend='local',network='host',allow_local_fallback=True,
            allowed_executables=(Path(sys.executable).name,),timeout_seconds=40))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(runner.run,protocol)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            logs=list((tmp_path/'output').glob('*/stdout.log'))
            if logs and 'started' in logs[0].read_text(): break
            if future.done(): pytest.fail(str(future.result()))
            time.sleep(.05)
        else: pytest.fail('Worker did not start')
        start=time.monotonic();cancel.set();result=future.result(timeout=10)
    assert time.monotonic()-start<10
    assert result.status=='FAILED' and 'cancelled' in result.error
    assert result.exit_code is not None and not result.timed_out


def test_workbench_routes_have_one_prefix_and_return_typed_view(wb):
    rid=completed(wb)
    app=FastAPI();app.include_router(create_fastapi_router(runtime_factory=lambda:wb.runtime,prefix='/api/agentic'))
    with TestClient(app) as client:
        page=client.get('/api/agentic/conversations').json()
        assert page['items'][0]['conversation_id']=='conv'
        view=client.get(f'/api/agentic/runs/{rid}/view').json()
        assert all(k in view for k in ('message','activities','approvals','artifacts'))
        assert client.get(f'/api/agentic/runs/{rid}/notebook').status_code==200
        assert client.get('/api/agentic/papers').json()['available'] is False


def test_sse_last_event_id_resumes_after_delivered_event(wb):
    rid=completed(wb)
    marker=wb.db.emit_event(rid,'test.first',{})
    wb.db.emit_event(rid,'test.second',{})
    app=FastAPI();app.include_router(create_fastapi_router(runtime_factory=lambda:wb.runtime,prefix='/api/agentic'))
    with TestClient(app) as client:
        output=client.get(f'/api/agentic/runs/{rid}/events',headers={'Last-Event-ID':str(marker)}).text
    assert 'test.second' in output and 'test.first' not in output
