import copy
import json

import pytest

from paperpilot_agentic.models import ToolCallStatus
from paperpilot_agentic.prototype_workspace import plan, complete
from test_prototype_workspace import waiting, tasks


def completed(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    runtime.db.update_tool_call(state['pending']['call'], ToolCallStatus.COMPLETED,
                               result={'ok':True,'output':{'passed':True}})
    harness._training(rid, state)
    rows = tasks()+[{'id':'review','goal':'review','capability':'review',
                     'depends_on':['design','code','check'],'acceptance':'consistent'}]
    plan(state['work'], rows)
    complete(state['work'], 'review', 'review', {'implementation_consistent':True,'issues':[],
                                               'limitations':['Maintained fixture only']})
    harness._apply(rid, state, {'action':'finish','reason':'fixture','arguments':{'summary':'fixture completed'}}, 6)
    assert state['completed']
    harness._save(rid, state)
    harness._publish(rid, state)
    return runtime, harness, rid


def test_revision_reuses_code_but_invalidates_validation_and_review_without_reset(tmp_path, monkeypatch):
    runtime, harness, rid = completed(tmp_path, monkeypatch)
    before = harness._load(rid)
    run = runtime.db.get_run(rid)
    result = harness.revise_constraints(rid, {'input_cases':[[1,48,80]],'max_parameters':10},
                                        expected_revision=before['work']['revision'])
    after = harness._load(rid)
    assert result['revision_applied'] is True and after['phase']=='READY'
    assert after['work']['task_order'] == before['work']['task_order']
    for key in ('read','design','code'):
        assert after['work']['tasks'][key] == before['work']['tasks'][key]
    assert all(after['work']['tasks'][k]['status']=='PENDING' for k in ('check','review'))
    assert after['work']['artifacts'] == before['work']['artifacts']
    assert after['constraint_revisions'][-1]['previous_summary']=='fixture completed'
    assert runtime.db.get_run(rid)['metadata']['final_answer'] is None
    assert after['work']['repairs']==before['work']['repairs'] and after['turn']==before['turn']
    for key in ('created_at','budget','used_llm_calls','used_tokens'):
        assert runtime.db.get_run(rid)[key] == run[key]
    with pytest.raises(RuntimeError, match='revision'):
        harness.revise_constraints(rid, {}, expected_revision=before['work']['revision'])


@pytest.mark.parametrize('already_approved',[False,True])
def test_revision_revokes_unsent_approval_and_requires_new_specific_approval(tmp_path, monkeypatch, already_approved):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before = harness._load(rid)
    old = before['pending']
    if already_approved:
        runtime.approve(old['approval'])
    harness.revise_constraints(rid, {'max_parameters':10}, expected_revision=before['work']['revision'])
    assert runtime.db.get_approval(old['approval'])['status']==('APPROVED' if already_approved else 'REJECTED')
    assert runtime.db.get_tool_call(old['call'])['status']=='REJECTED'
    state = harness._load(rid)
    harness._apply(rid, state, {'action':'work','reason':'recheck',
                              'arguments':{'task_id':'check','content':None}}, 6)
    assert state['pending']['approval'] != old['approval']
    assert state['pending']['binding']['code'] == old['binding']['code']
    assert harness._training(rid, state) is False


def test_revision_is_atomic_if_state_commit_fails(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before = harness._load(rid)
    with runtime.db.transaction() as connection:
        connection.execute("CREATE TRIGGER fail_revision BEFORE UPDATE ON research_episodes BEGIN SELECT RAISE(ABORT,'fixture commit failure'); END")
    with pytest.raises(Exception, match='fixture commit failure'):
        harness.revise_constraints(rid, {'max_parameters':10}, expected_revision=before['work']['revision'])
    assert harness._load(rid)==before
    assert runtime.db.get_approval(before['pending']['approval'])['status']=='PENDING'
    assert runtime.db.get_run(rid)['status']=='WAITING_APPROVAL'


def test_revision_refuses_unknown_or_running_work(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    state['phase']='UNKNOWN'
    harness._save(rid, state)
    with pytest.raises(ValueError, match='stable'):
        harness.revise_constraints(rid, {}, expected_revision=state['work']['revision'])
    state['phase']='WAITING_APPROVAL'
    harness._save(rid,state)
    runtime.db.update_tool_call(state['pending']['call'],ToolCallStatus.RUNNING)
    with pytest.raises(RuntimeError, match='execution'):
        harness.revise_constraints(rid, {}, expected_revision=state['work']['revision'])
    runtime.db.update_tool_call(state['pending']['call'],ToolCallStatus.FAILED,result={'ok':False})
    with pytest.raises(RuntimeError, match='execution'):
        harness.revise_constraints(rid, {'max_parameters':10}, expected_revision=state['work']['revision'])


def test_identical_contract_is_a_read_only_noop(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before = copy.deepcopy(harness._load(rid))
    result=harness.revise_constraints(rid, {}, expected_revision=before['work']['revision'])
    assert result['revision_applied'] is False
    assert harness._load(rid)==before
    assert runtime.db.get_approval(before['pending']['approval'])['status']=='PENDING'


def test_revision_saves_unfinished_requirements_when_decision_budget_is_spent(tmp_path, monkeypatch):
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before = runtime.db.get_run(rid)
    budget = before['budget'] | {'max_llm_calls':before['used_llm_calls']}
    with runtime.db.transaction() as connection:
        connection.execute('UPDATE runs SET budget_json=? WHERE run_id=?',(json.dumps(budget),rid))
    state=harness._load(rid)
    harness.revise_constraints(rid, {'max_parameters':10}, expected_revision=state['work']['revision'])
    after=harness._load(rid)
    assert after['validation_contract']['max_parameters']==10 and after['phase']=='PARTIAL'
    assert 'budget exhausted' in after['stop_reason']
    assert runtime.db.get_run(rid)['used_llm_calls']==before['used_llm_calls']
    assert runtime.db.get_run(rid)['budget']==budget


def test_unknown_physical_request_and_unapplied_response_cannot_be_bypassed(tmp_path, monkeypatch):
    from paperpilot_agentic.llm_attempts import AttemptLedger
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    before=harness._load(rid)
    with runtime.db.transaction() as connection:
        connection.execute('INSERT INTO research_episode_turns VALUES(?,?,?,?,?,?,?,?)',
                           (rid,6,'lead','{}','RETURNED',0,'{"text":"saved response"}',None))
    with pytest.raises(RuntimeError, match='saved model turn'):
        harness.revise_constraints(rid, {}, expected_revision=before['work']['revision'])
    assert harness._load(rid)==before
    AttemptLedger(runtime.db).begin(rid,stage='fixture-unknown',model='fixture',messages=[],output_tokens=1,
        required_tokens=2,policy_sha256=None,max_calls=16,max_tokens=100000)
    with pytest.raises(RuntimeError, match='unknown'):
        harness.revise_constraints(rid, {}, expected_revision=before['work']['revision'])
    assert harness._load(rid)==before


def test_workbench_revision_is_owned_and_idempotent(tmp_path, monkeypatch):
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background
    runtime,harness,rid=waiting(tmp_path,monkeypatch)
    background=Background()
    wb=ResearchWorkbench(runtime,background)
    payload={'request_id':'revision-web-fixture','action':'revise_prototype_constraints',
             'conversation_id':rid,'parent_run_id':rid,'expected_revision':harness._load(rid)['work']['revision'],
             'prototype_constraints':{'max_parameters':10}}
    with pytest.raises(ValueError,match='another conversation'):
        wb.submit(payload | {'request_id':'revision-foreign-fixture','conversation_id':'foreign'})
    result=wb.submit(payload)
    assert result['run_id']==rid and result['start']
    assert wb.submit(payload)['reused'] and background.submitted==[rid]
    assert len(harness._load(rid)['constraint_revisions'])==1


def test_reusing_code_with_new_contract_freezes_separate_execution_assets(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runtime,harness,rid=waiting(tmp_path,monkeypatch)
    state=harness._load(rid)
    code_id=state['pending']['binding']['code']
    sources=[]

    def execute(self,protocol):
        sources.append(protocol.source_dir)
        return SimpleNamespace(status='COMPLETED',exit_code=0,timed_out=False,
                               metrics={'passed':True,'parameters':7},to_dict=lambda:{'status':'COMPLETED'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run',execute)
    runtime.approve(state['pending']['approval'])
    assert harness._training(rid,state)
    harness.revise_constraints(rid,{'max_parameters':10,'input_cases':[[1,48,80]]},
                               expected_revision=state['work']['revision'])
    state=harness._load(rid)
    harness._apply(rid,state,{'action':'work','reason':'new inputs',
                            'arguments':{'task_id':'check','content':None}},6)
    runtime.approve(state['pending']['approval'])
    assert harness._training(rid,state)
    assert state['pending']['binding']['code']==code_id
    assert len(sources)==2 and sources[0]!=sources[1]


def test_identical_code_in_independent_runs_cannot_overwrite_execution_history(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runtime,harness,first=waiting(tmp_path,monkeypatch)
    second=harness.start('Independent fixture','independent-fixture','parent',['source'])['run']['run_id']
    harness.advance(second)
    states=[harness._load(rid) for rid in (first,second)]
    assert states[0]['pending']['binding']['code']==states[1]['pending']['binding']['code']
    sources=[]

    def execute(self,protocol):
        sources.append(protocol.source_dir)
        return SimpleNamespace(status='COMPLETED',exit_code=0,timed_out=False,metrics={'passed':True},
                               to_dict=lambda:{'status':'COMPLETED'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run',execute)
    for rid,state in zip((first,second),states):
        runtime.approve(state['pending']['approval'])
        assert harness._training(rid,state)
    assert sources[0]!=sources[1]


def test_delivery_reads_historical_verifier_from_execution_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from io import BytesIO
    from zipfile import ZipFile
    from paperpilot_agentic import prototype_delivery
    from paperpilot_agentic.prototype_workspace import digest
    runtime,harness,rid=waiting(tmp_path,monkeypatch)
    state=harness._load(rid)
    runtime.approve(state['pending']['approval'])
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run',lambda *a:
        SimpleNamespace(status='COMPLETED',exit_code=0,timed_out=False,metrics={'passed':True},to_dict=lambda:{}))
    harness._training(rid,state)
    monkeypatch.setattr(prototype_delivery,'Path',lambda *a:SimpleNamespace(
        with_name=lambda *a:SimpleNamespace(read_bytes=lambda:b'# Later verifier version\n')))
    with ZipFile(BytesIO(prototype_delivery.export_prototype(runtime,rid))) as archive:
        assert digest(archive.read('verifier/verify.py').decode())==state['worker_identity']
