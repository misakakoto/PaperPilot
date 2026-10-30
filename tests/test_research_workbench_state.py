import json

import pytest

from paperpilot_agentic.models import Budget, RunStatus
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_journal import ResearchJournal
from paperpilot_agentic.session_router import needs_planning, validate_route
from paperpilot_agentic.workbench import ResearchWorkbench
from test_reader_pipeline import factory
from test_workbench import Background, completed


class RouterModel:
    model = 'fixture-router'

    def __init__(self, action='clarify'):
        self.calls = 0
        self.action = action

    def complete(self, messages, before_attempt=None, **kwargs):
        before_attempt()
        self.calls += 1
        return LLMResponse(text=json.dumps({'action':self.action,'goal':'Test hypothesis',
            'steps':['Read sources','Review protocol'],'gaps':['Need a reviewed protocol'],
            'reader_id':None,'source_ids':[],'validation_id':None}),finish_reason='stop')


def test_router_complexity_fast_lane():
    assert not needs_planning('What is the main architecture?')
    assert not needs_planning('论文使用什么训练数据，实验条件是什么？')
    assert not needs_planning('What training data did the experiment use?')
    assert needs_planning('根据论文设计实验，但不启动训练')


def test_routed_execution_request_stops_before_authorization_and_replays(tmp_path):
    model=RouterModel()
    runtime=factory(tmp_path,model)
    wb=ResearchWorkbench(runtime,Background())
    result=wb.submit({'request_id':'request-routing-1','action':'auto','question':'请开始训练实验'})
    rid=result['run_id']
    assert runtime.db.get_run(rid)['metadata']['engine']=='research_session'
    current=runtime.resume(rid)
    assert current['run']['status']=='COMPLETED'
    assert current['run']['research_outcome']=='UNRESOLVED'
    assert current['tool_calls']==[]
    assert wb.journal.state(result['conversation_id'])['revision']==1
    runtime.resume(rid)
    assert model.calls==1
    assert wb.journal.view(result['conversation_id'])['total']==1


def test_model_cannot_route_to_training(tmp_path):
    model=RouterModel('execute')
    runtime=factory(tmp_path,model)
    wb=ResearchWorkbench(runtime,Background())
    rid=wb.submit({'request_id':'request-routing-2','action':'auto','question':'实验设计'})['run_id']
    current=runtime.resume(rid)
    assert current['run']['status']=='FAILED'
    assert not current['tool_calls']


def test_router_rejects_foreign_evidence():
    with pytest.raises(ValueError,match='owned'):
        validate_route({'action':'design','goal':'Draft','steps':[],'gaps':[],
            'reader_id':'foreign','source_ids':['foreign'],'validation_id':'foreign'},
            {'readers':[],'validations':[]})


def test_uncertain_planning_call_is_not_replayed(tmp_path):
    from paperpilot_agentic.durable_reader import _reader_checkpoint
    import paperpilot_agentic.session_router as module
    model=RouterModel()
    runtime=factory(tmp_path,model)
    wb=ResearchWorkbench(runtime,Background())
    rid=wb.submit({'request_id':'request-routing-3','action':'auto','question':'实验设计'})['run_id']
    real=module._reader_checkpoint
    def interrupted(db,run,fingerprint,**kwargs):
        if kwargs.get('report') is None:
            _reader_checkpoint(db,run,fingerprint,**kwargs)
            return real(db,run,fingerprint,**kwargs)
        return real(db,run,fingerprint,**kwargs)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module,'_reader_checkpoint',interrupted)
        runtime.resume(rid)
    runtime.resume(rid)
    assert model.calls==0
    assert runtime.db.get_run(rid)['status']=='RUNNING'
    assert runtime.db.list_events(rid)[-1]['event_type']=='research.routing_blocked'


def test_journal_revision_conflicts_and_paged_hundred_experiments(tmp_path):
    runtime=factory(tmp_path,RouterModel())
    journal=ResearchJournal(runtime.db)
    goal={'goal':'Compare models','constraints':['never use test split'],'open_questions':['generalization']}
    journal.revise('c',goal,expected_revision=0,source_run_id=None)
    journal.revise('c',{**goal,'goal':'Compare encoders'},expected_revision=1,source_run_id=None)
    with pytest.raises(ValueError,match='changed'):
        journal.revise('c',goal,expected_revision=1,source_run_id=None)
    for i in range(101):
        rid=runtime.db.create_run(f'Experiment {i}',Budget(),{'conversation_id':'c','engine':'method_comparison',
            'method_comparison':{'large_logs':'x'*20000},'final_answer':{'review_required':True}})
        runtime.db.update_run_status(rid,RunStatus.FAILED if i%5==0 else RunStatus.COMPLETED)
        journal.record(rid)
    first=journal.view('c',limit=20)
    assert first['total']==101 and len(first['items'])==20
    assert len(json.dumps(journal.state('c')))<1000  # Logs never enter state context.
    assert all(item['scientific_status']=='NOT_ASSESSED' for item in first['items'])
    assert len(journal.view('c',offset=100)['items'])==1
    assert len(first['revisions'])==2


def test_old_parent_cannot_bypass_active_conversation(tmp_path):
    wb=ResearchWorkbench(factory(tmp_path,RouterModel()),Background())
    parent=completed(wb)
    wb.db.create_run('active',Budget(),{'conversation_id':'conv'})
    with pytest.raises(ValueError,match='unfinished'):
        wb.submit({'request_id':'active-old-parent','action':'attach_reference',
            'conversation_id':'conv','parent_run_id':parent,'binding_id':'anything'})


def test_approval_projection_contains_frozen_context(tmp_path):
    from paperpilot_agentic.models import RiskLevel
    from paperpilot_agentic.tooling import ToolSpec
    wb=ResearchWorkbench(factory(tmp_path,RouterModel()),Background())
    wb.runtime.registry.register(ToolSpec(name='fixture.approval',description='Bound diagnostic',
        input_schema={'type':'object','properties':{},'additionalProperties':False},
        handler=lambda c,a:None,risk_level=RiskLevel.HIGH,requires_approval=True))
    rid=wb.db.create_run('test',Budget())
    step=wb.db.create_step(rid,'fixture',{})
    wb.runtime.tools.invoke(run_id=rid,step_id=step,tool_name='fixture.approval',arguments={},
        approval_context={'binding_id':'frozen','data_scope':{'train_images':2}})
    approval=wb.turn(rid)['approvals'][0]
    assert approval['tool_name']=='fixture.approval'
    assert approval['context']['binding_id']=='frozen'
