from copy import deepcopy
import pytest
from test_request_state import initial,apply
from paperpilot_agentic.request_state import request_work_view,apply_request_batch


def scoped():
    state=initial()
    return apply(state,'r2','计划延迟测试，batch size=1；不授权训练',[
        {'kind':'add_todo','task_id':'A','key':'latency','text':'延迟测试','quote':'计划延迟测试'},
        {'kind':'set_condition','task_id':'A','key':'batch','text':'batch size=1','quote':'batch size=1','applies_to':'latency'},
        {'kind':'set_condition','task_id':'A','key':'training','text':'不授权训练','quote':'不授权训练','applies_to':None}])


def test_cancel_only_retires_bound_parameters_and_does_not_revive_them():
    state=scoped()
    old=state['tasks']['A']['todos']['latency']['id']
    state=apply(state,'r3','取消延迟测试',[{'kind':'cancel_todo','task_id':'A','key':'latency','replaces':old,'quote':'取消延迟测试'}])
    assert 'batch' not in request_work_view(state)['view']['conditions']
    assert 'training' in request_work_view(state)['view']['conditions']
    archived=[r for r in state['tasks']['A']['archive'] if r['status']=='activity_cancelled']
    assert archived[0]['text']=='batch size=1'
    assert archived[0]['source_id']=='r2' and archived[0]['changed_by']=='r3'
    state=apply(state,'r4','重新计划延迟测试',[{'kind':'add_todo','task_id':'A','key':'latency','text':'延迟测试','quote':'延迟测试'}])
    assert 'batch' not in request_work_view(state)['view']['conditions']


def test_pause_and_resume_retain_activity_parameters_separately():
    state=scoped()
    for sid,kind in [('r3','pause_todo'),('r4','resume_todo')]:
        state=apply(state,sid,'修改延迟测试状态',[{'kind':kind,'task_id':'A','key':'latency',
            'replaces':state['tasks']['A']['todos']['latency']['id'],'quote':'延迟测试'}])
        view=request_work_view(state)['view']
        assert 'training' in view['conditions']
        assert ('batch' in view['paused_conditions'])==(kind=='pause_todo')
        assert ('batch' in view['conditions'])==(kind=='resume_todo')


def test_replacement_preserves_existing_scope_unless_explicitly_changed():
    state=scoped()
    old=state['tasks']['A']['conditions']['batch']['id']
    state=apply(state,'r3','batch改为2',[{'kind':'set_condition','task_id':'A','key':'batch',
        'replaces':old,'text':'batch=2','quote':'batch改为2'}])
    assert state['tasks']['A']['conditions']['batch']['applies_to']=='latency'


@pytest.mark.parametrize('scope',['unknown',{},True])
def test_invalid_scope_rejected_atomically(scope):
    state=scoped()
    before=deepcopy(state)
    with pytest.raises(ValueError,match='activity scope'):
        apply(state,'r3','test',[{'kind':'set_condition','task_id':'A','key':'new',
            'text':'test','quote':'test','applies_to':scope}])
    assert state==before


def test_batch_cancellation_of_activity_uses_local_reference_and_cascades():
    state=initial()
    result=apply_request_batch(state,[{'id':'r2','text':'延迟测试batch1'},{'id':'r3','text':'取消延迟测试'}],{
        'base_revision':state['revision'],'frames':[
            {'source_id':'r2','operations':[
                {'kind':'add_todo','task_id':'A','key':'latency','text':'延迟测试','quote':'延迟测试'},
                {'kind':'set_condition','task_id':'A','key':'batch','text':'batch1','quote':'batch1','applies_to':'latency'}]},
            {'source_id':'r3','operations':[{'kind':'cancel_todo','task_id':'A','key':'latency',
                'replaces':{'frame':'r2','operation':0},'quote':'取消延迟测试'}]}]})
    assert 'batch' not in result['tasks']['A']['conditions']
    assert result['tasks']['A']['archive'][-1]['status']=='activity_cancelled'
