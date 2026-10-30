import json
from copy import deepcopy
import pytest
from paperpilot_agentic.request_state import empty_request_state, apply_request_delta, request_work_view


def apply(state, sid, text, ops):
    return apply_request_delta(state, {'id':sid,'text':text},
        {'expected_revision':state['revision'], 'operations':ops})


def initial():
    text = '任务A：设计RGB变化检测，先设计实验。'
    return apply(empty_request_state(), 'r1', text, [
        {'kind':'create_task','task_id':'A','label':'变化检测','quote':text},
        {'kind':'activate_task','task_id':'A','quote':text},
        {'kind':'set_condition','task_id':'A','key':'sensor','text':'RGB','quote':'RGB'},
        {'kind':'add_todo','task_id':'A','key':'design','text':'设计实验','quote':'设计实验'}])


def test_replacement_removes_old_value_only_from_work_view():
    state = initial()
    old = state['tasks']['A']['conditions']['sensor']['id']
    revised = apply(state,'r2','输入改用单通道SAR',[{
        'kind':'set_condition','task_id':'A','key':'sensor','text':'单通道SAR',
        'quote':'输入改用单通道SAR','replaces':old}])
    view = request_work_view(revised)['view']
    assert view['conditions']['sensor']['text'] == '单通道SAR'
    assert view['pending_actions']['design']['status'] == 'pending'
    assert revised['tasks']['A']['archive'][0]['text'] == 'RGB'
    assert revised['tasks']['A']['archive'][0]['status'] == 'superseded'
    assert revised['sources']['r1'] == '任务A：设计RGB变化检测，先设计实验。'
    assert state['tasks']['A']['conditions']['sensor']['text'] == 'RGB'


def test_task_switch_and_return_isolate_conditions_and_todos():
    state = initial()
    state = apply(state,'r2','另开任务B，分析SAR分割。',[
        {'kind':'create_task','task_id':'B','label':'分割','quote':'任务B'},
        {'kind':'activate_task','task_id':'B','quote':'另开任务B'},
        {'kind':'set_condition','task_id':'B','key':'sensor','text':'SAR','quote':'SAR'}])
    assert request_work_view(state)['view']['pending_actions'] == {}
    state = apply(state,'r3','回到任务A',[{'kind':'activate_task','task_id':'A','quote':'回到任务A'}])
    assert request_work_view(state)['view']['conditions']['sensor']['text'] == 'RGB'
    assert 'design' in request_work_view(state)['view']['pending_actions']


@pytest.mark.parametrize('kind,key', [('retract_condition','sensor'),('cancel_todo','design')])
def test_retraction_archives_provenance(kind,key):
    state = initial()
    slot = 'conditions' if kind=='retract_condition' else 'todos'
    old = state['tasks']['A'][slot][key]
    state = apply(state,'r2','取消这项要求和待办',[{'kind':kind,'task_id':'A','key':key,
        'quote':'取消这项要求和待办','replaces':old['id']}])
    assert key not in state['tasks']['A'][slot]
    assert state['tasks']['A']['archive'][0]['status'] == 'cancelled'


@pytest.mark.parametrize('bad', [
    {'kind':'set_condition','task_id':'A','key':'sensor','text':'SAR','quote':'改为SAR'},
    {'kind':'set_condition','task_id':'A','key':'new','text':'SAR','quote':'invented'},
    {'kind':'complete_todo','task_id':'A','key':'design','quote':'改为SAR'},
    {'kind':'activate_task','task_id':'missing','quote':'改为SAR'},
    None,
])
def test_invalid_late_edit_is_atomic(bad):
    state = initial()
    before = deepcopy(state)
    with pytest.raises(ValueError):
        apply(state,'r2','改为SAR',[
            {'kind':'add_todo','task_id':'A','key':'extra','text':'check','quote':'改为SAR'},bad])
    assert state == before


def test_stale_revision_and_conflicting_replay_rejected():
    state = initial()
    source = {'id':'r2','text':'继续'}
    delta = {'expected_revision':state['revision'],'operations':[]}
    current = apply_request_delta(state,source,delta)
    assert apply_request_delta(current,source,delta) == current
    with pytest.raises(ValueError,match='Stale'):
        apply_request_delta(current,{'id':'r3','text':'继续'},delta)
    with pytest.raises(ValueError,match='different'):
        apply_request_delta(current,{'id':'r2','text':'改了'},delta)


def test_many_updates_keep_bounded_view_and_reloadable_originals():
    state = initial()
    for i in range(200):
        text = f'输入改为传感器{i}'
        state = apply(state,f'update{i}',text,[{
            'kind':'set_condition','task_id':'A','key':'sensor','text':f'传感器{i}',
            'quote':text,'replaces':state['tasks']['A']['conditions']['sensor']['id']}])
    state = json.loads(json.dumps(state,ensure_ascii=False))
    result = request_work_view(state,max_bytes=1000)
    assert result['status'] == 'READY'
    assert result['view']['conditions']['sensor']['text'] == '传感器199'
    assert len(state['sources']) == 201
    assert len(state['tasks']['A']['archive']) == 200
    assert len(state['events']) == 201
    assert request_work_view(state,max_bytes=10)['view'] is None
    assert request_work_view(state,max_bytes=10)['status'] == 'OVERFLOW'


@pytest.mark.parametrize('field', ['task_id','quote','key','text'])
def test_missing_field_error_identifies_exact_operation(field):
    op={'kind':'add_todo','task_id':'A','key':'extra','text':'检查','quote':'检查'}
    del op[field]
    with pytest.raises(ValueError,match=rf'operations\[0\].{field}'):
        apply(initial(),'r2','检查',[op])


def test_pause_resume_preserves_unfinished_work_and_provenance():
    state=initial()
    original=deepcopy(state['tasks']['A']['todos']['design'])
    state=apply(state,'r2','先暂停设计',[{'kind':'pause_todo','task_id':'A','key':'design',
        'quote':'暂停设计','replaces':original['id']}])
    view=request_work_view(state)['view']
    assert view['pending_actions']=={}
    assert view['paused_actions']['design']['status']=='paused'
    assert view['paused_actions']['design']['status_source_id']=='r2'
    paused=state['tasks']['A']['todos']['design']
    assert paused['source_id']==original['source_id']
    assert paused['status_source_id']=='r2'
    with pytest.raises(ValueError,match='expected state'):
        apply(state,'r3','恢复设计',[{'kind':'resume_todo','task_id':'A','key':'design',
            'quote':'恢复设计','replaces':original['id']}])
    state=apply(state,'r3','恢复设计',[{'kind':'resume_todo','task_id':'A','key':'design',
        'quote':'恢复设计','replaces':paused['id']}])
    assert 'design' in request_work_view(state)['view']['pending_actions']
    assert request_work_view(state)['view']['paused_actions']=={}
    assert state['tasks']['A']['todos']['design']['text']==original['text']


def test_rename_keeps_old_label_and_does_not_replace_task_identity():
    state=initial()
    original=deepcopy(state['tasks']['A'])
    state=apply(state,'r2','输入改成SAR，任务仍是变化检测',[{
        'kind':'rename_task','task_id':'A','previous_label':original['label'],
        'label':'SAR变化检测','quote':'输入改成SAR'}])
    assert state['active_task_id']=='A'
    assert state['tasks']['A']['label_history'][0]['label']==original['label']
    assert state['tasks']['A']['todos']==original['todos']
    assert state['tasks']['A']['conditions']==original['conditions']
    with pytest.raises(ValueError,match='current task label'):
        apply(state,'r3','修改名称',[{'kind':'rename_task','task_id':'A',
            'previous_label':'wrong','label':'new','quote':'修改名称'}])
