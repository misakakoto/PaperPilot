from copy import deepcopy
import pytest
from test_request_state import initial,apply
from paperpilot_agentic.request_state import apply_request_batch


def test_content_precondition_can_replace_without_model_generated_record_ids():
    state=initial()
    op={'kind':'set_condition','task_id':'A','key':'sensor','text':'SAR','quote':'SAR',
        'expect':{'text':'RGB','status':'active','applies_to':None}}
    result=apply(state,'r2','改为SAR',[op])
    assert result['tasks']['A']['conditions']['sensor']['text']=='SAR'
    assert 'replaces' not in op
    assert result['events'][-1]['operations'][0]==op


@pytest.mark.parametrize('expected', [None, {'text':'wrong','status':'active','applies_to':None},
    {'text':'RGB','status':'paused','applies_to':None},
    {'text':'RGB','status':'active','applies_to':'other'}, {'text':'RGB'}])
def test_wrong_or_incomplete_prior_state_is_not_inferred(expected):
    state=initial()
    before=deepcopy(state)
    with pytest.raises(ValueError):
        apply(state,'r2','SAR',[{'kind':'set_condition','task_id':'A','key':'sensor','text':'SAR',
                              'quote':'SAR','expect':expected}])
    assert state==before


def test_expectations_batch_uses_sequential_prior_values_and_replays():
    state=initial()
    sources=[{'id':'r2','text':'改为SAR'},{'id':'r3','text':'改为RGB'}]
    proposal={'base_revision':state['revision'],'frames':[
        {'source_id':'r2','operations':[{'kind':'set_condition','task_id':'A','key':'sensor','text':'SAR',
            'quote':'SAR','expect':{'text':'RGB','status':'active','applies_to':None}}]},
        {'source_id':'r3','operations':[{'kind':'set_condition','task_id':'A','key':'sensor','text':'RGB',
            'quote':'RGB','expect':{'text':'SAR','status':'active','applies_to':None}}]}]}
    result=apply_request_batch(state,sources,proposal)
    assert result['tasks']['A']['conditions']['sensor']['text']=='RGB'
    assert apply_request_batch(result,sources,proposal)==result
    assert len(result['events'])==3


def test_old_reference_and_content_expectation_cannot_be_mixed():
    state=initial()
    with pytest.raises(ValueError,match='either expect or replaces'):
        apply(state,'r2','SAR',[{'kind':'set_condition','task_id':'A','key':'sensor','text':'SAR',
            'quote':'SAR','expect':None,'replaces':state['tasks']['A']['conditions']['sensor']['id']}])


def test_existing_tasks_without_focus_are_not_a_ready_empty_context():
    from paperpilot_agentic.request_state import request_work_view,empty_request_state
    state=apply(empty_request_state(),'r1','任务A',[{'kind':'create_task','task_id':'A',
        'label':'研究','quote':'任务A'}])
    assert request_work_view(state)['status']=='NEEDS_TASK_SELECTION'
    assert request_work_view(state)['view'] is None
    assert request_work_view(empty_request_state())['status']=='READY'
