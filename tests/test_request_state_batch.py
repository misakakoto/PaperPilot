from copy import deepcopy
import pytest
from paperpilot_agentic.request_state import empty_request_state,apply_request_batch
from paperpilot_agentic.request_state_store import RequestStateStore
from test_request_state_store import setup,run


def proposal(state,ids):
    return {'base_revision':state['revision'],'frames':[
        {'source_id':ids[0],'operations':[{'kind':'set_condition','task_id':'A','key':'limit',
            'text':'4GB','quote':'4GB','replaces':None}]},
        {'source_id':ids[1],'operations':[{'kind':'set_condition','task_id':'A','key':'limit',
            'text':'3GB','quote':'3GB','replaces':{'frame':ids[0],'operation':0}}]}]}


def test_batch_local_references_commit_and_replay(tmp_path):
    db,root,store,state,_=setup(tmp_path)
    first=run(db,'4GB',parent=root)
    second=run(db,'3GB',parent=first)
    p=proposal(state,[first,second])
    result=store.commit_batch(root,source_run_ids=[first,second],proposal=p)
    assert result['tasks']['A']['conditions']['limit']['text']=='3GB'
    assert result['tasks']['A']['archive'][-1]['text']=='4GB'
    assert len(result['events'])==3
    reopened=RequestStateStore(db)
    assert reopened.commit_batch(root,source_run_ids=[first,second],proposal=p)==result


@pytest.mark.parametrize('failure',['missing','reordered','future_ref','bad_quote','sibling'])
def test_invalid_batch_never_saves_earlier_frames(tmp_path,failure):
    db,root,store,state,_=setup(tmp_path)
    first=run(db,'4GB',parent=root)
    second=run(db,'3GB',parent=root if failure=='sibling' else first)
    ids=[first,second]
    p=proposal(state,ids)
    if failure=='missing': p['frames'].pop()
    if failure=='reordered': p['frames'].reverse()
    if failure=='future_ref': p['frames'][1]['operations'][0]['replaces']={'frame':second,'operation':0}
    if failure=='bad_quote': p['frames'][1]['operations'][0]['quote']='invented'
    with pytest.raises(ValueError): store.commit_batch(root,source_run_ids=ids,proposal=p)
    assert store.load(root,requester_id=root)==state


def test_explicit_noop_frame_is_required_for_coverage(tmp_path):
    db,root,store,state,_=setup(tmp_path)
    child=run(db,'继续',parent=root)
    p={'base_revision':state['revision'],'frames':[{'source_id':child,'operations':[]}]}
    result=store.commit_batch(root,source_run_ids=[child],proposal=p)
    assert child in result['sources']
    assert result['tasks']==state['tasks']
    assert result['events'][-1]['operations']==[]


def test_fresh_revision_cannot_append_an_older_or_sibling_branch(tmp_path):
    db,root,store,state,_=setup(tmp_path)
    first=run(db,'4GB',parent=root)
    second=run(db,'3GB',parent=first)
    saved=store.commit_batch(root,source_run_ids=[first,second],proposal=proposal(state,[first,second]))
    sibling=run(db,'其他分支',parent=first)
    with pytest.raises(ValueError,match='outside root ancestry'):
        store.commit(root,source_run_id=sibling,delta={'expected_revision':saved['revision'],'operations':[]})
    assert store.load(root,requester_id=root)==saved
