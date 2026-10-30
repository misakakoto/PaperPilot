import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget
from paperpilot_agentic.request_state_store import RequestStateStore


def run(db,text,*,parent=None,conversation='owned'):
    return db.create_run(text,Budget(),metadata={'conversation_id':conversation,'parent_run_id':parent})


def setup(tmp_path):
    db=RuntimeDatabase(tmp_path/'state.sqlite3')
    root=run(db,'设计实验，不进行训练')
    store=RequestStateStore(db)
    delta={'expected_revision':'initial','operations':[
        {'kind':'create_task','task_id':'A','label':'研究','quote':'设计实验'},
        {'kind':'activate_task','task_id':'A','quote':'设计实验'},
        {'kind':'add_todo','task_id':'A','key':'design','text':'设计实验','quote':'设计实验'}]}
    state=store.commit(root,source_run_id=root,delta=delta)
    return db,root,store,state,delta


def test_reopen_replay_and_authoritative_source_reload(tmp_path):
    db,root,store,state,delta=setup(tmp_path)
    reopened=RequestStateStore(RuntimeDatabase(db.path))
    assert reopened.load(root,requester_id=root)==state
    assert reopened.commit(root,source_run_id=root,delta=delta)==state
    assert reopened.reload_sources(root,requester_id=root,source_ids=[root,root])==[
        {'id':root,'text':'设计实验，不进行训练'}]
    assert len(state['events'])==1


@pytest.mark.parametrize('conversation,parent_owned', [('other',True),('owned',False),(None,True)])
def test_unowned_request_cannot_read_commit_or_reload(tmp_path,conversation,parent_owned):
    db,root,store,state,_=setup(tmp_path)
    outsider=run(db,'private',parent=root if parent_owned else None,conversation=conversation)
    with pytest.raises(ValueError): store.load(root,requester_id=outsider)
    with pytest.raises(ValueError): store.commit(root,source_run_id=outsider,
        delta={'expected_revision':state['revision'],'operations':[]})
    with pytest.raises(ValueError): store.reload_sources(root,requester_id=outsider,source_ids=[root])
    assert store.load(root,requester_id=root)==state


def test_quote_from_another_request_cannot_be_used_as_current_source(tmp_path):
    db,root,store,state,_=setup(tmp_path)
    child=run(db,'继续',parent=root)
    with pytest.raises(ValueError,match='current user request'):
        store.commit(root,source_run_id=child,delta={'expected_revision':state['revision'],'operations':[
            {'kind':'add_todo','task_id':'A','key':'extra','text':'设计实验','quote':'设计实验'}]})
    assert store.load(root,requester_id=root)==state
    with pytest.raises(ValueError,match='not in this request ledger'):
        store.reload_sources(root,requester_id=child,source_ids=[child])


def test_parallel_stale_proposals_do_not_overwrite_each_other(tmp_path):
    db,root,store,state,_=setup(tmp_path)
    ids=[run(db,f'要求{i}',parent=root) for i in range(2)]
    stores=[RequestStateStore(RuntimeDatabase(db.path)) for _ in ids]
    barrier=Barrier(2)
    def write(index):
        barrier.wait(timeout=10)
        try:
            stores[index].commit(root,source_run_id=ids[index],delta={
                'expected_revision':state['revision'],'operations':[
                    {'kind':'set_condition','task_id':'A','key':'limit','text':f'要求{index}',
                     'quote':f'要求{index}','replaces':None}]})
            return 'committed'
        except ValueError as exc:
            assert 'Stale' in str(exc)
            return 'stale'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(write,range(2)))
    assert sorted(results)==['committed','stale']
    saved=store.load(root,requester_id=root)
    assert len(saved['events'])==2
    assert sum(sid in saved['sources'] for sid in ids)==1


def test_database_failure_keeps_previous_ledger_and_revision(tmp_path):
    db,root,store,state,_=setup(tmp_path)
    child=run(db,'继续',parent=root)
    with db.transaction() as conn:
        conn.execute("CREATE TRIGGER fail_request_state BEFORE UPDATE ON research_request_states BEGIN SELECT RAISE(ABORT,'injected write failure'); END")
    with pytest.raises(sqlite3.IntegrityError,match='injected'):
        store.commit(root,source_run_id=child,delta={'expected_revision':state['revision'],'operations':[]})
    assert RequestStateStore(RuntimeDatabase(db.path)).load(root,requester_id=root)==state


def test_changed_original_is_detected_on_reload(tmp_path):
    db,root,store,_,_=setup(tmp_path)
    with db.transaction() as conn:
        conn.execute('UPDATE runs SET question=? WHERE run_id=?',('tampered test fixture',root))
    with pytest.raises(ValueError,match='changed after state commit'):
        store.reload_sources(root,requester_id=root,source_ids=[root])
