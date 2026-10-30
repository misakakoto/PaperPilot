import json
import pytest
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget
from paperpilot_agentic.research_memory import advance_memory,parent_memory


def new(db,question,conversation='c'):
    rid = db.create_run(question,Budget(),metadata={'conversation_id':conversation})
    return db.get_run(rid)


def test_root_and_explicit_pinned_requests_survive_many_followups(tmp_path):
    db = RuntimeDatabase(tmp_path/'state.sqlite3')
    root = new(db,'比较A和B，只讨论相同数据集上的实验')
    db.update_run_metadata(root['run_id'],{'reader_task_memory':advance_memory(root,None,continue_task=False,scope={'sha'})})
    parent = db.get_run(root['run_id'])
    pin = None
    for i in range(15):
        question = '请记住：后续都只看消融实验' if i==2 else f'继续比较第{i}个设置'
        row = new(db,question)
        if i==2:
            pin = row['run_id']
        state = advance_memory(row,parent_memory(db,parent),continue_task=True,scope={'sha'})
        db.update_run_metadata(row['run_id'],{'reader_task_memory':state})
        parent = db.get_run(row['run_id'])
    restored = parent_memory(RuntimeDatabase(tmp_path/'state.sqlite3'),parent)
    assert restored['root_run_id']==root['run_id']
    assert pin in [r['run_id'] for r in restored['requests']]
    assert len(restored['recent_run_ids'])==4
    assert restored['working_bytes']<=24000
    assert all('answer' not in r for r in restored['requests'])


def test_new_subject_resets_old_constraints(tmp_path):
    db = RuntimeDatabase(tmp_path/'state.sqlite3')
    old = new(db,'只分析医学论文')
    current = new(db,'改为研究机器人控制')
    state = advance_memory(current,parent_memory(db,old),continue_task=False,scope={'robot'})
    assert state['root_run_id']==current['run_id'] and state['pinned_run_ids']==[]
    assert state['resolved_paper_sha256s']==['robot']


def test_explicit_correction_is_not_forgotten_after_recent_window(tmp_path):
    db = RuntimeDatabase(tmp_path/'state.sqlite3')
    parent = new(db,'请记住：只比较F1')
    correction = None
    for i in range(9):
        row = new(db,'改为比较准确率，不再沿用F1' if i==0 else '继续分析')
        if i==0:
            correction = row['run_id']
        state = advance_memory(row,parent_memory(db,parent),continue_task=True,scope={'sha'})
        db.update_run_metadata(row['run_id'],{'reader_task_memory':state})
        parent = db.get_run(row['run_id'])
    history = parent_memory(db,parent)
    assert correction in history['pinned_run_ids']
    assert any('改为比较准确率' in r['user_request'] for r in history['requests'])


def test_independent_pins_do_not_evict_an_earlier_metric_correction(tmp_path):
    db = RuntimeDatabase(tmp_path/'state.sqlite3')
    parent = new(db, '比较A和B，只比较F1')
    updates = ['改为比较准确率，不再沿用F1',
               '请记住：仅分析相同数据划分', '请记住：记录训练轮数',
               '请记住：记录参数量', '请记住：记录输入分辨率',
               '请记住：说明是否预训练', '请记住：区分监督条件',
               '请记住：保留方法的局限性']
    for question in updates + ['继续分析'] * 6:
        row = new(db, question)
        state = advance_memory(row, parent_memory(db, parent), continue_task=True, scope={'sha'})
        db.update_run_metadata(row['run_id'], {'reader_task_memory': state})
        parent = db.get_run(row['run_id'])
    history = parent_memory(db, parent)
    retained = [row['user_request'] for row in history['requests']]
    assert all(update in retained for update in updates)
    assert history['working_bytes'] <= 24000


def test_cross_conversation_memory_is_rejected(tmp_path):
    db = RuntimeDatabase(tmp_path/'state.sqlite3')
    outside = new(db,'secret','outside')
    parent = new(db,'normal')
    db.update_run_metadata(parent['run_id'],{'reader_task_memory':{'root_run_id':outside['run_id']}})
    with pytest.raises(ValueError,match='outside conversation'):
        parent_memory(db,db.get_run(parent['run_id']))


def test_resolver_receives_user_requests_not_model_claims():
    from paperpilot_agentic.follow_up_reader import resolve_follow_up
    from paperpilot_agentic.providers import LLMResponse
    class Model:
        def complete(self,messages,**kwargs):
            payload = json.loads(messages[1]['content'])
            assert payload['history']['task_requests'][0]['user_request']=='只比较相同数据集'
            assert 'MODEL INVENTED FACT' not in str(payload)
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps({'question':'比较相同数据集上的结果',
                'use_prior_sources':True,'needs_clarification':False,'reason':'initial requirement'}),finish_reason='stop')
    resolve_follow_up('按最开始的要求继续',{'prior_answer':'MODEL INVENTED FACT',
        'task_history':{'requests':[{'run_id':'r1','user_request':'只比较相同数据集'}],'omitted_run_ids':[]}},Model())


@pytest.mark.parametrize('continue_task', [True, False])
def test_incomplete_requirements_cannot_silently_drive_continued_research(continue_task):
    from paperpilot_agentic.follow_up_reader import resolve_follow_up
    from paperpilot_agentic.providers import LLMResponse

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps({
                'question': '继续比较F1' if continue_task else '研究机器人控制',
                'use_prior_sources': continue_task, 'needs_clarification': False,
                'reason': 'resolved subject'}), finish_reason='stop')

    value = resolve_follow_up('继续' if continue_task else '现在换个无关主题', {
        'task_history': {'requests': [], 'omitted_run_ids': ['correction'],
                         'requirements_complete': False}}, Model())
    assert value['needs_clarification'] is continue_task


def test_request_window_reports_unavailable_explicit_requirements(tmp_path):
    db = RuntimeDatabase(tmp_path/'state.sqlite3')
    parent = new(db, '只比较F1')
    for index in range(9):
        row = new(db, f'请记住第{index}项约束：' + '资料' * 900)
        state = advance_memory(row, parent_memory(db, parent), continue_task=True, scope={'sha'})
        db.update_run_metadata(row['run_id'], {'reader_task_memory': state})
        parent = db.get_run(row['run_id'])
    history = parent_memory(db, parent)
    assert not history['requirements_complete']
    assert history['omitted_run_ids']
    assert history['working_bytes'] <= 24000


def test_compact_history_preserves_original_requests_and_external_provenance():
    from paperpilot_agentic.follow_up_reader import project_request_history
    rows = [{'run_id': 'r1', 'created_at': '2026-09-12T10:00:00Z', 'role': 'initial_request',
             'user_request': '只比较F1'},
            {'run_id': 'r2', 'created_at': '2026-09-13T10:00:00Z', 'role': 'pinned_request',
             'user_request': '改为准确率，不再比较F1'}]
    view, mapping = project_request_history('按更新后的指标继续', {'requests': rows, 'omitted_run_ids': []})
    assert [r['user_request'] for r in view['task_requests']] == [r['user_request'] for r in rows]
    assert all('run_id' not in r and 'created_at' not in r for r in view['task_requests'])
    assert [r['run_id'] for r in mapping['references']] == ['r1', 'r2']
    assert rows[0]['created_at'] == '2026-09-12T10:00:00Z'


def test_temporal_reference_keeps_dates_in_compact_history():
    from paperpilot_agentic.follow_up_reader import project_request_history
    history = {'requests': [{'run_id': 'r1', 'created_at': '2026-09-12T10:00:00Z',
                             'user_request': '只比较F1'}], 'omitted_run_ids': []}
    view, _ = project_request_history('恢复9月12日的要求', history)
    assert view['task_requests'][0]['created_at'] == '2026-09-12T10:00:00Z'


@pytest.mark.parametrize('requirement', [
    '实验只使用时间上互不重叠的训练和测试影像',
    '每个方案的显存必须低于八GB',
    '请用英文回答',
    '继续分析，但输入必须是单通道SAR',
])
def test_unrecognized_requirements_survive_neutral_followups(tmp_path, requirement):
    db = RuntimeDatabase(tmp_path/'memory.sqlite3')
    parent = new(db, '设计变化检测实验')
    for text in [requirement] + ['继续分析'] * 12:
        row = new(db, text)
        state = advance_memory(row, parent_memory(db,parent), continue_task=True, scope={'s'})
        db.update_run_metadata(row['run_id'], {'reader_task_memory':state})
        parent = db.get_run(row['run_id'])
    history = parent_memory(db,parent)
    assert requirement in [r['user_request'] for r in history['requests']]
    assert history['requirements_complete']


def test_old_lossy_snapshot_recovers_requests_from_owned_parent_chain(tmp_path):
    db = RuntimeDatabase(tmp_path/'memory.sqlite3')
    root = new(db, '设计实验')
    parent = root
    condition = '输入必须是单通道SAR'
    for text in [condition] + ['继续分析'] * 6:
        row = new(db, text)
        db.update_run_metadata(row['run_id'], {'parent_run_id':parent['run_id'],
            'reader_task_memory':{'root_run_id':root['run_id'], 'pinned_run_ids':[],
                'recent_run_ids':[row['run_id']], 'retention_policy':'explicit-update-ledger-v2'}})
        parent = db.get_run(row['run_id'])
    history = parent_memory(db,parent)
    assert history['ancestry_complete']
    assert condition in [r['user_request'] for r in history['requests']]


def test_unrecoverable_old_snapshot_is_not_called_complete(tmp_path):
    db = RuntimeDatabase(tmp_path/'memory.sqlite3')
    root, parent = new(db, '设计实验'), new(db, '继续')
    db.update_run_metadata(parent['run_id'], {'reader_task_memory':{
        'root_run_id':root['run_id'], 'retention_policy':'explicit-update-ledger-v2'}})
    history = parent_memory(db,db.get_run(parent['run_id']))
    assert not history['requirements_complete']
