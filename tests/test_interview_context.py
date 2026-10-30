"""Storage/retraction oracle; deterministic edits do not measure model quality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.interview.context import request_trace, comparison_record  # noqa: E402
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget
from paperpilot_agentic.request_state_store import RequestStateStore


def test_long_trace_preserves_retractions_and_originals_beyond_sliding_window(tmp_path, record_property):
    texts = request_trace()
    db = RuntimeDatabase(tmp_path/'context.sqlite3')
    store = RequestStateStore(db)
    parent = root = None
    for index, text in enumerate(texts):
        rid = db.create_run(text, Budget(), metadata={'conversation_id': 'owned', 'parent_run_id': parent})
        root = root or rid
        state = store.load(root, requester_id=rid)
        ops = []
        if index == 0:
            ops = [
                {'kind': 'create_task', 'task_id': 'research', 'label': '消融实验对比', 'quote': '对比两篇论文的消融实验'},
                {'kind': 'activate_task', 'task_id': 'research', 'quote': '当前研究目标'},
            ]
            for key, quote in [('scope', '论文范围为 ChangeMamba 和 FlowMamba'),
                               ('short', '回答要简洁'), ('memory', '实验显存预算为 8GB'),
                               ('execution', '只进行阅读和设计，不启动训练')]:
                ops.append({'kind': 'set_condition', 'task_id': 'research', 'key': key,
                            'text': quote, 'quote': quote, 'replaces': None})
        if index == 1:
            ops = [
                {'kind': 'rename_task', 'task_id': 'research', 'previous_label': '消融实验对比',
                 'label': '理解 token 组织机制', 'quote': '切换目标为理解 ChangeMamba 的 token 组织机制'},
                {'kind': 'retract_condition', 'task_id': 'research', 'key': 'short',
                 'replaces': state['tasks']['research']['conditions']['short']['id'], 'quote': '撤销回答要简洁的要求'},
                {'kind': 'set_condition', 'task_id': 'research', 'key': 'scope',
                 'replaces': state['tasks']['research']['conditions']['scope']['id'],
                 'text': '排除 FlowMamba', 'quote': '排除 FlowMamba'},
            ]
        state = store.commit(root, source_run_id=rid, delta={'expected_revision': state['revision'], 'operations': ops})
        parent = rid
    reopened = RequestStateStore(RuntimeDatabase(db.path))
    restored = reopened.load(root, requester_id=rid)
    assert restored == state
    task = state['tasks']['research']
    assert task['label'] == '理解 token 组织机制' and 'short' not in task['conditions']
    assert task['conditions']['scope']['text'] == '排除 FlowMamba'
    assert task['conditions']['memory']['text'] == '实验显存预算为 8GB'
    assert task['conditions']['execution']['text'] == '只进行阅读和设计，不启动训练'
    originals = reopened.reload_sources(root, requester_id=rid, source_ids=list(state['sources']))
    assert [s['text'] for s in originals] == texts
    page = reopened.history_page(root, requester_id=rid, task_ids=['research'],
                                expected_revision=state['revision'], query='8GB')
    assert texts[0] in [s['text'] for s in page['sources']]
    metrics = comparison_record(texts, state)
    assert metrics['raw_history_bytes'] > 12000
    assert metrics['derived_work_bytes'] < metrics['sliding_last_6_bytes']
    assert '8GB' not in str(texts[-6:]) and '排除 FlowMamba' not in str(texts[-6:])
    assert metrics['sources_preserved'] == len(texts)
    for key in ('raw_history_bytes', 'sliding_last_6_bytes', 'derived_work_bytes', 'sources_preserved'):
        record_property(key, metrics[key])
