"""Pressure-triggered request memory for the default text Reader.

Only requirements are compiled; paper evidence and experiment receipts retain
their separate stores. The surrounding Reader checkpoint owns run replay.
"""
import json
import hashlib

from .durable_reader import DurableResearchModel, _reader_checkpoint
from .request_state import request_work_view
from .request_state_compiler import compile_request_batch
from .request_state_store import RequestStateStore


def prepare_reader_request_memory(runtime, run, parent, *, trigger_bytes=12000):
    """Checkpoint the complete memory stage, including no-op and failure results.

    Text and notebook callers share this stage. A crash during a paid call cannot
    be treated as a free retry; STARTED is rejected for explicit reconciliation.
    Completed preparation reuses the exact view even after later state commits.
    """
    fingerprint = hashlib.sha256(json.dumps({
        'version': 'request-memory-entry-v3-history-search', 'question': run['question'],
        'parent_id': parent['run_id'], 'parent_question': parent['question'],
        'conversation': run['metadata'].get('conversation_id'),
        'memory_root': parent['metadata'].get('request_state_root'),
        'raw_memory': parent['metadata'].get('reader_task_memory'),
        'trigger_bytes': trigger_bytes,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    key = 'request_memory_checkpoint'
    cached = _reader_checkpoint(runtime.db, run['run_id'], fingerprint, key=key)
    if cached is not None:
        return cached['result']
    result = _prepare_reader_request_memory(runtime, run, parent, trigger_bytes=trigger_bytes)
    _reader_checkpoint(runtime.db, run['run_id'], fingerprint, key=key, report={'result': result})
    return result


def _prepare_reader_request_memory(runtime, run, parent, *, trigger_bytes):
    """Return None for short raw history, otherwise a READY or failure report.

    Reconstruct the owned root-to-current path, including requests absent from an
    old working window. Never pass a source fragment as a complete original.
    All compression calls consume the CURRENT run's shared budget. A validated
    intermediate batch may persist if later work fails; it is not published as
    complete memory and a future child can continue from that committed prefix.
    """
    db = runtime.db
    prior_meta = parent['metadata']
    root = prior_meta.get('request_state_root') or (
        prior_meta.get('reader_task_memory') or {}).get('root_run_id', parent['run_id'])
    conversation = run['metadata'].get('conversation_id')
    if not isinstance(conversation, str) or not conversation:
        raise ValueError('Request memory needs a conversation')
    rows, seen, cursor = [], set(), run
    while True:
        if cursor['run_id'] in seen or cursor['metadata'].get('conversation_id') != conversation:
            raise ValueError('Invalid request memory ancestry')
        seen.add(cursor['run_id'])
        rows.append({'id': cursor['run_id'], 'text': cursor['question']})
        if cursor['run_id'] == root:
            break
        previous = cursor['metadata'].get('parent_run_id')
        if not previous:
            raise ValueError('Incomplete request memory ancestry')
        cursor = db.get_run(previous)
    rows.reverse()
    size = len(json.dumps(rows, ensure_ascii=False).encode('utf-8'))
    if not prior_meta.get('request_state_root') and size <= trigger_bytes:
        return None
    db.update_run_metadata(run['run_id'], {'request_state_root': root,
        'request_memory_status': 'COMPILING', 'request_memory_original_bytes': size})
    store = RequestStateStore(db)
    state = store.load(root, requester_id=run['run_id'])
    pending = [row for row in rows if row['id'] not in state['sources']]
    # A committed sibling or later branch must not be mistaken for this path.
    if any(sid not in seen for sid in state['sources']):
        raise ValueError('Request state belongs to a different branch')
    model = DurableResearchModel(runtime.llm_runtime, run['run_id'], stage='request_memory')
    batches = 0
    while pending:
        batch, used = [], 0
        for row in pending[:32]:
            required = len(json.dumps(row, ensure_ascii=False).encode('utf-8'))
            if batch and used + required > 18000:
                break
            batch.append(row)
            used += required
        def progress(report):
            db.update_run_metadata(run['run_id'], {'request_memory_progress': report})
        def reload(request):
            return store.history_page(root, requester_id=run['run_id'], task_ids=request['task_ids'],
                expected_revision=state['revision'], offset=request.get('offset', 0),
                query=request.get('query'), max_items=4 if request.get('query') else 32, max_bytes=12000)
        report = compile_request_batch(state, batch, model, on_progress=progress, history_loader=reload,
                                       allow_work_batches=True)
        batches += 1
        if report['status'] != 'MODEL_ACCEPTED_UNVERIFIED':
            db.update_run_metadata(run['run_id'], {'request_memory_status': report['status']})
            return {'status': report['status'], 'root_run_id': root,
                    'error_type': report.get('error_type'), 'batches': batches}
        if db.get_run(run['run_id'])['status'] == 'CANCELLED':
            raise InterruptedError('Run cancelled before request-memory commit')
        state = store.commit_batch(root, source_run_ids=[row['id'] for row in batch], proposal=report['proposal'])
        pending = pending[len(batch):]
    work = request_work_view(state)
    if work.get('work_plan', {}).get('status') == 'NEEDS_WORK_BATCHES':
        work = {**work, 'status': 'NEEDS_WORK_BATCHES'}
        db.update_run_metadata(run['run_id'], {'workset_review_available': True,
            'workset_review_endpoint': f"/agentic/runs/{run['run_id']}/workset-review"})
    db.update_run_metadata(run['run_id'], {'request_memory_status': work['status'],
        'request_memory_revision': state['revision'], 'request_memory_view': work,
        'request_memory_batches': batches})
    return {'status': work['status'], 'root_run_id': root, 'work': work,
            'active_task_id': state['active_task_id'], 'batches': batches}
