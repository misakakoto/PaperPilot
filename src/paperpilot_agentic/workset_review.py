"""Revision-bound, resumable READ-ONLY review of request worksets.

All pages share one run budget. Completed pages are replayed; an acquired but
unfinished page is never stolen or automatically re-billed. This workflow cannot
execute experiments or mark the underlying user activities completed.
"""
import hashlib
import json
from datetime import datetime, timezone
from itertools import islice

from .durable_reader import (DurableResearchModel, ReaderAlreadyStarted, _reader_checkpoint,
                            persist_reader_evidence, publish_reader_answer, run_durable_reader)
from .llm_runtime import LLMBudgetExhausted
from .models import Budget, ResearchOutcome, RunStatus
from .reader_workset import source_aware_workset
from .request_state import request_work_view
from .request_state_store import RequestStateStore


def enqueue_review(runtime, parent_run_id, *, budget=None, synthesize=True):
    if type(synthesize) is not bool:
        raise ValueError('synthesize must be boolean')
    if not runtime.llm_runtime or not runtime.search_adapter or not runtime.text_reader_protocol:
        raise ValueError('Workset review requires a configured Reader and retriever')
    parent = runtime.db.get_run(parent_run_id)
    if parent['status'] not in {'COMPLETED', 'FAILED', 'BUDGET_EXHAUSTED'}:
        raise ValueError('Workset review requires a finished parent')
    root = parent['metadata'].get('request_state_root')
    if not root:
        raise ValueError('No committed request ledger is available')
    state = RequestStateStore(runtime.db).load(root, requester_id=parent_run_id, require_ancestor=True)
    work = request_work_view(state)
    if work['status'] == 'READY':
        pages = [work['view']]
    elif work.get('work_plan', {}).get('status') == 'NEEDS_WORK_BATCHES':
        pages = work['work_plan']['pages']
    else:
        raise ValueError('Requirements cannot be safely partitioned; narrow the task first')
    if not any(page['pending_actions'] for page in pages):
        raise ValueError('No pending activities to review')
    meta = parent['metadata']
    scope = meta.get('resolved_paper_sha256s')
    if scope is None:
        # Memory preparation can stop the parent before normal scope resolution.
        # Never turn that missing field into an unrestricted corpus search.
        from .request_scope import resolve_request_scope
        edits = resolve_request_scope(parent['question'],
            lambda text: runtime._resolve_explicit_source_scope(text, None, None))
        selected = set(meta.get('paper_sha256s') or [])
        if selected and set(edits['included']) - selected:
            raise ValueError('Named papers are outside the explicit selection')
        inherited = meta.get('paper_sha256s') or meta.get('conversation_context', {}).get('prior_paper_sha256s') or []
        scope = sorted((set(edits['included']) or set(inherited)) - set(edits['excluded']))
        if not scope:
            raise ValueError('Source scope is unresolved; select papers before workset review')
    notebook_root = meta.get('notebook_root_run_id')
    if notebook_root:
        notebook = runtime.db.get_run(notebook_root)
        if notebook['metadata'].get('conversation_id') != meta.get('conversation_id'):
            raise ValueError('Notebook belongs to a different conversation')
        inventory = set(notebook['metadata'].get('resolved_paper_sha256s') or [])
        if not scope or set(scope) - inventory:
            raise ValueError('Review must stay within the selected notebook')
    if (not isinstance(scope, list) or any(not isinstance(x, str) or not x for x in scope)
            or meta.get('source_scope_explicitly_empty')):
        raise ValueError('Invalid or explicitly empty source scope')
    manifest = {'version': 4, 'root': root, 'parent': parent_run_id, 'synthesize': synthesize,
                'cross_group_review': True,
                'group_large_evidence': True,
                'revision': state['revision'], 'pages': pages, 'paper_sha256s': scope,
                'protocol': runtime.text_reader_protocol}
    return runtime.enqueue(question='只读审阅当前研究活动的文献依据与实验设计；不执行实验。',
        budget=budget, metadata={'engine': 'workset_review', 'workset_manifest': manifest,
            'conversation_id': meta['conversation_id'], 'parent_run_id': parent_run_id,
            'request_state_root': root, 'resolved_paper_sha256s': scope,
            'capability_boundary': 'read-only literature review; no experiment execution'})


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resume_review(runtime, run_id, *, max_batches=None):
    """max_batches yields between pages; it never resets or expands the run budget."""
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError('max_batches must be a positive integer')
    db = runtime.db
    run = db.get_run(run_id)
    if run['status'] in {'COMPLETED', 'FAILED', 'BUDGET_EXHAUSTED', 'CANCELLED'}:
        return runtime.status(run_id)
    manifest = run['metadata']['workset_manifest']
    store = RequestStateStore(db)

    def guard():
        current = db.get_run(run_id)
        if current['status'] == 'CANCELLED':
            raise InterruptedError('Review cancelled')
        parent = db.get_run(manifest['parent'])
        if parent['metadata'].get('conversation_id') != current['metadata'].get('conversation_id'):
            raise ValueError('Review is outside its source conversation')
        state = store.load(manifest['root'], requester_id=manifest['parent'], require_ancestor=True)
        if state['revision'] != manifest['revision']:
            raise ValueError('Request revision changed; create a new review')
        limits = Budget.from_dict(current['budget'])
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(
            current['created_at'].replace('Z', '+00:00'))).total_seconds()
        if elapsed >= limits.max_seconds:
            raise LLMBudgetExhausted('Review time budget exhausted')
        return current, limits

    completed = []
    spent = 0

    def publish_partial():
        answer = '\n\n'.join(f"### 只读审阅 {i + 1}/{len(manifest['pages'])}\n{r.get('answer') or '本批未获得可支持的回答。'}"
                               for i, r in enumerate(completed))
        supported = any(r.get('answer') and r.get('direct_answer_supported') is not False for r in completed)
        report = {'status': 'REVIEW_REQUIRED', 'answer': answer,
            'direct_answer_supported': supported, 'answer_scope': 'PARTIAL',
            'limitations': [f"已审阅 {len(completed)}/{len(manifest['pages'])} 批；文献审阅不代表研究目标完成，未执行任何实验。"] +
                [text for r in completed for text in r.get('limitations', [])],
            'unverified_limitations': [text for r in completed for text in r.get('unverified_limitations', [])]}
        publish_reader_answer(db, run_id, report)
        db.update_run_outcome(run_id, ResearchOutcome.PARTIAL if supported else ResearchOutcome.UNRESOLVED)

    try:
        if not runtime.llm_runtime or not runtime.search_adapter or not runtime.text_reader_protocol:
            raise ValueError('Workset review requires a configured Reader and retriever')
        if manifest['protocol'] != runtime.text_reader_protocol:
            raise ValueError('Reader protocol changed; create a new review')
        guard()
        db.update_run_status(run_id, RunStatus.RUNNING)
        for index, page in enumerate(manifest['pages']):
            current, limits = guard()
            key = f'workset_page_{index}'
            fingerprint = _fingerprint({'manifest': manifest, 'index': index})
            cached = current['metadata'].get(key)
            if cached is None:
                if max_batches is not None and spent >= max_batches:
                    break
                if current['used_tool_calls'] >= limits.max_tool_calls:
                    raise LLMBudgetExhausted('Review tool budget exhausted')
                if current['used_steps'] >= limits.max_steps:
                    raise LLMBudgetExhausted('Review step budget exhausted')
                if current['used_llm_calls'] + 2 > limits.max_llm_calls:
                    raise LLMBudgetExhausted('Insufficient remaining calls for Writer and Auditor')
            report = _reader_checkpoint(db, run_id, fingerprint, key=key)
            if report is None:
                # Claim precedes retrieval/accounting: concurrent workers cannot
                # execute the same page. Never keep the SQL lock across I/O.
                guard()
                db.increment_usage(run_id, steps=1, tool_calls=1)
                question = ('仅针对以下活动进行只读文献分析或实验方案审阅。'
                            '不得声称训练、实验或业务操作已执行。保留所有全局约束；'
                            '没有原文依据的结论须明确标为待验证。\n'
                            + json.dumps(page, ensure_ascii=False))
                query = page['label'] + ' ' + ' '.join(
                    item['text'] for item in page['pending_actions'].values())
                scope = manifest['paper_sha256s']
                hits = list(islice(runtime.search_adapter.search(query, top_k=8,
                                  paper_sha256s=scope or None), 8))
                guard()
                sources = []
                for hit in hits:
                    if scope and hit.get('paper_sha256') not in scope:
                        raise ValueError('Retrieval crossed frozen paper scope')
                    text = hit.get('content') or hit.get('text')
                    sid = hit.get('stable_id')
                    if not text or not sid:
                        continue
                    sources.append({'evidence_id': sid, 'content': text,
                        'source_uri': hit.get('source_uri') or f"paper://{hit.get('paper_sha256')}/{sid}",
                        'page': hit.get('page'),
                        'provenance': {'paper_sha256': hit.get('paper_sha256'), 'stable_id': sid}})
                sources = persist_reader_evidence(db, run_id, sources)
                selected, stats = source_aware_workset(sources, max_bytes=24000)
                report = run_durable_reader(db, run_id, selected,
                    DurableResearchModel(runtime.llm_runtime, run_id, stage='workset_review'),
                    protocol_id=manifest['protocol'], question=question, finalize=False,
                    checkpoint_key=f'workset_answer_{index}')
                report = {**report, 'workset_context': stats}
                _reader_checkpoint(db, run_id, fingerprint, key=key, report=report)
                spent += 1
                db.emit_event(run_id, 'workset.page_reviewed', {'page': index + 1,
                    'total_pages': len(manifest['pages']), 'status': report.get('status')})
            completed.append(report)
            db.update_run_metadata(run_id, {'workset_progress': {
                'reviewed_pages': len(completed), 'total_pages': len(manifest['pages']),
                'revision': manifest['revision'], 'activities_executed': 0}})
            if report.get('error_type') == 'LLMBudgetExhausted':
                raise LLMBudgetExhausted('Page exhausted the shared model budget')
            if report.get('status') in {'FAILED', 'ROUTING_FAILED'}:
                raise RuntimeError('A review page failed; saved result requires inspection')
        guard()
        if len(completed) == len(manifest['pages']):
            publish_partial()
            if manifest.get('synthesize', False):
                from .workset_synthesis import synthesize_worksets
                synthesis = synthesize_worksets(runtime, run_id, manifest, completed, guard=guard)
                db.update_run_metadata(run_id, {'workset_synthesis': synthesis})
                if synthesis['status'] in {'REVIEWED', 'REVIEWED_GROUPS'}:
                    report = synthesis['report']
                    publish_reader_answer(db, run_id, {**report, 'answer_scope': 'PARTIAL',
                        'limitations': ['跨批综合仅基于已取回原文，未经独立事实评测；未执行任何实验。'] +
                                       report.get('limitations', [])})
                    db.update_run_outcome(run_id, ResearchOutcome.PARTIAL)
                else:
                    # Keep the useful per-page report and disclose the missing stage.
                    final = db.get_run(run_id)['metadata']['final_answer']
                    final['limitations'].append('未发布跨批综合结论：' + synthesis['status'])
                    db.update_run_metadata(run_id, {'final_answer': final})
            db.update_run_status(run_id, RunStatus.COMPLETED)
    except ReaderAlreadyStarted:
        db.emit_event(run_id, 'workset.review_blocked', {'reason': 'Page active or interrupted; no automatic replay'})
    except InterruptedError:
        if db.get_run(run_id)['status'] != 'CANCELLED':
            raise
    except Exception as exc:
        if db.get_run(run_id)['status'] != 'CANCELLED':
            db.update_run_outcome(run_id, ResearchOutcome.PARTIAL if completed else ResearchOutcome.UNRESOLVED)
            if completed:
                publish_partial()
            db.update_run_status(run_id, RunStatus.BUDGET_EXHAUSTED if isinstance(exc, LLMBudgetExhausted)
                                 else RunStatus.FAILED, error=str(exc))
    return runtime.status(run_id)
