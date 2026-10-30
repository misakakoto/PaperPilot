"""Source research entry with an opt-in, explicitly targeted table bridge.

Not a replacement for visual research or experiments.
Interrupted provider work is never automatically replayed.
"""
import hashlib
import json
import re
from datetime import datetime, timezone

from .durable_reader import (
    DurableResearchModel, _reader_checkpoint, publish_reader_answer,
    reader_completion_state, run_durable_reader, persist_reader_evidence,
)
from .llm_runtime import LLMBudgetExhausted
from .models import Budget, RunStatus
from .section_router import read_outline_fallback, read_selected_sections, select_sections, structural_overviews
from .reader_workset import source_aware_workset, revision_workset
from .reader_recovery import (plan_evidence_reload, revision_feedback, citation_revision_plan,
                              boundary_revision_plan)
from .request_scope import (resolve_request_scope, automatic_table_targets,
                            visual_policy, prioritize_requested_evidence,
                            numeric_lookup_query, table_target_query)


def resume_text_reader(runtime, run_id, *, protocol_id):
    db = runtime.db
    run = db.get_run(run_id)
    metadata = run['metadata']
    from .structured_retrieval import select_retriever
    selected_adapter, retrieval_identity = select_retriever(runtime, metadata)
    scope = metadata.get('paper_sha256s') or []
    question = run['question']
    fingerprint = hashlib.sha256(json.dumps({
        'question': run['question'], 'scope': scope, 'protocol': protocol_id,
        **({'retrieval_identity': retrieval_identity} if retrieval_identity != 'legacy' else {}),
        'parent_run_id': metadata.get('parent_run_id'),
        'visual_targets': metadata.get('reader_visual_targets'),
        'allow_visual': metadata.get('allow_visual', False),
        'visual_policy': metadata.get('visual_policy', 'explicit'),
        'disable_cross_run_visual_cache': metadata.get('disable_cross_run_visual_cache', False),
        **({'reader_memory_version':metadata['reader_memory_version']}
           if metadata.get('reader_memory_version') else {}),
        **({'evidence_selection':metadata['evidence_selection']}
           if metadata.get('evidence_selection','off') != 'off' else {}),
        **({'initial_evidence_read': {'policy': metadata['reader_evidence_read_policy'],
            'max_bytes': metadata.get('reader_evidence_read_max_bytes', 24000),
            'protocol': 'initial-whole-source-read-v1'}}
           if metadata.get('reader_evidence_read_policy', 'full') != 'full' else {}),
        'pipeline': ('text-navigation-reader-v36-object-precision-selection'
                     if metadata.get('evidence_selection','off') != 'off'
                      else 'text-navigation-reader-v36-object-precision'),
    }, sort_keys=True).encode()).hexdigest()
    saved = _reader_checkpoint(db, run_id, fingerprint, key='text_reader_pipeline')
    if saved is not None:
        return runtime.status(run_id)
    report = None
    stage = 'initialization'
    def finish(report):
        if db.get_run(run_id)['status'] != 'CANCELLED':
            publish_reader_answer(db, run_id, report)
            status, outcome = reader_completion_state(report)
            db.update_run_outcome(run_id, outcome)
            db.update_run_status(run_id, status, error=report.get('error_type'))
        _reader_checkpoint(db, run_id, fingerprint, report=report, key='text_reader_pipeline')
        return runtime.status(run_id)
    try:
        selection_mode = metadata.get('evidence_selection','off')
        if selection_mode not in {'off','select-v2'}:
            raise ValueError('Unknown evidence_selection mode')
        policy = visual_policy(metadata)
        targets = list(metadata.get('reader_visual_targets') or [])
        if not isinstance(scope, list) or any(not isinstance(s, str) or not s for s in scope):
            raise ValueError('paper_sha256s must be a list of source hashes')
        explicit_scope = set(scope)
        if targets and policy == 'explicit' and metadata.get('allow_visual') is not True:
            raise ValueError('Visual targets require explicit allow_visual opt-in')
        if policy == 'off':
            targets = []
        if runtime.llm_runtime is None or runtime.search_adapter is None:
            raise ValueError('Text reader requires a configured model and retriever')
        if metadata.get('initial_claims') or metadata.get('initial_edges'):
            raise ValueError('Text reader cannot execute supplied Claim or experiment plans')
        db.update_run_status(run_id, RunStatus.RUNNING)
        db.update_run_metadata(run_id, {'planning_state': 'BYPASSED_TEXT_READER',
            'capability_boundary': 'text retrieval and source audit; bounded visual tables when enabled; no experiments'})
        adapter = selected_adapter
        budget = Budget.from_dict(run['budget'])
        task_history = None
        compiled_memory = None
        continue_task = False
        prior_empty = False
        if metadata.get('parent_run_id'):
            from .follow_up_reader import resolve_follow_up, prior_source_scope
            parent = runtime.status(metadata['parent_run_id'], include_trace=False)
            if parent['run']['metadata'].get('conversation_id') != metadata.get('conversation_id'):
                raise ValueError('Follow-up parent is outside conversation')
            context = runtime._follow_up_context(parent)
            if metadata.get('research_state_context'):
                context['user_research_requirements_not_evidence'] = metadata['research_state_context']
            if metadata.get('reader_memory_version') == 'request-ledger-v1':
                from .reader_request_memory import prepare_reader_request_memory
                compiled_memory = prepare_reader_request_memory(runtime, run, parent['run'])
                if compiled_memory is not None:
                    if compiled_memory['status'] != 'READY':
                        return finish({'status': 'NO_EVIDENCE', 'answer': '',
                            'error_type': compiled_memory.get('error_type'),
                            'limitations': [('研究要求已保存，但本轮工作集过大。可使用“分批审阅研究活动”继续查阅文献；不会执行实验。'
                                if compiled_memory['status'] == 'NEEDS_WORK_BATCHES' else
                                '研究要求整理尚未完成，未使用残缺记忆继续回答。请缩小当前任务或明确本轮要求。')],
                            'clarification_required': True})
                    context['task_state'] = compiled_memory['work']['view']
                    # Avoid reintroducing an oversized original via prior_question.
                    if len(context.get('prior_question', '').encode('utf-8')) > 2000:
                        context['prior_question'] = ''
                from .research_memory import parent_memory
                if compiled_memory is None:
                    task_history = parent_memory(db,parent['run'])
                    context['task_history'] = task_history
            resolution = metadata.get('entry_resolution') or resolve_follow_up(question, context,
                DurableResearchModel(runtime.llm_runtime, run_id, stage='follow_up_resolution'))
            db.update_run_metadata(run_id, {'follow_up_resolution': resolution})
            if resolution['needs_clarification']:
                return finish({'status': 'NO_EVIDENCE', 'answer': '',
                    'limitations': [resolution['reason']], 'clarification_required': True})
            question = resolution['question']
            continue_task = resolution['use_prior_sources']
            prior_empty = continue_task and parent['run']['metadata'].get('source_scope_explicitly_empty', False)
            db.update_run_metadata(run_id, {'resolved_question': question})
            if resolution['use_prior_sources'] and not scope:
                scope = prior_source_scope(context)
            db.update_run_metadata(run_id, {'resolved_paper_sha256s': scope})

        def tool(name, operation):
            nonlocal stage
            stage = name
            current = db.get_run(run_id)
            if current['status'] == 'CANCELLED':
                raise InterruptedError('Run cancelled')
            if current['used_tool_calls'] >= budget.max_tool_calls:
                raise LLMBudgetExhausted('Tool budget exhausted')
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(
                current['created_at'].replace('Z', '+00:00'))).total_seconds()
            if elapsed >= budget.max_seconds:
                raise LLMBudgetExhausted('Time budget exhausted')
            db.increment_usage(run_id, tool_calls=1)
            db.emit_event(run_id, 'reader.tool', {'name': name})
            return operation()

        def resolve(text):
            return runtime._resolve_explicit_source_scope(text, None, None)
        request_scope = resolve_request_scope(question, resolve)
        original_scope = resolve_request_scope(run['question'], resolve)
        # Resolve explicit user edits independently of the model rewrite. An
        # inherited scope is a default, not permission to ignore a new title;
        # an explicit caller selection remains a hard restriction for this run.
        named = set(original_scope['included'])
        additive = bool(re.search(r'再(?:加入|加上)|加上|加入|恢复|\b(?:add|also include|restore)\b',
                                  run['question'], re.I))
        if named:
            hashes = (set(scope) | named) if additive and continue_task else named
        elif prior_empty and not explicit_scope:
            hashes = set()
        else:
            hashes = set(scope) or set(request_scope['included'])
        if explicit_scope:
            hashes &= explicit_scope
        scope_conflicts = named - explicit_scope if explicit_scope else set()
        hashes -= set(original_scope['excluded']) | set(request_scope['excluded'])
        explicitly_empty = not hashes and bool(prior_empty or original_scope['excluded'] or request_scope['excluded'])
        db.update_run_metadata(run_id, {'request_scope': request_scope,
            'original_request_scope': original_scope, 'visual_policy_effective': policy,
            'requested_sources_outside_selection': sorted(scope_conflicts),
            'source_scope_explicitly_empty': explicitly_empty,
            'resolved_paper_sha256s': sorted(hashes)})
        if scope_conflicts:
            return finish({'status': 'NO_EVIDENCE', 'answer': '', 'clarification_required': True,
                           'limitations': ['本轮点名的论文超出已选择的来源范围，请调整论文选择后继续。']})
        if not hashes and (original_scope['included'] or original_scope['excluded']
                           or request_scope['needs_clarification'] or explicitly_empty):
            return finish({'status': 'NO_EVIDENCE', 'answer': '', 'clarification_required': True,
                           'limitations': ['请明确要使用的论文；排除来源后无法确定检索范围。']})
        target_query = table_target_query(run['question'], question)
        requested_targets, target_gaps = automatic_table_targets(target_query, hashes, resolve)
        if target_gaps and policy == 'auto' and any(g['reason'] == 'ambiguous_table_owner' for g in target_gaps):
            return finish({'status': 'NO_EVIDENCE', 'answer': '', 'clarification_required': True,
                           'limitations': ['请明确每张表对应的论文，当前无法唯一确定表格来源。']})
        if policy == 'auto' and not targets:
            targets = requested_targets
        db.update_run_metadata(run_id, {'requested_table_targets': requested_targets,
                                       'table_target_gaps': target_gaps,
                                       'table_target_request_basis': 'owned_original_request',
                                       'rewrite_only_table_request_ignored': target_query != question})
        if targets:
            from .reader_visual import validate_visual_targets
            validate_visual_targets(targets, hashes)
        # Persist the effective scope after title resolution, not the pre-resolution
        # empty list. Later turns must see the sources actually requested here.
        db.update_run_metadata(run_id, {'resolved_paper_sha256s': sorted(hashes)})
        from .research_memory import advance_memory
        db.update_run_metadata(run_id,{'reader_task_memory':advance_memory(
            run,task_history,continue_task=continue_task and task_history is not None,scope=hashes)})
        # Retrieval rewrites may omit output requirements. Preserve the owned
        # task's original requests separately for generation and source audit.
        from .research_memory import RESET
        answer_history = []
        if compiled_memory is not None:
            answer_history = [{'role': 'derived_request_state', 'user_request': json.dumps(
                compiled_memory['work']['view'], ensure_ascii=False)}]
        elif continue_task and task_history and not RESET.search(run['question']):
            if not task_history.get('requirements_complete', True):
                raise ValueError('Incomplete task requirements cannot enter answer generation')
            answer_history = [{'user_request': row['user_request'], 'role': row['role']}
                              for row in task_history['requests']]
        if metadata.get('research_state_context',{}).get('goal'):
            answer_history.append({'role':'user_research_requirements_not_evidence',
                'user_request':json.dumps(metadata['research_state_context'],ensure_ascii=False)})
        db.update_run_metadata(run_id, {'answer_request_history': answer_history})
        hits = tool('paper.search', lambda: adapter.search(
            question, top_k=8, paper_sha256s=hashes or None))
        if selection_mode == 'select-v2':
            if len(hashes) != 1 or not callable(getattr(adapter,'evidence_candidates',None)):
                raise ValueError('Evidence selection requires one resolved paper and a compatible retriever')
            candidates = tool('paper.evidence_candidates',lambda: adapter.evidence_candidates(
                question,paper_sha256s=hashes))
            stage = 'evidence_selection'
            from .reader_evidence_selection import select_reader_evidence
            hits = select_reader_evidence(runtime,run_id,question,candidates,hits,hashes)
        value_query = numeric_lookup_query(question)
        if value_query:
            hits = list(hits) + list(tool('paper.search.numeric', lambda: adapter.search(
                value_query, top_k=4, paper_sha256s=hashes or None)))
        # Explicit multi-paper questions must not lose an entire requested paper
        # just because another paper dominates the global top-k.
        if len(hashes) > 1:
            for sha in sorted(hashes):
                hits = list(hits) + list(tool('paper.search.scoped', lambda sha=sha: adapter.search(
                    question, top_k=4, paper_sha256s={sha})))
        evidence = {}
        for hit in hits:
            if hashes and hit.get('paper_sha256') not in hashes:
                raise ValueError('Retriever returned evidence outside requested sources')
            eid = 'e_' + str(hit.get('stable_id') or hashlib.sha256(
                json.dumps(dict(hit), sort_keys=True).encode()).hexdigest())
            evidence[eid] = {**hit, 'evidence_id': eid}
        store = getattr(adapter, 'store', None)
        router = DurableResearchModel(runtime.llm_runtime, run_id, stage='section_router')
        papers = hashes or {str(h['paper_sha256']) for h in hits if h.get('paper_sha256')}
        if store is not None:
            routing_reports = {}
            source_inventories = {}
            for sha in sorted(papers):
                outline = tool('paper.outline', lambda sha=sha: store.paper_outline(sha))
                from .source_navigation import locate_requested_objects
                anchored = tool('paper.locate_objects', lambda sha=sha, outline=outline:
                    locate_requested_objects(store, sha, outline, run['question']))
                evidence.update({e['evidence_id']: e for e in anchored})
                from .answer_coverage import indexed_table_inventory
                source_inventories[sha] = tool('paper.source_inventory',
                    lambda sha=sha, outline=outline: indexed_table_inventory(store, outline, sha, requested_targets))
                db.update_run_metadata(run_id, {'reader_source_inventory_diagnostics': source_inventories,
                    'reader_source_inventory': [row for inventory in source_inventories.values()
                                                for row in inventory['rows']]})
                stage = 'section_router'
                def routing_progress(value, sha=sha):
                    routing_reports[sha] = value
                    db.update_run_metadata(run_id, {
                        'reader_routing': {'paper_sha256': sha, **value},
                        'reader_routing_by_paper': routing_reports})
                routing = select_sections(question, outline, router,
                    on_progress=routing_progress, original_question=run['question'],
                    request_history=answer_history)
                if routing['status'] == 'FAILED':
                    db.emit_event(run_id, 'reader.routing_degraded', {
                        'paper_sha256': sha, 'error_type': routing['error_type']})
                    if routing.get('error_type') not in {'LLMOutputTruncated', 'ValueError', 'JSONDecodeError'}:
                        continue  # Provider/authentication/unknown outcomes do not get a retry.
                if routing['status'] in {'EMPTY_SELECTION', 'FAILED'}:
                    extra, fallback = tool('paper.read_outline_fallback', lambda: read_outline_fallback(
                        store, sha, outline, run['question']))
                    routing = {**routing, 'fallback': fallback}
                    routing_progress(routing)
                    db.emit_event(run_id, 'reader.routing_degraded', {
                        'paper_sha256': sha, 'reason': 'unusable_heading_selection', **fallback})
                    # The original retriever missed this material. Preserve the
                    # fallback ranking before allocating the source-aware window.
                    evidence = {**{e['evidence_id']: e for e in extra}, **evidence}
                else:
                    ids = list(dict.fromkeys(structural_overviews(outline) + routing['section_ids']))
                    extra, omitted = tool('paper.read_sections', lambda: read_selected_sections(store, sha, outline, ids))
                    db.emit_event(run_id, 'reader.sections', {'paper_sha256': sha, 'omitted_chunks': omitted})
                    evidence.update({e['evidence_id']: e for e in extra})
                    # A selected heading can still omit its definitions because
                    # the section read cap is reached. Reload bounded originals
                    # for mechanism requests; add no provider attempt or verdict.
                    if re.search(r'机制|原理|组织|\b(?:mechanism|token|architecture|alignment)\b',
                                 run['question'], re.I):
                        definitions, navigation = tool('paper.read_method_definitions',
                            lambda: read_outline_fallback(store, sha, outline, run['question']))
                        routing_progress({**routing, 'method_definition_read': navigation})
                        evidence.update({e['evidence_id']: e for e in definitions})
        if report is None:
            if targets:
                from .reader_visual import collect_visual_evidence
                visual = collect_visual_evidence(runtime, run_id, question,
                    targets, hashes, tool)
                if not visual:
                    if policy != 'auto':
                        return finish({'status': 'NO_EVIDENCE', 'answer': '',
                                       'limitations': ['指定表格未取得可用的视觉转录，不能据此回答表格数值。']})
                    db.update_run_metadata(run_id, {'reader_visual_degraded_to_text': True})
                # Put selected table observations before generic text candidates.
                if visual:
                    evidence = {**{row['evidence_id']: row for row in visual}, **evidence}
                    db.update_run_metadata(run_id, {'capability_boundary':
                        'text and explicitly selected table transcriptions; no experiments'})
            if db.get_run(run_id)['status'] == 'CANCELLED':
                raise InterruptedError('Run cancelled')
            stage = 'context_selection'
            candidates = persist_reader_evidence(db, run_id, [
                {**item, 'page_numbering': metadata.get('page_numbering', 'unspecified')}
                for item in evidence.values()])
            from .numeric_scope import deterministic_numeric_attribution
            from .reader_visual import deterministic_ablation_summary, deterministic_table_difference
            exact = deterministic_numeric_attribution(question, candidates)
            if exact is None:
                exact = deterministic_table_difference(question, candidates)
            if exact is None:
                exact = deterministic_ablation_summary(question, candidates)
            if exact is not None:
                db.update_run_metadata(run_id, {'reader_deterministic_calculation': exact})
            # Matching an arithmetic subquestion does not prove that a template
            # satisfies the whole user request, including on the first turn.
            # All open-ended answers use the same Writer/Auditor and presentation
            # contract. Keep calculations diagnostic, never promote them into
            # source evidence or silently fall back to them when budget runs out.
            if exact is not None:
                db.update_run_metadata(run_id, {'reader_calculation_dispatch': {
                    'mode': 'REQUIREMENTS_AWARE_READER',
                    'has_request_history': bool(answer_history),
                    'question_rewritten': question != run['question'],
                    'calculation_is_source_evidence': False}})
            # Reserve room for Writer, Auditor and the shared repair. This is
            # a working-window allocation, not a claim of exact token counting.
            remaining = budget.max_tokens - db.get_run(run_id)['used_tokens']
            window = 50000 if budget.unlimited_llm_usage else min(50000, max(0, (remaining - 12000) // 3))
            candidates = prioritize_requested_evidence(candidates, requested_targets or targets)
            from .read_packet import build_read_packet
            packet = build_read_packet(run['question'], candidates, max_bytes=min(24000, window),
                                       requested_tables=requested_targets or targets)
            working, context = packet.rows, {**packet.metadata, 'packet_sha256': packet.identity}
            selected_ids = set(context['selected_evidence_ids'])
            context['requested_tables'] = requested_targets
            from .structured_extraction import table_observation_status
            context['table_evidence'] = [{**t, 'status': 'TRANSCRIBED' if any(
                row['evidence_id'] in selected_ids and table_observation_status(
                    row, t['paper_sha256'], t['label']) == 'TRANSCRIBED' for row in working) else 'NOT_VERIFIED'}
                for t in requested_targets]
            db.update_run_metadata(run_id, {'reader_workset': context})
            if context['unrepresented_sources']:
                from .read_packet import SourceCoverageError
                raise SourceCoverageError('Reader evidence does not represent every required source')
            stage = 'reader'
            report = run_durable_reader(db, run_id, working,
                DurableResearchModel(runtime.llm_runtime, run_id, stage='answer_first', reserve_class='synthesis'),
                protocol_id=protocol_id, question=question, finalize=False, shared_read_packet=packet)
            if report.get('read_packet'):
                actual_ids = set(report['read_packet']['selected_evidence_ids'])
                working = [row for row in working if row['evidence_id'] in actual_ids]
                context = {**context, 'selected_evidence_ids': [row['evidence_id'] for row in working],
                           'answer_packet': report['read_packet']}
                db.update_run_metadata(run_id, {'reader_workset': context})
            recover_empty = (report.get('status') == 'NO_EVIDENCE' and
                             bool(hashes or papers) and metadata.get('gap_search', True) is True)
            if (recover_empty or (report.get('status') == 'REVIEW_REQUIRED' and
                    report.get('answer_scope') != 'MODEL_REVIEWED')):
                # Keep the first result durable; a failed optional recovery must
                # not erase an already supported partial answer.
                db.update_run_metadata(run_id, {'reader_rounds': [report]})
                first = report
                stage = 'recovery_preflight'
                try:
                    current = db.get_run(run_id)
                    citation_plan = citation_revision_plan(report, working, question=run['question'],
                                                          request_history=answer_history)
                    citation_plan = citation_plan or boundary_revision_plan(report, working)
                    # The citation path needs a Writer/Auditor pair and the
                    # existing protected format-repair allowance, no decision call.
                    required_calls = 3 if citation_plan else 4
                    if (not budget.unlimited_llm_usage and
                            budget.max_llm_calls - current['used_llm_calls'] < required_calls):
                        raise LLMBudgetExhausted('Insufficient calls for recovery and re-audit')
                    stage = 'recovery_decision'
                    decision = citation_plan or plan_evidence_reload(question, report, candidates,
                        set(context['selected_evidence_ids']),
                        DurableResearchModel(runtime.llm_runtime, run_id, stage='reader_recovery'),
                        on_progress=lambda value: db.update_run_metadata(run_id, {'reader_recovery_diagnostics': value}),
                        allow_search=bool(hashes or papers) and metadata.get('gap_search', True) is True,
                        original_question=run['question'], request_history=answer_history)
                    db.update_run_metadata(run_id, {'reader_recovery': decision})
                    fresh = []
                    if decision['action'] == 'SEARCH':
                        from .reader_recovery import retrieve_gap_evidence
                        fresh = tool('paper.search.gap', lambda: retrieve_gap_evidence(
                            adapter, question, decision['query'], hashes or papers, candidates))
                        if fresh:
                            fresh = persist_reader_evidence(db, run_id, fresh)
                            candidates = candidates + fresh
                        db.update_run_metadata(run_id, {'reader_gap_search': {
                            'query': decision['query'], 'paper_sha256s': sorted(hashes or papers),
                            'new_evidence_ids': [row['evidence_id'] for row in fresh],
                            'status': 'NEW_EVIDENCE' if fresh else 'STOP_NO_NEW_EVIDENCE',
                            'max_queries': 1, 'top_k': 8}})
                        # No additional Writer/Auditor calls for empty or wholly
                        # repeated retrieval. Keep the supported partial result.
                        decision = {**decision, 'action': 'RELOAD' if fresh else 'STOP',
                                    'evidence_ids': [row['evidence_id'] for row in fresh]}
                    if decision['action'] in {'RELOAD', 'REVISE'}:
                        ids = set(decision['evidence_ids'])
                        requested = fresh or (tool('paper.reload_saved', lambda: [c for c in candidates if c['evidence_id'] in ids])
                                              if ids else [])
                        remaining = budget.max_tokens - db.get_run(run_id)['used_tokens']
                        if decision['action'] == 'REVISE':
                            expanded, reload_context = revision_workset(working, context)
                        else:
                            expanded, reload_context = source_aware_workset(
                                requested + [c for c in working if c['evidence_id'] not in ids],
                                max_bytes=50000 if budget.unlimited_llm_usage else
                                    min(50000, max(0, (remaining - 12000) // 3)))
                        required_ids = ids if decision['action'] == 'RELOAD' else {c['evidence_id'] for c in working}
                        if (not required_ids <= set(reload_context['selected_evidence_ids']) or
                                reload_context['unrepresented_sources']):
                            # An unfitted RELOAD does not license rewriting the
                            # same incomplete sources as if the gap were filled.
                            db.update_run_metadata(run_id, {'reader_recovery_fallback': {
                                'effective_action': 'STOP',
                                'reason': 'requested_reload_did_not_fit_without_source_loss'}})
                            from .read_packet import SourceCoverageError
                            raise SourceCoverageError('Requested evidence reload cannot fit without source loss')
                        db.update_run_metadata(run_id, {'reader_reload_workset': reload_context})
                        stage = 'reader_revision'
                        revision_packet = (packet if decision['action'] == 'REVISE' and
                            (first.get('read_packet') or {}).get('packet_sha256') == packet.identity else None)
                        revised = run_durable_reader(db, run_id, expanded,
                            DurableResearchModel(runtime.llm_runtime, run_id, stage='answer_first', reserve_class='synthesis'),
                            protocol_id=protocol_id, question=question, finalize=False,
                            checkpoint_key='answer_first_reload_checkpoint',
                            revision_feedback=revision_feedback(first, expanded), shared_read_packet=revision_packet)
                        db.update_run_metadata(run_id, {'reader_rounds': [first, revised]})
                        if revised.get('status') == 'FAILED':
                            # answer_then_audit returns failure receipts instead
                            # of raising. Do not silently hide a paid revision
                            # whose replacement never reached a completed audit.
                            if revised.get('error_type') == 'LLMBudgetExhausted':
                                raise LLMBudgetExhausted('Revision audit did not finish within the remaining budget')
                            raise RuntimeError('Revision failed: ' + str(revised.get('error_type', 'unknown')))
                        if revised.get('status') == 'REVIEW_REQUIRED' and revised.get('direct_answer_supported'):
                            report = revised
                except Exception as exc:
                    not_started = stage == 'recovery_preflight' and isinstance(exc, LLMBudgetExhausted)
                    db.update_run_metadata(run_id, {'reader_recovery_status':
                        'NOT_STARTED_BUDGET' if not_started else 'FAILED'})
                    db.update_run_metadata(run_id, {'reader_recovery_error': type(exc).__name__})
                    db.update_run_metadata(run_id, {'reader_recovery_error_detail': str(exc)[:500]})
                    db.update_run_metadata(run_id, {'reader_recovery_failure_stage': stage})
                    chinese = bool(re.search(r'[\u4e00-\u9fff]', run['question']))
                    if not_started:
                        notice = ('剩余调用预算不足，尚未启动补充取证；保留部分回答，仍需复核。' if chinese else
                                  'Additional evidence review was not started because the remaining call budget was insufficient; the partial answer still requires review.')
                    elif isinstance(exc, LLMBudgetExhausted):
                        notice = ('补充取证或修订审核因预算不足而未完成；保留旧版部分回答，新草稿未作为结论发布。' if chinese else
                                  'Additional evidence or revision review stopped for insufficient budget; the earlier partial answer is retained, not the unaudited replacement.')
                    else:
                        notice = ('补充取证未完成；保留部分回答，仍需复核。' if chinese else
                                  'Additional evidence review could not be completed; the partial answer still requires review.')
                    report = {**first, 'limitations': list(first.get('limitations', [])) + [notice]}
    except Exception as exc:
        report = {'status': 'FAILED', 'error_type': type(exc).__name__, 'failure_stage': stage,
                  'limitations': [f'本轮在 {stage} 阶段未完成，未将未核验的草稿作为最终回答。']}
    missing_visual = db.get_run(run_id)['metadata'].get('reader_visual_skipped') or []
    if missing_visual and report is not None:
        report = {**report, 'limitations': list(report.get('limitations', [])) + [
            '部分指定图表未取得可用视觉转录；这些图表的数值仍未核实。']}
        if report.get('answer_scope') == 'MODEL_REVIEWED':
            report['answer_scope'] = 'PARTIAL'
    # A cancelled run remains cancelled; do not publish late output over it.
    return finish(report)
