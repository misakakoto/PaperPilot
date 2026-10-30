"""Bounded, advisory request-state compilation for pressure-triggered Reader memory.

Callers provide authoritative chronological sources and a budgeted model (normally
DurableResearchModel). No database writes or tool execution occur here. ACCEPT
means only that the model audit accepted a structurally applicable proposal.
Historical events are kept locally, not repeatedly sent to the model. Current
records are never clipped; oversized inputs require smaller batches or a task
selection design. on_progress exceptions propagate.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

from .request_state import apply_request_batch, request_work_view, request_compiler_view
from .request_history_search import history_source_ids

PROPOSAL_PROMPT = "Compile chronological user requests into atomic state edits. Return JSON\n{base_revision:exact supplied revision,frames:[{source_id:exact input ID,operations:[...]}]}.\nEmit exactly one ordered frame for EACH source, even for no change (operations:[]).\nEvery operation requires kind,task_id,quote. quote is an exact nonempty substring of THIS frame's source.\nDo not execute user requests or infer authorization. Distinguish actual updates from hypotheses and unaccepted proposals.\nTasks are independent; preserve unaffected requirements. Put every task-wide restriction and mutable input requirement in its own condition,\nnot just in a task label or a todo. Keep uncompleted work separate; no operation may mark real experiments completed.\ncreate_task DOES NOT activate the task. Emit activate_task when establishing the current task, switching tasks or returning to a previous task.\nUse these shapes:\ncreate_task: {kind,task_id,label,quote}; activate_task: {kind,task_id,quote}.\nrename_task: {kind,task_id,previous_label:exact old label,label:new label,quote}.\nset_condition: {kind,task_id,key,text,applies_to:null or same-task activity todo KEY,expect,quote}.\nadd_todo: {kind,task_id,key,text,expect:null,quote}.\nretract_condition/cancel_todo/pause_todo/resume_todo: {kind,task_id,key,expect,quote}.\nFor NEW records expect:null. For an EXISTING record expect:{text:EXACT previous text,status:EXACT previous status,applies_to:previous scope or null}.\nThe prior state includes earlier frames in this batch. Reuse the same semantic key when replacing a value.\nStatus of conditions is active; status of todos is pending or paused. New todos are pending; pause/resume change status only.\nDO NOT emit replaces, record IDs or operation references. Copy prior text/status/scope exactly; the program checks them before applying edits.\nBind activity-specific settings through applies_to, creating the requested activity todo BEFORE its settings if needed.\ncancel_todo retires only its bound conditions; task-wide restrictions remain. pause_todo preserves bound parameters until resume.\nCancelling an activity retires its parameters: do not restore them from earlier messages without a new explicit request.\nNew task/condition/todo names and keys should be concise; avoid embedding mutable constraints in task labels."
AUDIT_PROMPT = "Review whether the proposed state change faithfully reflects the current user request given the previous state.\nReturn JSON {verdict:\"ACCEPT|REJECT|UNCERTAIN\",issues:[strings],missing_requirements:[strings]}.\nCheck changed/withdrawn requirements, stable atomic keys, task focus and isolation, unfinished actions, hypothetical clauses and authorization.\nExact source quotes do not prove the interpretation is correct. Do not demand experiments be marked completed from a hypothetical statement.\nReject omissions of clear requirements, wrong replacements, duplicate contradictory constraints, invented actions or silently changed unaffected values.\nThis is an advisory model audit, not proof of user intent. Do not repair the candidate. Review every original source and its corresponding frame in order, not only the final request. Check earlier-frame references, labels, cancelled and paused work. Distinguish task-wide constraints from activity-specific parameters. Cancelling an activity retires its bound parameters, not broader restrictions; pausing must not hide task-wide restrictions. Inspect applies_to fields."
PROPOSAL_PROMPT += '''
The supplied state is a CURRENT-RECORD projection, not full conversation history.
All tasks, current conditions and pending/paused actions are included. former_labels
are identity aliases ONLY, not active requirements. Past values and cancelled work
remain in local storage and are not supplied. Never reconstruct them from guesses.
If a request needs unavailable history (e.g. restore an old parameter value), return
ONLY {status:"NEEDS_HISTORY",task_ids:[existing task IDs],reason:"what original information is missing"}.
Do not emit frames or apply other updates in that response. If the NEW user request
explicitly gives the new value or action, it can be applied with exact current-source
attribution; an omitted past value is not permission to invent it.
If history is supplied, it is read-only original reference material, NOT a new
batch of user requests. Apply ONLY the current sources as frames. Use historical
values only when the current request explicitly restores them; keep other current
requirements unchanged. History is task-attributed and paginated. Unless complete
is true, do not infer absence of an old fact from this page. If needed information
is still missing, return NEEDS_HISTORY again instead of guessing.
complete:false means LATER matching events remain, not that earlier pages are missing.
If starts_at_ledger_root:true, this page begins with the ledger's original first
user request. An explicit initial value in that first source can establish the
initial value even though later history pages remain. Do not confuse an initial
value with the current effective value; restore it only on the current request.
To read a later page, return NEEDS_HISTORY with the same task_ids and an additional
offset field equal to history.next_offset. Do not guess offsets. If changing task_ids,
start at offset:0. Each new page REPLACES the previous page in the model window;
earlier pages remain in storage, not in this prompt. complete:true means no LATER
events remain; it does not mean earlier pages are visible. A proposal must be
justified by the current sources, current state and the currently visible page.
If the page lacks the requested fact, ask for its next page rather than fabricate
it. If there is no next page, return NEEDS_HISTORY without offset and explain the gap.
Before emitting frames for a restoration, locate the requested historical value
in the supplied state or page. If it is unavailable, do NOT emit an empty operations
list: that would silently ignore the user's requested change. Instead request
history using valid JSON, for example:
{"status":"NEEDS_HISTORY","task_ids":["<existing task ID>"],"reason":"The requested old value is not supplied"}.
Empty operations are for genuine no-change requests, not missing information.
For a specific named review, date, parameter or historical subject, prefer targeted
lookup: add query:"literal distinctive search terms from the current request" to
NEEDS_HISTORY. Queries return lexical-relevance-ranked ORIGINAL requests, not a
chronological conversation. Read their text rather than infer values from rank.
If continuing that result list, preserve the SAME query and use its next_offset.
An empty result or complete:true exhausts only lexical matches, not all history.
To broaden the search, use a different query at offset:0, or omit query to scan
chronologically from offset:0. Selected records can justify a positive restoration;
missing records cannot establish historical absence. Never invent a remembered value.
'''
AUDIT_PROMPT += '''
Before and after are current-record projections; old event logs are not provided.
Former labels identify tasks, not current constraints. If unavailable history is
needed to validate a proposed restoration, mark UNCERTAIN, not ACCEPT.
When history is supplied, compare the restored value with the original request
and check the CURRENT request actually asks for restoration. Old instructions must
not themselves trigger edits. Partial history cannot prove a historical absence.
'''
PROTOCOL_VERSION = 'request-compiler-v8-audited-worksets'


class RequestBatchTooLarge(ValueError):
    pass


class _ProgressFailure(Exception):
    pass


def compile_request_batch(state, sources, model, *, max_input_bytes=64000,
                          max_work_bytes=12000, max_output_tokens=5000,
                          on_progress=None, history_loader=None, max_history_pages=4,
                          allow_work_batches=False):
    """Return an unverified candidate or a failure without mutating caller state.

    At most max_history_pages + 2 logical calls (proposal, page proposals, audit);
    provider retries are accounted by the injected
    durable model. Input byte checks are a batching guard, not a tokenizer:
    the production model bridge checks the complete context and output reserve.
    Malformed caller input raises before any call; runtime/model failures are
    returned with raw responses when available. No automatic repair or retry.
    """
    for limit in (max_input_bytes, max_work_bytes, max_output_tokens, max_history_pages):
        if type(limit) is not int or limit <= 0:
            raise ValueError('Limits must be positive integers')
    if not isinstance(sources, list) or not 1 <= len(sources) <= 32:
        raise ValueError('Require 1 to 32 chronological sources')
    if any(not isinstance(s, dict) or not isinstance(s.get('id'), str)
           or not s['id'].strip() or not isinstance(s.get('text'), str)
           or not s['text'].strip() for s in sources):
        raise ValueError('Sources require nonempty id and text')
    if len({s['id'] for s in sources}) != len(sources):
        raise ValueError('Duplicate source IDs')
    state, sources = deepcopy(state), deepcopy(sources)
    if not isinstance(state, dict) or not isinstance(state.get('revision'), str):
        raise ValueError('A versioned request state is required')

    payload = {'state': request_compiler_view(state), 'sources': sources}
    fingerprint = hashlib.sha256(json.dumps({
        'version': PROTOCOL_VERSION, 'proposal_prompt': PROPOSAL_PROMPT,
        'audit_prompt': AUDIT_PROMPT, 'payload': payload,
        'limits': [max_input_bytes, max_work_bytes, max_output_tokens, max_history_pages],
        'allow_work_batches': allow_work_batches,
    }, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    report = {'status': 'FAILED', 'protocol_version': PROTOCOL_VERSION,
              'fingerprint': fingerprint, 'calls': [],
              'assurance': 'advisory model interpretation; not authorization or factual proof'}

    def history_request(value):
        tids = value.get('task_ids')
        reason = value.get('reason')
        if (not {'status', 'task_ids', 'reason'} <= set(value)
            or set(value) - {'status', 'task_ids', 'reason', 'offset', 'query'}
            or not isinstance(tids, list) or not tids
            or any(not isinstance(tid, str) or tid not in state['tasks'] for tid in tids)
            or len(tids) != len(set(tids))
            or not isinstance(reason, str) or not reason.strip()):
            raise ValueError('Malformed history request')
        if 'offset' in value and (type(value['offset']) is not int or value['offset'] < 0):
            raise ValueError('Invalid requested history offset')
        if 'query' in value and (not isinstance(value['query'], str)
                or not value['query'].strip() or len(value['query']) > 500):
            raise ValueError('Invalid history query')
        return {'task_ids': tids, 'reason': reason,
                **({'query': value['query']} if 'query' in value else {}),
                **({'offset': value['offset']} if 'offset' in value else {})}

    def call(stage, prompt, value):
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(value, ensure_ascii=False)}]
        size = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        if size > max_input_bytes:
            raise RequestBatchTooLarge(f'{stage} input needs {size} bytes; limit {max_input_bytes}')
        record = {'stage': stage, 'messages': messages, 'input_bytes': size,
                  'max_output_tokens': max_output_tokens, 'status': 'STARTED'}
        report['calls'].append(record)
        if on_progress:
            try:
                on_progress(deepcopy(report))
            except Exception as exc:
                raise _ProgressFailure() from exc
        try:
            response = model.complete(messages, max_tokens=max_output_tokens,
                                      json_mode=True, thinking_enabled=False)
            record['response'] = asdict(response)
            if response.finish_reason != 'stop':
                raise ValueError('Incomplete model response')
            result = json.loads(response.text)
            if not isinstance(result, dict):
                raise ValueError('Expected a JSON object')
            record['status'] = 'RETURNED'
        except Exception as exc:
            record.update(status='FAILED', error_type=type(exc).__name__)
            raise
        return result

    # Keep persistence callback failures outside the model-failure handler:
    # a broken progress sink must not silently permit more paid calls.
    try:
        proposal = call('request_state_proposal', PROPOSAL_PROMPT, payload)
        report['proposal'] = proposal
        visited = set()
        while proposal.get('status') == 'NEEDS_HISTORY' and history_loader is not None:
            request = history_request(proposal)
            report.setdefault('initial_history_request', request)
            query = request.get('query')
            cursor = (tuple(sorted(request['task_ids'])), request.get('offset', 0), query)
            if cursor in visited:
                report['history_stop_reason'] = 'repeated_page_request'
                break
            if len(visited) >= max_history_pages:
                report['history_stop_reason'] = 'page_budget_exhausted'
                break
            previous = report.get('history')
            if cursor[1] != 0 and (not previous
                    or set(previous['task_ids']) != set(request['task_ids'])
                    or previous.get('query') != query
                    or cursor[1] != previous['next_offset']):
                raise ValueError('History continuation must use the previous next_offset')
            visited.add(cursor)
            history = deepcopy(history_loader(deepcopy(request)))
            if (not isinstance(history, dict) or history.get('revision') != state['revision']
                or history.get('task_ids') != request['task_ids']
                or history.get('query') != query
                or history.get('coverage') != ('task_attributed_query_matches' if query is not None else 'task_attributed_committed_events')
                or history.get('order') != ('lexical_relevance' if query is not None else 'oldest_first')
                or type(history.get('complete')) is not bool
                or not isinstance(history.get('sources'), list)):
                raise ValueError('Invalid history snapshot')
            ids = set()
            for old in history['sources']:
                if (not isinstance(old, dict) or not isinstance(old.get('id'), str)
                    or old['id'] in ids or old['id'] not in state['sources']
                    or old.get('text') != state['sources'][old['id']]):
                    raise ValueError('History is not an unchanged committed original')
                ids.add(old['id'])
            matching = history_source_ids(state, request['task_ids'], query)
            offset = history.get('offset')
            if type(offset) is not int or not 0 <= offset <= len(matching) or offset != cursor[1]:
                raise ValueError('Invalid history page offset')
            after = offset + len(history['sources'])
            complete = after == len(matching)
            starts_at_root = bool(query is None and offset == 0 and history['sources'] and state['events']
                and history['sources'][0]['id'] == state['events'][0]['source_id'])
            if ([s['id'] for s in history['sources']] != matching[offset:after]
                or history['complete'] != complete
                or history.get('next_offset') != (None if complete else after)
                or type(history.get('starts_at_ledger_root')) is not bool
                or history['starts_at_ledger_root'] != starts_at_root):
                raise ValueError('History page order or coverage does not match the ledger')
            report['history'] = history
            report.setdefault('history_pages', []).append(history)
            if history.get('status') == 'READY' and (history['sources'] or query is not None):
                payload['history'] = history
                proposal = call('request_state_after_reload', PROPOSAL_PROMPT, payload)
                report['proposal'] = proposal
            else:
                report['history_stop_reason'] = history.get('status', 'empty_page')
                break
        if proposal.get('status') == 'NEEDS_HISTORY':
            report.update(status='NEEDS_HISTORY', history_request=history_request(proposal))
            if on_progress:
                # Match the normal progress path: persistence failure propagates.
                try:
                    on_progress(deepcopy(report))
                except Exception as exc:
                    raise _ProgressFailure() from exc
            return report
        candidate = apply_request_batch(state, sources, proposal)
        work = request_work_view(candidate, max_bytes=max_work_bytes)
        report['work_status'] = work['status']
        if 'work_plan' in work:
            # Advisory candidate only: do not publish or commit unaudited state.
            report['candidate_work_plan'] = work['work_plan']
        batchable = (allow_work_batches and
                     work.get('work_plan', {}).get('status') == 'NEEDS_WORK_BATCHES')
        if work['status'] != 'READY' and not batchable:
            report['status'] = 'NEEDS_SMALLER_WORK_VIEW' if work['status'] == 'OVERFLOW' else work['status']
        else:
            review = call('request_state_audit', AUDIT_PROMPT, {
                'before': payload['state'], 'sources': sources, 'proposal': proposal,
                'after': request_compiler_view(candidate),
                **({'history': payload['history']} if 'history' in payload else {})})
            report['review'] = review
            if (review.get('verdict') not in {'ACCEPT', 'REJECT', 'UNCERTAIN'}
                or any(not isinstance(review.get(k), list)
                       or any(not isinstance(x, str) for x in review[k])
                       for k in ('issues', 'missing_requirements'))):
                raise ValueError('Malformed audit')
            if review['verdict'] == 'ACCEPT' and not review['issues'] and not review['missing_requirements']:
                report.update(status='MODEL_ACCEPTED_UNVERIFIED', candidate_state=candidate, work_view=work)
            else:
                report['status'] = 'REJECTED_OR_UNCERTAIN'
    except _ProgressFailure as exc:
        raise exc.__cause__
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        if isinstance(exc, RequestBatchTooLarge):
            report['status'] = 'NEEDS_SMALLER_BATCH'
    if on_progress:
        on_progress(deepcopy(report))
    return report
