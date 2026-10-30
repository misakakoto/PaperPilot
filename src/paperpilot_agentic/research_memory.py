"""Provenance-backed research request memory; never a memory of model claims."""
import json
import re

VERSION = 'request-ledger-v1'
PIN = re.compile(r'请记住|后续.*(?:都|始终)|全程|始终|for the rest|always|remember that',re.I)
CORRECTION = re.compile(r'改为|改成|不再|不要再|替换|取消.*要求|instead|replace|no longer|stop using',re.I)
RESET = re.compile(r'清除之前的约束|清除之前的要求|重新开始研究|reset (?:the )?(?:constraints|research)',re.I)
RETENTION = 'retain-user-requests-v3'
NEUTRAL = re.compile(r'\s*(?:继续|继续分析|继续比较|接着说|continue|go on)\s*[。.!！?？]?\s*', re.I)


def parent_memory(db,parent_run):
    meta = parent_run['metadata']
    saved = meta.get('reader_task_memory') or {}
    root = saved.get('root_run_id',parent_run['run_id'])
    recent = saved.get('recent_run_ids',[parent_run['run_id']])[-4:]
    # Persist references independently of the model window. An unrelated new
    # requirement must not evict an older correction and revive the root value.
    pins = list(dict.fromkeys(saved.get('pinned_run_ids', [])))
    # Old snapshots selected requests by keywords. Rebuild from owned ancestry,
    # when available, rather than declaring a lossy snapshot complete.
    ancestry_complete = True
    if saved and saved.get('retention_policy') != RETENTION:
        cursor, seen, recovered = parent_run, set(), []
        while cursor['run_id'] != root:
            cid = cursor['run_id']
            if cid in seen:
                raise ValueError('Research memory ancestry cycle')
            seen.add(cid)
            if cursor['metadata'].get('conversation_id') != meta.get('conversation_id'):
                raise ValueError('Research memory source is outside conversation')
            if not NEUTRAL.fullmatch(cursor['question']):
                recovered.append(cid)
            previous = cursor['metadata'].get('parent_run_id')
            if not previous:
                ancestry_complete = False
                break
            cursor = db.get_run(previous)
        pins = list(dict.fromkeys([*reversed(recovered), *pins]))
    elif saved.get('requirements_complete') is False:
        ancestry_complete = False
    ids = list(dict.fromkeys([root,*pins,*recent]))
    records,omitted,used = [],[],0
    for rid in ids:
        row = db.get_run(rid)
        if row['metadata'].get('conversation_id') != meta.get('conversation_id'):
            raise ValueError('Research memory source is outside conversation')
        item = {'run_id':rid,'user_request':row['question'],
                'created_at':row['created_at'],
                'role':'initial_request' if rid==root else 'pinned_request' if rid in pins else 'recent_request'}
        size = len(json.dumps(item,ensure_ascii=False).encode('utf-8'))
        if used+size>24000:
            if rid==root:
                raise ValueError('Initial research request exceeds memory window; restate the active requirements')
            omitted.append(rid)
            continue
        records.append(item)
        used += size
    records.sort(key=lambda r:r['created_at'])
    return {'version':VERSION,'root_run_id':root,'recent_run_ids':recent,'pinned_run_ids':pins,
            'requests':records,'omitted_run_ids':omitted,'working_bytes':used,
            'retention_policy':RETENTION,
            'requirements_complete': ancestry_complete and not any(rid in pins for rid in omitted),
            'ancestry_complete': ancestry_complete,
            'purpose':'User requests for reference resolution only; not scientific evidence or permanent authorization'}


def advance_memory(run,history,*,continue_task,scope):
    reset = not continue_task or RESET.search(run['question'])
    rid = run['run_id']
    root = rid if reset else history['root_run_id']
    recent = [] if reset else list(history['recent_run_ids'])
    pins = [] if reset else list(history['pinned_run_ids'])
    # The root is always retained; do not consume a pin slot on a new task.
    if rid != root and not NEUTRAL.fullmatch(run['question']):
        pins.append(rid)
    return {'version':VERSION,'root_run_id':root,
            'recent_run_ids':list(dict.fromkeys([*recent,rid]))[-4:],
            'pinned_run_ids':list(dict.fromkeys(pins)),
            'retention_policy':RETENTION,
            'requirements_complete': True if reset else history.get('ancestry_complete', True),
            'resolved_paper_sha256s':sorted(scope),
            'history_is_evidence':False}
