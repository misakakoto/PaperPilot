"""Conversation-facing interface over durable research runs.

The UI consumes messages, activities, approvals and artifacts. Engine details and
execution authority remain with the existing runtime; a displayed plan is not a grant.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from .models import utc_now
from .research_journal import ResearchJournal
from .runtime_profile import DEFAULT_PROFILE

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'}
CONVERSATION = "COALESCE(json_extract(metadata_json,'$.conversation_id'),run_id)"


def _research_progress(runtime, run_id, episode):
    """Read owned cursor receipts for display, never load weights or authorize work."""
    if not episode.get('candidates') or not episode['candidates'][-1].get('card'):
        return None
    from .research_training import episode_folder
    from .reference_methods import encoded, sha
    card = episode['candidates'][-1]['card']
    binding, protocol = card['binding_id'], card.get('protocol', {})
    if not re.fullmatch('[a-f0-9]{64}', binding):
        return None
    root = episode_folder(runtime, run_id)/binding
    if root.is_symlink() or not root.resolve().is_relative_to(runtime.workspace.resolve()):
        return None
    names = {f'trials/{s}-{v}': ('formal', s, v) for s in (7,11,19) for v in ('baseline','candidate')}
    names.update({'preflight': ('preflight',7,'candidate'),
                  'recovery-continuous': ('recovery',7,'candidate'),
                  'recovery-resumed': ('recovery',7,'candidate')})
    progress = {'recorded_trials': 0, 'total_trials': 6, 'current': None,
                'assurance': 'Saved progress only; not a checkpoint payload or metric verification'}
    try:
        saved = root/'state.json'
        if saved.exists() and not saved.is_symlink() and saved.resolve().is_relative_to(root.resolve()) and saved.stat().st_size < 256000:
            rows = json.loads(saved.read_bytes()).get('trials', {})
            progress['recorded_trials'] = sum(rows.get(f'{s}-{v}',{}).get('status')=='VERIFIED'
                for s in (7,11,19) for v in ('baseline','candidate'))
        launches = episode['gpu_budget']['launches']
        if not launches:
            return progress
        latest = max(launches, key=lambda row: row['started'])
        for name, (stage, seed, variant) in names.items():
            folder = root/name
            if Path(latest['stage']).resolve() != folder.resolve():
                continue
            pointer = folder/'checkpoints'/'latest.json'
            if (pointer.is_symlink() or not pointer.resolve().is_relative_to(root.resolve())
                    or not pointer.exists() or pointer.stat().st_size > 4096):
                return progress
            receipt = json.loads(pointer.read_bytes())
            expected = {'binding_id': binding, 'protocol_sha256': sha(encoded(protocol)),
                        'stage':stage, 'seed':seed, 'variant':variant}
            target = (protocol['epochs']*protocol['train_images']//protocol['batch_size'] if stage=='formal'
                else protocol['preflight']['steps'] if stage=='preflight' else protocol['recovery_check']['steps'])
            if (receipt.get('identity') == expected and type(receipt.get('step')) is int
                    and 0 <= receipt['step'] <= target):
                progress['current'] = {'stage':stage,'seed':seed,'variant':variant,
                    'step':receipt['step'],'target':target,'process_status':latest['status']}
            break
    except (ValueError, KeyError, TypeError, AttributeError, OSError, ZeroDivisionError):
        pass  # A transient/malformed progress pointer must not become a task verdict.
    return progress


def cursor_encode(values):
    return base64.urlsafe_b64encode(json.dumps(values).encode()).decode()


def cursor_decode(value):
    if not value:
        return None
    try:
        result = json.loads(base64.urlsafe_b64decode(value))
        if not isinstance(result, list) or len(result) != 2 or not all(isinstance(v, str) for v in result):
            raise ValueError()
        return result
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError('Invalid pagination cursor') from exc


def activity(event):
    kind, data = event['event_type'], event.get('payload') or {}
    labels = {
        'run.created': '任务已接收', 'orchestrator.queued': '已进入研究队列',
        'reader.sections': '已读取相关章节', 'reader.routing_degraded': '章节定位未完成，保留已检索材料',
        'evidence.added': '已保存来源证据', 'tool.started': '正在使用研究工具',
        'notebook.context_split': '材料较长，已拆分为可重载批次',
        'notebook.extraction_rejected': '一批笔记需要重新核查',
    }
    if kind == 'run.status':
        labels[kind] = {'PENDING': '任务排队中', 'RUNNING': '正在研究',
            'WAITING_APPROVAL': '等待你确认执行协议', 'COMPLETED': '本轮处理结束',
            'FAILED': '本轮遇到错误', 'CANCELLED': '已请求停止',
            'BUDGET_EXHAUSTED': '本轮预算已用尽'}.get(data.get('status'), '任务状态已更新')
    if kind.startswith('llm.'):
        labels[kind] = '模型调用状态已更新'
    if kind == 'llm.attempt.started':
        labels[kind] = '正在调用模型（取消不保证已发请求停止计费）'
    if kind == 'reader.tool':
        labels[kind] = {'paper.search':'正在检索论文原文', 'paper.read':'正在阅读原文',
                       'paper.search.scoped':'正在补充指定论文的证据'}.get(data.get('name'),'正在读取研究材料')
    if kind == 'research.plan':
        labels[kind] = '已生成本轮研究计划；执行仍受程序校验与审批约束'
    if kind == 'research.action':
        role = {'lead': '研究负责人', 'engineer': '实现 Agent', 'analyst': '分析 Agent'}.get(data.get('role'), '科研任务')
        labels[kind] = role+'：'+str(data.get('reason') or data.get('error') or data.get('action') or '')
    if kind == 'research.trial_verified':
        labels[kind] = f"已独立复算种子 {data['seed']} 的 {data['variant']} 指标"
    if kind == 'research.blocked':
        labels[kind] = '科研任务已停止：'+data['reason']
    if kind == 'prototype.goal.revised':
        labels[kind] = '研究目标已更新，设计和实现需要重新验证；历史产物已保留'
    if kind == 'prototype.sources.revised':
        labels[kind] = '资料版本已更新，受影响的任务需要重做；历史产物已保留'
    if kind == 'prototype.constraints.revised':
        labels[kind] = '验证要求已更新，历史验证已保留'
    if kind == 'research.task.progress':
        labels[kind] = f"已读取 {data['completed']}/{data['total']} 篇；正在整理可核对的原文"
    if kind == 'research.task.planned':
        labels[kind] = '研究计划已保存，等待确认总预算'
    if kind == 'research.task.budget_approved':
        labels[kind] = '研究总预算已确认；不授权实验或训练'
    if kind.startswith('tool.') and kind not in labels:
        labels[kind] = '工具执行状态已更新'
    if kind not in labels:
        return None
    return {'id': event['event_id'], 'kind': 'activity', 'text': labels[kind],
            'created_at': event.get('created_at')}


class ResearchWorkbench:
    def __init__(self, runtime, background):
        self.runtime, self.db, self.background = runtime, runtime.db, background
        self.journal = ResearchJournal(self.db)
        with self.db.transaction() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS workbench_submissions(
                request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                state TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL)''')
            columns = {r['name'] for r in conn.execute('PRAGMA table_info(workbench_submissions)')}
            if 'conversation_id' not in columns:
                conn.execute('ALTER TABLE workbench_submissions ADD COLUMN conversation_id TEXT')
            conn.execute(f'CREATE INDEX IF NOT EXISTS runs_conversation ON runs({CONVERSATION},created_at,run_id)')

    def conversations(self, *, cursor=None, limit=30):
        size = max(1, min(100, int(limit)))
        position = cursor_decode(cursor)
        condition, args = ('', []) if position is None else (
            'WHERE (updated_at,conversation_id) < (?,?)', position)
        with self.db.connect() as conn:
            rows = conn.execute(f'''WITH grouped AS (
                SELECT {CONVERSATION} conversation_id, MAX(updated_at) updated_at,
                    COUNT(*) turn_count FROM runs GROUP BY {CONVERSATION})
                SELECT * FROM grouped {condition} ORDER BY updated_at DESC,conversation_id DESC LIMIT ?''',
                (*args, size + 1)).fetchall()
            items = []
            for row in rows[:size]:
                first = conn.execute(f'SELECT question FROM runs WHERE {CONVERSATION}=? ORDER BY created_at,run_id LIMIT 1',
                    (row['conversation_id'],)).fetchone()
                latest = conn.execute(f'SELECT run_id FROM runs WHERE {CONVERSATION}=? ORDER BY created_at DESC,run_id DESC LIMIT 1',
                    (row['conversation_id'],)).fetchone()
                items.append({**dict(row), 'title': first['question'][:160],
                              'latest_run_id': latest['run_id']})
        return {'items': items, 'next_cursor': cursor_encode([rows[size-1]['updated_at'],
                rows[size-1]['conversation_id']]) if len(rows) > size else None}

    def runs(self, conversation_id, *, cursor=None, limit=20):
        size = max(1, min(50, int(limit)))
        position = cursor_decode(cursor)
        condition, args = ('', []) if position is None else ('AND (created_at,run_id)<(?,?)', position)
        with self.db.connect() as conn:
            rows = conn.execute(f'''SELECT run_id,created_at FROM runs WHERE {CONVERSATION}=?
                {condition} ORDER BY created_at DESC,run_id DESC LIMIT ?''',
                (conversation_id, *args, size+1)).fetchall()
        if not rows and not cursor:
            raise KeyError('Conversation not found')
        return {'items': [self.turn(r['run_id']) for r in reversed(rows[:size])],
                'next_cursor': cursor_encode([rows[size-1]['created_at'],rows[size-1]['run_id']])
                    if len(rows) > size else None}

    def owned(self, conversation_id, run_id):
        run = self.db.get_run(run_id)
        if (run['metadata'].get('conversation_id') or run['run_id']) != conversation_id:
            raise ValueError('Run belongs to another conversation')
        return run

    def turn(self, run_id):
        data = self.runtime.status(run_id, include_trace=False)
        run, final = data['run'], data.get('final_answer') or {}
        active = self.background.active(run_id)
        unresolved_execution = any(c['status'] == 'RUNNING' for c in data['tool_calls'])
        status = run['status']
        if status == 'CANCELLED' and (active or unresolved_execution):
            status = 'STOPPING' if active else 'EXECUTION_UNKNOWN'
        elif status in {'PENDING','RUNNING'} and not active:
            status = 'INTERRUPTED'
        state_text = {
            'PENDING': '已排队', 'RUNNING': '正在研究', 'STOPPING': '正在停止，请等待执行进程退出',
            'EXECUTION_UNKNOWN': '已阻止后续动作，但上次执行是否结束仍需核对',
            'INTERRUPTED': '任务未结束，服务可能已重启；恢复前会核对检查点',
            'WAITING_APPROVAL': '等待确认实验协议', 'FAILED': '本轮失败，已有材料仍保留',
            'CANCELLED': '本轮已停止', 'BUDGET_EXHAUSTED': '预算已用尽，不代表研究已完成',
            'COMPLETED': '本轮已结束',
        }.get(status, status)
        if status == 'COMPLETED' and (final.get('review_required') or run.get('research_outcome') != 'ANSWERED'):
            state_text += '，部分内容仍需核验'
        failure = data.get('failure')
        if failure and status != 'CANCELLED':
            state_text = failure['title'] + '；' + failure['action']
        evidence = data['evidence']
        hashes = {str(e.get('source_uri','')).split('/')[2] for e in evidence
                  if str(e.get('source_uri','')).startswith('paper://')}
        store = getattr(self.runtime.search_adapter, 'store', None)
        titles = {r['sha256']: r.get('title') or r['file_name']
                  for r in store.paper_metadata(hashes)} if store and hashes else {}
        citations = []
        for i, eid in enumerate(final.get('citation_ids') or [], 1):
            item = next((e for e in evidence if e['evidence_id'] == eid), None)
            if not item:
                continue
            uri = str(item.get('source_uri') or '')
            sha = uri.split('/')[2] if uri.startswith('paper://') else ''
            provenance = item.get('provenance') or {}
            title = titles.get(sha) or provenance.get('title') or provenance.get('paper_title') or '来源证据'
            citations.append({'id': eid, 'number': i, 'title': title, 'page': item.get('page'),
                              'label': f'[{i}] {title}' + (f" · 第 {item['page']} 页" if item.get('page') else '')})
        notes_root = run['metadata'].get('notebook_root_run_id')
        if run['metadata'].get('engine') == 'research_notebook':
            notes_root = run_id
        prototype = None
        if run['metadata'].get('engine') == 'research_prototype_v1':
            from .prototype_harness import PrototypeHarness
            prototype = PrototypeHarness(self.runtime).view(run_id)
            state_text = {
                'WAITING_ENVIRONMENT': '执行环境尚未就绪；候选已保存，恢复时仅重新检查环境',
                'UNKNOWN': '请求或执行结果未知，禁止自动重发',
                'PARTIAL': '原型部分完成；详见验证结果与未完成项',
                'FAILED': '原型任务遇到错误，已有产物保留',
            }.get(prototype['phase'], state_text)
        task = None
        prototype_codes = [t for t in prototype['work']['tasks'].values() if t['capability']=='implement'] if prototype else []
        can_revise_prototype = bool(prototype and prototype.get('validation_contract') and prototype_codes
            and all(t['status']=='DONE' for t in prototype_codes) and not active and not unresolved_execution
            and prototype['phase'] in {'READY','WAITING_APPROVAL','WAITING_ENVIRONMENT','COMPLETED','PARTIAL'}
            and not (data.get('llm_budget') or {}).get('unknown_results'))
        episode = None
        if run['metadata'].get('engine') in {'research_loop_v1', 'research_diagnosis_v1'}:
            from .research_harness import ResearchHarness
            episode = ResearchHarness(self.runtime).view(run_id)
            episode['execution_progress'] = _research_progress(self.runtime, run_id, episode)
            if episode['phase'] in {'UNKNOWN', 'FAILED', 'PARTIAL', 'PAUSED'}:
                state_text = {'UNKNOWN': '请求或进程结果未知，禁止自动重发', 'FAILED': '科研任务遇到错误，记录保留',
                              'PARTIAL': '科研任务部分完成', 'PAUSED': '训练已暂停，检查点保留'}[episode['phase']]
        if run['metadata'].get('engine') == 'research_task':
            from .research_task import ResearchTask
            task = ResearchTask(self.runtime, run_id).view()
            if task and task['state'] == 'AWAITING_BUDGET':
                state_text = '等待确认研究总预算；尚未开始批量读取，不授权实验'
            elif task and task.get('pause_reason'):
                state_text = task['pause_reason']
        tool_names = {c['tool_call_id']: c['tool_name'] for c in data['tool_calls']}
        return {'run_id': run_id, 'conversation_id': run['metadata'].get('conversation_id') or run_id,
            'failure': failure,
            'models': list(dict.fromkeys(c['model'] for c in data['llm_calls'])),
            'research_task': task,
            'research_episode': episode,
            'can_revise_prototype': can_revise_prototype,
            'created_at': run['created_at'], 'status': status, 'state_text': state_text,
            'worker_active':active,
            'error':(run['metadata'].get('reader_report') or {}).get('failure_detail') or run.get('error'),
            'message': {'question': run['question'], 'answer': final.get('answer',''),
                'answer_scope': final.get('answer_scope'),
                'answer_sections': final.get('answer_sections'),
                'limitations': final.get('limitations') or [], 'review_required': bool(final.get('review_required')),
                'review_notes': final.get('review_notes') or [], 'citations': citations},
            'activities': [v for e in data['events'] if (v := activity(e)) is not None][-30:],
            'approvals': [{'approval_id': a['approval_id'],
                'tool_name': a['request'].get('tool') or tool_names.get(a['tool_call_id'], 'unknown'),
                'description': a['request'].get('description'),
                'context': a['request'].get('context') or {},
                'arguments': a['request'].get('arguments') or {}}
                for a in data['pending_approvals']],
            'artifacts': [{k: a.get(k) for k in ('artifact_id','kind','sha256','metadata')} | {
                'name': Path(a['path']).name} for a in data['artifacts']],
            'budget': {**(data.get('llm_budget') or {}),
                       'used_tokens': run.get('used_tokens',0), 'used_llm_calls': run.get('used_llm_calls',0),
                       'max_tokens': run['budget']['max_tokens']},
            'research': {'scope': run['metadata'].get('resolved_paper_sha256s') or [],
                'notebook_root': notes_root, 'design': run['metadata'].get('research_design'),
                'method': run['metadata'].get('method_card'),
                'requirements': run['metadata'].get('answer_request_history') or [],
                'workset': run['metadata'].get('reader_workset'),
                'review': run['metadata'].get('research_design_review'),
                'plan': run['metadata'].get('research_plan'),
                'engine': run['metadata'].get('engine','legacy'),
                'prototype': prototype},
            'active_evidence': data.get('active_evidence'),
            'can_resume': status == 'INTERRUPTED' and (not episode or episode['phase'] not in {'UNKNOWN','FAILED','PARTIAL','COMPLETED'})
                and (not prototype or prototype['phase'] not in {'UNKNOWN','FAILED','PARTIAL','COMPLETED','CANCELLED'}),
            'can_cancel': run['status'] not in TERMINAL,
            'can_follow_up': run['status'] in TERMINAL and not active and not unresolved_execution}

    def submit(self, payload):
        """Reserve before doing anything; uncertain submissions cannot be replayed blindly."""
        allowed = {'request_id','conversation_id','parent_run_id','question','action','paper_ids',
                   'source_ids','reference_ids','validation_run_id','binding_id','findings','decision',
                   'research_state','expected_revision','visual_policy','plan_revision','budget','recipe_id','candidate_id','bundle_id','retrieval_profile','research_case_id','delivery_mode','design_mode','prototype_mode','prototype_profile_id','prototype_constraints','source_replacements','prototype_literature','runtime_profile'}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError('Unknown submission fields')
        key = payload.get('request_id')
        if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,100}',key):
            raise ValueError('A unique request_id is required')
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.db.transaction() as conn:
            row = conn.execute('SELECT * FROM workbench_submissions WHERE request_id=?',(key,)).fetchone()
            if row:
                if row['fingerprint'] != fingerprint:
                    raise ValueError('request_id reused with different input')
                if row['state'] != 'COMPLETED':
                    raise RuntimeError('Previous submission is failed or uncertain; inspect its state before a new request')
                return {**json.loads(row['result_json']), 'reused': True}
            cid = payload.get('conversation_id')
            if cid and conn.execute("SELECT 1 FROM workbench_submissions WHERE conversation_id=? AND state='STARTED'",(cid,)).fetchone():
                raise RuntimeError('A submission in this conversation is still pending or uncertain')
            conn.execute('INSERT INTO workbench_submissions(request_id,fingerprint,state,created_at,conversation_id) VALUES(?,?,?,?,?)',
                         (key,fingerprint,'STARTED',utc_now(),cid))
        try:
            result = self._dispatch(payload)
        except Exception as exc:
            with self.db.transaction() as conn:
                conn.execute('UPDATE workbench_submissions SET state=?,result_json=? WHERE request_id=?',
                    ('FAILED',json.dumps({'error_type':type(exc).__name__}),key))
            raise
        with self.db.transaction() as conn:
            conn.execute('UPDATE workbench_submissions SET state=?,result_json=? WHERE request_id=?',
                         ('COMPLETED',json.dumps(result),key))
        if result.get('start'):
            rid = result['run_id']
            self.background.submit(rid, lambda: self.runtime.resume(rid))
        return result

    def _dispatch(self, value):
        action = value.get('action','answer')
        parent_id, cid = value.get('parent_run_id'), value.get('conversation_id')
        parent = self.owned(cid,parent_id) if parent_id and cid else None
        if bool(parent_id) != bool(cid):
            raise ValueError('Supply both conversation and parent run')
        if action == 'paper_active':
            from .active_evidence import ActiveEvidenceHarness
            result = ActiveEvidenceHarness(self.runtime).start(value.get('question'),value['request_id'],
                paper_ids=value.get('paper_ids'),budget=value.get('budget'),conversation_id=cid)
            run = result['run']
            return {'run_id':run['run_id'],'conversation_id':run['metadata']['conversation_id'],
                    'start':run['status']=='PENDING'}
        if action in {'revise_prototype_constraints', 'revise_prototype_goal', 'revise_prototype_sources'}:
            if parent is None or parent['metadata'].get('engine') != 'research_prototype_v1':
                raise ValueError('Select an owned prototype task')
            if self.background.active(parent_id):
                raise RuntimeError('Wait for the current prototype worker')
            from .prototype_harness import PrototypeHarness
            harness = PrototypeHarness(self.runtime)
            revise, argument = {'revise_prototype_goal': (harness.revise_goal, 'question'),
                                'revise_prototype_sources': (harness.revise_sources, 'source_replacements'),
                                'revise_prototype_constraints': (harness.revise_constraints, 'prototype_constraints')}[action]
            result = revise(parent_id, value.get(argument), expected_revision=value.get('expected_revision'))
            return {'run_id':parent_id,'conversation_id':cid,
                    'revision_applied':result['revision_applied'],
                    'start':bool(result['revision_applied'] and result['research_episode']['phase']=='READY')}
        if action == 'research_prototype':
            if parent is None:
                raise ValueError('Select an owned reading run')
            if self.background.active(parent_id) or parent['status'] not in TERMINAL:
                raise ValueError('Wait for the reading task to finish before handing off its evidence')
            from .prototype_harness import PrototypeHarness
            result = PrototypeHarness(self.runtime).start(value['question'], value['request_id'],
                parent_id, value.get('source_ids', []), conversation_id=cid, mode=value.get('prototype_mode', 'adaptive'),
                profile_id=value.get('prototype_profile_id'), constraints=value.get('prototype_constraints'),
                literature_enabled=value.get('prototype_literature', False),
                runtime_profile=value.get('runtime_profile',DEFAULT_PROFILE), budget=value.get('budget'),
                retrieval_profile=value.get('retrieval_profile', 'legacy'))
            return {'run_id': result['run']['run_id'], 'conversation_id': cid, 'start': True}
        if action == 'imported_diagnosis':
            if parent and self.background.active(parent_id):
                raise ValueError('Wait for the current worker before starting diagnosis')
            from .research_harness import ResearchHarness
            state = ResearchHarness(self.runtime).start_imported_diagnosis(value.get('bundle_id'),
                value['request_id'], conversation_id=cid, report_protocol='metric-facts-v3',
                research_case_id=value.get('research_case_id'), delivery_mode=value.get('delivery_mode', 'assisted'),runtime_profile=value.get('runtime_profile',DEFAULT_PROFILE))
            rid = state['run']['run_id']
            return {'run_id': rid, 'conversation_id': state['run']['metadata'].get('conversation_id', rid), 'start': True}
        if action == 'research_diagnosis':
            if parent is None or parent['metadata'].get('engine') != 'research_loop_v1':
                raise ValueError('Select an owned research experiment')
            if self.background.active(parent_id):
                raise ValueError('Wait for the research worker before starting diagnosis')
            from .research_harness import ResearchHarness
            state = ResearchHarness(self.runtime).start_diagnosis(parent_id, value.get('candidate_id'),
                value['request_id'], conversation_id=cid,runtime_profile=value.get('runtime_profile',DEFAULT_PROFILE))
            return {'run_id': state['run']['run_id'], 'conversation_id': cid, 'start': True}
        if action == 'research_episode':
            if parent and not self.turn(parent_id)['can_follow_up']:
                raise ValueError('Finish the current turn before starting research')
            from .research_harness import ResearchHarness
            state = ResearchHarness(self.runtime).start(goal=value.get('question'),
                recipe_id=value.get('recipe_id'), request_id=value['request_id'],
                source_ids=value.get('source_ids'), conversation_id=cid,unlimited_llm_usage=True)
            return {'run_id': state['run']['run_id'],
                    'conversation_id': state['run']['metadata']['conversation_id'], 'start': True}
        if action in {'approve_research', 'continue_research'}:
            if parent is None or parent['metadata'].get('engine') != 'research_task':
                raise ValueError('Select an owned research task')
            if self.background.active(parent_id):
                raise ValueError('Research worker is still active')
            with self.db.connect() as conn:
                newest = conn.execute(f'SELECT run_id FROM runs WHERE {CONVERSATION}=? '
                    'ORDER BY created_at DESC,run_id DESC LIMIT 1', (cid,)).fetchone()
            if newest['run_id'] != parent_id:
                raise ValueError('Conversation changed; continue the latest research task')
            from .research_task import ResearchTask
            ResearchTask(self.runtime, parent_id).approve(value.get('plan_revision'), value.get('budget'),
                continuing=action == 'continue_research')
            return {'run_id': parent_id, 'conversation_id': cid, 'start': True}
        if parent:
            if not self.turn(parent_id)['can_follow_up']:
                raise ValueError('Wait for or stop the current task before submitting another')
            with self.db.connect() as conn:
                last = conn.execute(f'SELECT run_id FROM runs WHERE {CONVERSATION}=? ORDER BY created_at DESC,run_id DESC LIMIT 1',
                                    (cid,)).fetchone()
                unfinished = conn.execute(f'''SELECT run_id FROM runs WHERE {CONVERSATION}=?
                    AND status NOT IN ('COMPLETED','FAILED','CANCELLED','BUDGET_EXHAUSTED')''',(cid,)).fetchall()
            if unfinished or not self.turn(last['run_id'])['can_follow_up']:
                raise ValueError('Conversation has unfinished work; stop or resolve it first')
            if last['run_id'] != parent_id and action in {'auto','answer','notebook','evidence_comparison','update_state'}:
                raise ValueError('Conversation changed; reload before sending')
        question = value.get('question','').strip()
        if action in {'auto','answer','notebook','evidence_comparison'}:
            if not question or len(question.encode()) > 16000:
                raise ValueError('Supply a question of at most 16000 UTF-8 bytes')
            scope = value.get('paper_ids')
            if scope is None and action in {'notebook','evidence_comparison'} and parent:
                scope = parent['metadata'].get('resolved_paper_sha256s')
            if scope is not None:
                if not isinstance(scope,list) or any(not isinstance(x,str) or not re.fullmatch('[a-f0-9]{64}',x) for x in scope):
                    raise ValueError('Invalid paper selection')
                store = getattr(self.runtime.search_adapter,'store',None)
                if scope and (store is None or len(store.paper_metadata(set(scope))) != len(set(scope))):
                    raise ValueError('Selected papers are not in the configured library')
            if action in {'notebook','evidence_comparison'} and not scope:
                raise ValueError('Select papers before creating a research inventory')
            visual = value.get('visual_policy','auto')
            if visual not in {'auto','off'}:
                raise ValueError('Unknown visual policy')
            meta = {'engine':'research_notebook' if action == 'notebook' else 'answer_first_text',
                    'visual_policy':visual, 'source':'workbench', 'submission_id':value['request_id']}
            if 'retrieval_profile' in value:
                meta['retrieval_profile'] = value['retrieval_profile']
            if action == 'evidence_comparison':
                meta['research_report_format'] = 'evidence_comparison_v1'
            if scope is not None:
                meta['paper_sha256s'] = scope
            if parent:
                meta['research_state_context'] = self.journal.state(cid)
            if parent and action in {'auto','answer'}:
                # Preserve the original engine when continuing a historical legacy conversation.
                if parent['metadata'].get('engine', 'legacy') == 'legacy':
                    meta.pop('engine')
                queued = self.runtime.enqueue_follow_up(parent_run_id=parent_id, question=question,
                                                       metadata=meta, budget=value.get('budget'))
            else:
                if parent:
                    meta.update(conversation_id=cid,parent_run_id=parent_id)
                queued = self.runtime.enqueue(question=question, metadata=meta, budget=value.get('budget'))
            if action in {'auto', 'answer', 'notebook', 'evidence_comparison'}:
                from .research_task import ResearchTask, needs_batches
                from .models import Budget
                if scope and (action == 'evidence_comparison' or not parent and (
                        action == 'notebook' or needs_batches(scope, Budget.from_dict(queued['run']['budget'])))):
                    ResearchTask(self.runtime, queued['run']['run_id']).prepare(scope)
                    return {'run_id': queued['run']['run_id'],
                            'conversation_id': queued['run']['metadata'].get('conversation_id') or queued['run']['run_id'],
                            'start': False}
                # Every automatic request can reach the structured route. Keywords
                # are not an authority to exclude an open-ended research goal.
                if action in {'auto', 'notebook'} and (not parent or parent['metadata'].get('engine','legacy') != 'legacy'):
                    catalog = self.materials(parent_id) if parent else {'readers':[],'validations':[],'profiles':[]}
                    # Selection metadata only: original contents are reloaded after checking IDs.
                    catalog['readers'] = [{**r,'sources':r['sources'][:12],'references':[]} for r in catalog['readers'][-4:]]
                    catalog['validations'] = catalog['validations'][-8:]
                    catalog['profiles'] = catalog['profiles'][:8]
                    self.db.update_run_metadata(queued['run']['run_id'],
                        {'engine':'research_session','research_action_catalog':catalog})
        elif action == 'update_state':
            if not parent:
                raise ValueError('Create a research conversation first')
            result = self.journal.revise(cid,value['research_state'],
                expected_revision=value['expected_revision'],source_run_id=parent_id)
            return {'run_id':parent_id,'conversation_id':cid,'start':False,'state':result}
        elif action == 'attach_reference':
            if not parent:
                raise ValueError('Read papers first')
            from .reference_methods import load_profile, PIN
            from .research_design import register_code_reference
            folder, manifest = load_profile(self.runtime.workspace,value['binding_id'])
            eid = register_code_reference(self.runtime,parent_id,
                source_url=manifest['source_url'],content=(folder/'upstream.py').read_text(encoding='utf-8'),
                license_url=PIN+'LICENSE',license_text=(folder/'LICENSE').read_text(encoding='utf-8'))
            return {'run_id':parent_id,'conversation_id':cid,'start':False,'reference_id':eid}
        elif action == 'design':
            if not parent:
                raise ValueError('Read papers first')
            from .research_design import enqueue_design
            queued = enqueue_design(self.runtime,parent_id,question=question,
                source_ids=value.get('source_ids'),reference_ids=value.get('reference_ids'),
                validation_run_id=value.get('validation_run_id'), design_mode=value.get('design_mode', 'legacy'), budget=value.get('budget'))
        elif action == 'validate':
            if not parent:
                raise ValueError('Read papers first')
            from .reference_methods import enqueue_validation
            queued = enqueue_validation(self.runtime,value['binding_id'],parent_run_id=parent_id)
        elif action == 'review':
            if not parent:
                raise ValueError('Select a completed draft')
            from .method_drafts import record_draft_review
            result = record_draft_review(self.runtime,parent_id,reviewer='local-workbench-user',
                decision=value['decision'],findings=value['findings'])
            return {'run_id':parent_id,'conversation_id':cid,'start':False,'review':result}
        elif action == 'prepare_comparison':
            if not parent:
                raise ValueError('Select a reviewed draft')
            from .paired_methods import prepare_comparison, describe_comparison
            binding = prepare_comparison(self.runtime,parent_id,seeds=(7,11,19),steps=2)
            self.db.update_run_metadata(parent_id,{'workbench_comparison_binding':binding})
            return {'run_id':parent_id,'conversation_id':cid,'start':False,
                    'binding_id':binding,'protocol':describe_comparison(self.runtime,binding)}
        elif action == 'compare':
            if not parent:
                raise ValueError('Select a reviewed draft')
            from .paired_methods import describe_comparison, enqueue_comparison
            if describe_comparison(self.runtime,value['binding_id'])['design_run_id'] != parent_id:
                raise ValueError('Protocol does not belong to this draft')
            queued = enqueue_comparison(self.runtime,value['binding_id'])
        else:
            raise ValueError('Unsupported research action; no code was executed')
        return {'run_id':queued['run']['run_id'],
                'conversation_id':queued['run']['metadata']['conversation_id'], 'start':True}

    def papers(self, query='', offset=0, limit=30):
        store = getattr(self.runtime.search_adapter,'store',None)
        if store is None:
            return {'items':[],'next_offset':None,'available':False}
        size = max(1,min(100,int(limit)))
        with store._connect() as conn:
            rows = conn.execute('''SELECT sha256,title,file_name,page_count FROM papers
                WHERE namespace=? AND active=1 AND (instr(lower(title),lower(?))>0 OR instr(lower(file_name),lower(?))>0)
                ORDER BY title,sha256 LIMIT ? OFFSET ?''',
                (store.namespace,str(query)[:200],str(query)[:200],size+1,max(0,int(offset)))).fetchall()
        return {'items':[dict(r) for r in rows[:size]], 'available':True,
                'next_offset':offset+size if len(rows)>size else None}

    def materials(self, run_id):
        run = self.db.get_run(run_id)
        cid = run['metadata'].get('conversation_id') or run_id
        with self.db.connect() as conn:
            ids = [r['run_id'] for r in conn.execute(f'SELECT run_id FROM runs WHERE {CONVERSATION}=? ORDER BY created_at,run_id',(cid,))]
        readers, validations = [], []
        for rid in ids:
            current = self.db.get_run(rid)
            if current['status'] != 'COMPLETED':
                continue
            sources = self.db.list_evidence(rid)
            original = []
            for e in sources:
                provenance = e.get('provenance',{})
                origin = provenance.get('origin')
                if origin == 'answer-first-source-material':
                    original.append({'id':e['evidence_id'],'preview':e['content'][:180],'page':e.get('page')})
                elif origin == 'active-paper-read-source' and current['metadata'].get('engine') == 'paper_active_v1':
                    original.append({'id':e['evidence_id'],'preview':e['content'][:180],'page':e.get('page'),
                        'source_uri':e['source_uri'],'paper_sha256':provenance['paper_sha256'],
                        'range':provenance['range'],'index_identity':provenance['index_identity'],
                        'assurance':'READ_SOURCE_ORIGIN_ONLY_NOT_ENTAILMENT'})
            references = [{'id':e['evidence_id'],'source_uri':e['source_uri']}
                for e in sources if e.get('provenance',{}).get('origin') == 'pinned-design-reference']
            if original:
                readers.append({'run_id':rid,'question':current['question'],'sources':original,'references':references})
            if current['metadata'].get('engine') == 'method_validation':
                validations.append({'run_id':rid,'binding_id':current['metadata'].get('binding_id')})
        from .reference_methods import describe_profile
        profiles = []
        root = self.runtime.workspace/'method-profiles'
        for manifest in sorted(root.glob('*/manifest.json')):
            try:
                profiles.append(describe_profile(self.runtime.workspace,manifest.parent.name))
            except (ValueError,OSError,KeyError,TypeError):
                continue
        return {'readers':readers,'validations':validations,'profiles':profiles,
                'comparison_binding':run['metadata'].get('workbench_comparison_binding')}

    def artifact(self, run_id, artifact_id):
        artifacts = self.db.list_artifacts(run_id)
        row = next((a for a in artifacts if a['artifact_id'] == artifact_id),None)
        if row is None:
            raise KeyError('Unknown artifact')
        path = Path(row['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(self.runtime.workspace.resolve()):
            raise PermissionError('Artifact is outside the research workspace')
        if path.stat().st_size > 1024*1024:
            raise ValueError('Artifact is too large for inline preview')
        raw = path.read_bytes()
        if row.get('sha256') and hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Artifact changed since it was registered')
        if path.suffix.lower() not in {'.json','.md','.txt','.log','.patch','.py','.csv'}:
            raise ValueError('Unsupported preview format')
        return {'name':path.name,'kind':row['kind'],'content':raw.decode('utf-8'),
                'sha256':hashlib.sha256(raw).hexdigest()}
