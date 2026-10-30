"""A budget-approved, serial research task; batches never receive fresh budgets.

Task checkpoints are separate from scientific conclusions. Saved source material
is reloadable, while only source-audited report sections may be published.
"""
import hashlib
import json
import math
import dataclasses
import os
import time
from contextlib import contextmanager

from .models import Budget, RunStatus, ResearchOutcome
from .llm_runtime import LLMBudgetExhausted
from .durable_reader import DurableResearchModel, persist_reader_evidence, publish_reader_answer
from .providers import LLMResponse


class UncertainBatch(RuntimeError):
    """A possibly paid operation has no durable response; do not replay it."""


class ProviderRequestRejected(RuntimeError):
    """An explicit billing/auth rejection is known, unlike a lost response."""


@contextmanager
def task_lock(workspace, run_id):
    """OS-owned lock is released on process death; never a time-based lease steal."""
    folder = workspace / '.research-task-locks'
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(run_id.encode()).hexdigest() + '.lock'
    with (folder / name).open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        try:
            yield acquired
        finally:
            if acquired:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def elapsed_seconds(run):
    meta = run['metadata']
    started = meta.get('research_active_since')
    return float(meta.get('research_active_seconds', 0)) + (max(0, time.time() - started) if started else 0)


class CheckpointModel:
    """Checkpoint individual responses so a completed Writer survives a paused audit."""
    def __init__(self, task, stage, reserve_class='general', identity_scope=None):
        self.task, self.stage, self.reserve_class = task, stage, reserve_class
        self.identity_scope = identity_scope

    def for_repair(self):
        return CheckpointModel(self.task, 'json_repair', 'repair', self.identity_scope)

    def complete(self, messages, *, max_tokens, json_mode=False, thinking_enabled=False, before_attempt=None):
        identity = [self.stage, self.identity_scope, messages, max_tokens, json_mode, thinking_enabled]
        key = 'model-' + hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        def invoke():
            response = DurableResearchModel(self.task.runtime.llm_runtime, self.task.run_id,
                stage=self.stage, reserve_class=self.reserve_class).complete(messages, max_tokens=max_tokens,
                json_mode=json_mode, thinking_enabled=thinking_enabled, before_attempt=before_attempt)
            return dataclasses.asdict(response)
        return LLMResponse(**self.task.operation(key, 'model', identity, invoke))


VERSION = 'multi-paper-task-v9-separated-audit-coverage'


def needs_batches(scope, budget):
    # One global search plus search/outline/section-read per paper, leaving four
    # tool operations for missing evidence. This is a preflight, not a cost claim.
    return len(set(scope)) > 1 and 1 + 3 * len(set(scope)) + 4 > budget.max_tool_calls


class ResearchTask:
    def __init__(self, runtime, run_id):
        self.runtime, self.db, self.run_id = runtime, runtime.db, run_id
        self.db.get_run(run_id)
        with self.db.transaction() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS research_tasks(
                run_id TEXT PRIMARY KEY REFERENCES runs(run_id), payload TEXT NOT NULL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS research_task_batches(
                run_id TEXT NOT NULL REFERENCES runs(run_id), batch_id TEXT NOT NULL,
                kind TEXT NOT NULL, position INTEGER NOT NULL, state TEXT NOT NULL,
                payload TEXT NOT NULL, result TEXT, PRIMARY KEY(run_id,batch_id))''')

    def load(self):
        with self.db.connect() as c:
            row = c.execute('SELECT payload FROM research_tasks WHERE run_id=?', (self.run_id,)).fetchone()
        return json.loads(row['payload']) if row else None

    def prepare(self, scope, *, dimensions=None):
        previous = self.load()
        if previous:
            return self.view()
        run = self.db.get_run(self.run_id)
        from .request_scope import resolve_request_scope
        edits = resolve_request_scope(run['question'],
            lambda text: self.runtime._resolve_explicit_source_scope(text, None, None))
        selected = set(scope)
        if set(edits['included']) - selected:
            raise ValueError('Named papers are outside the selected research scope')
        hashes = sorted((set(edits['included']) or selected) - set(edits['excluded']))
        if not hashes:
            raise ValueError('Select papers before preparing a research task')
        dimensions = dimensions or [run['question']]
        if (not isinstance(dimensions, list) or not 1 <= len(dimensions) <= 6
                or any(not isinstance(x, str) or not x.strip() for x in dimensions)):
            raise ValueError('Supply one to six research dimensions')
        report_format = run['metadata'].get('research_report_format', 'section_report_v1')
        if report_format not in {'section_report_v1', 'evidence_comparison_v1'}:
            raise ValueError('Unknown research report format')
        plan = {'version': VERSION, 'question': run['metadata'].get('resolved_question') or run['question'],
                'report_format': report_format,
                'original_question': run['question'], 'papers': hashes,
                'requirements': run['metadata'].get('entry_request_history') or [],
                'dimensions': dimensions, 'protocol': self.runtime.text_reader_protocol}
        revision = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        suggested = Budget(max_tool_calls=max(20, 6 * len(hashes) + 8),
                           max_llm_calls=max(20, len(hashes) * 10 + 12),
                           max_tokens=max(100000, len(hashes) * 180000 + 50000),
                           max_seconds=max(3600, len(hashes) * 180))
        store = getattr(self.runtime.search_adapter, 'store', None)
        names = {r['sha256']: r.get('title') or r.get('file_name') or r['sha256']
                 for r in store.paper_metadata(set(hashes))} if store else {}
        task = {**plan, 'plan_revision': revision, 'state': 'AWAITING_BUDGET',
                'suggested_budget': suggested.to_dict(), 'approved_budget': None,
                'pause_reason': '请确认总预算；这不会授权实验或训练。', 'sections': [],
                'comparison': None,
                'budget_basis': ('按每篇约两个证据工作集、字段审核和有界修订预估；未完成消费校准。维度共享提取记录。'
                                 if report_format == 'evidence_comparison_v1' else
                                 '章节报告的分组与用量依原文长度和维度变化；建议累计上限尚未校准。'),
                'active_seconds': 0.0, 'coverage': [
                    {'paper_sha256': sha, 'title': names.get(sha, sha), 'state': 'PENDING', 'evidence_count': 0, 'gaps': []}
                    for sha in hashes]}
        with self.db.transaction() as c:
            c.execute('INSERT INTO research_tasks VALUES(?,?)', (self.run_id, json.dumps(task, ensure_ascii=False)))
        self.db.update_run_metadata(self.run_id, {'engine': 'research_task',
            'resolved_paper_sha256s': hashes, 'research_task_version': VERSION})
        self.db.update_run_status(self.run_id, RunStatus.WAITING_APPROVAL)
        self.db.emit_event(self.run_id, 'research.task.planned', {'papers': len(hashes), 'plan_revision': revision})
        return self.view()

    def view(self):
        task = self.load()
        if task is None:
            return None
        run = self.db.get_run(self.run_id)
        sections = [{k: section[k] for k in ('dimension', 'group', 'papers')} | {
            'status': section['report']['status'], 'answer_scope': section['report'].get('answer_scope')}
            for section in task['sections']]
        return {**task, 'sections': sections, 'active_seconds': elapsed_seconds(run), 'usage': {key: run[key] for key in
                ('used_tokens', 'used_llm_calls', 'used_tool_calls')},
                'budget_is_experiment_authorization': False}

    def approve(self, revision, budget, *, continuing=False):
        """Atomically authorize a total ceiling, never reset already incurred usage."""
        self.check_unknown()
        self.check_protocol()
        mutable = {'max_tool_calls', 'max_llm_calls', 'max_tokens', 'max_seconds', 'max_steps'}
        with self.db.transaction() as c:
            row = c.execute('SELECT payload FROM research_tasks WHERE run_id=?', (self.run_id,)).fetchone()
            if row is None:
                raise ValueError('Not a research task')
            task = json.loads(row['payload'])
            run = c.execute('SELECT * FROM runs WHERE run_id=?', (self.run_id,)).fetchone()
            if revision != task['plan_revision']:
                raise ValueError('Research plan changed; reload before confirming')
            allowed = {'PAUSED_BUDGET', 'PAUSED_CANCELLED'} if continuing else {'AWAITING_BUDGET'}
            if task['state'] not in allowed:
                raise ValueError('Research task is not awaiting this confirmation')
            old = json.loads(run['budget_json'])
            if not isinstance(budget, dict) or set(budget) - set(old):
                raise ValueError('Invalid total budget')
            merged = {**old, **budget}
            for key, value in merged.items():
                if key not in mutable:
                    if value != old[key]:
                        raise ValueError('Research approval cannot change risk or reserve policy')
                elif (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                      or (key != 'max_seconds' and type(value) is not int)):
                    raise ValueError('Budget limits must be finite positive numbers')
                elif continuing and value < old[key]:
                    raise ValueError('Continuation must preserve total allowances')
            for field, used in [('max_llm_calls', 'used_llm_calls'), ('max_tokens', 'used_tokens'),
                                ('max_tool_calls', 'used_tool_calls')]:
                if merged[field] <= run[used]:
                    raise ValueError('Continuation needs remaining budget; usage is not reset')
            if continuing and merged['max_seconds'] <= elapsed_seconds(self.db.get_run(self.run_id)):
                raise ValueError('Continuation needs remaining active time')
            task.update(state='READY', approved_budget=merged, pause_reason=None)
            task['plan_revision'] = hashlib.sha256(json.dumps([revision, merged], sort_keys=True).encode()).hexdigest()
            c.execute('UPDATE research_tasks SET payload=? WHERE run_id=?',
                      (json.dumps(task, ensure_ascii=False), self.run_id))
            c.execute("UPDATE runs SET budget_json=?,status='PENDING',error=NULL WHERE run_id=?",
                      (json.dumps(merged), self.run_id))
        self.db.emit_event(self.run_id, 'research.task.budget_approved', {'budget': merged})
        self.db.update_run_metadata(self.run_id, {'cancel_requested': False})
        return self.view()

    def check_protocol(self):
        task = self.load()
        if task and (task['version'] != VERSION or task['protocol'] != self.runtime.text_reader_protocol):
            raise ValueError('Research protocol changed; create a new plan before running')

    def save(self, **updates):
        with self.db.transaction() as c:
            value = json.loads(c.execute('SELECT payload FROM research_tasks WHERE run_id=?',
                                        (self.run_id,)).fetchone()['payload'])
            value.update(updates)
            c.execute('UPDATE research_tasks SET payload=? WHERE run_id=?',
                      (json.dumps(value, ensure_ascii=False), self.run_id))

    def guard(self):
        run = self.db.get_run(self.run_id)
        if run['status'] == 'CANCELLED':
            raise InterruptedError('Research task cancelled')
        if elapsed_seconds(run) >= run['budget']['max_seconds']:
            raise LLMBudgetExhausted('Research active-time budget exhausted')
        if not self.load().get('approved_budget'):
            raise ValueError('Research budget has not been approved')
        with self.db.connect() as c:
            rejected = c.execute("SELECT result FROM research_task_batches WHERE run_id=? AND state='FAILED_KNOWN' LIMIT 1",
                                 (self.run_id,)).fetchone()
        if rejected:
            status = json.loads(rejected['result'])['http_status']
            raise ProviderRequestRejected(f'模型供应商明确拒绝请求（HTTP {status}）；请检查额度或授权。记录已保存，不自动重试。')
        return run

    def operation(self, key, kind, payload, operation):
        """Local reads may be retried after a crash; paid unknowns may not."""
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with self.db.transaction() as c:
            row = c.execute('SELECT * FROM research_task_batches WHERE run_id=? AND batch_id=?',
                            (self.run_id, key)).fetchone()
            if row:
                if row['payload'] != encoded:
                    raise ValueError('Batch inputs changed; create a new research plan')
                if row['state'] == 'COMPLETED':
                    return json.loads(row['result'])
                if row['state'] in {'STARTED', 'UNKNOWN'} and kind == 'model':
                    raise UncertainBatch('Previous model response is unknown; not repeated')
            c.execute('''INSERT INTO research_task_batches VALUES(?,?,?,0,'PENDING',?,NULL)
                ON CONFLICT(run_id,batch_id) DO NOTHING''', (self.run_id, key, kind, encoded))
        run = self.guard()
        if kind == 'tool':
            # Reserve a real operation, not a batch-sized loophole in tool accounting.
            with self.db.transaction() as c:
                changed = c.execute('''UPDATE runs SET used_tool_calls=used_tool_calls+1
                    WHERE run_id=? AND used_tool_calls<? AND status!='CANCELLED' ''',
                    (self.run_id, run['budget']['max_tool_calls'])).rowcount
                if not changed:
                    raise LLMBudgetExhausted('Research tool budget exhausted')
        with self.db.transaction() as c:
            c.execute("UPDATE research_task_batches SET state='STARTED' WHERE run_id=? AND batch_id=?",
                      (self.run_id, key))
        try:
            result = operation()
        except Exception as exc:
            now = self.db.get_run(self.run_id)
            state = 'UNKNOWN' if kind == 'model' and now['used_llm_calls'] > run['used_llm_calls'] else 'PENDING'
            rejection = getattr(exc, 'status_code', None) if kind == 'model' else None
            if rejection in {401, 402, 403}:
                state = 'FAILED_KNOWN'
            with self.db.transaction() as c:
                c.execute('UPDATE research_task_batches SET state=?,result=? WHERE run_id=? AND batch_id=?',
                          (state, json.dumps({'http_status': rejection}) if state == 'FAILED_KNOWN' else None,
                           self.run_id, key))
            if state == 'FAILED_KNOWN':
                self.guard()  # Report the explicit provider rejection, not an unknown response.
            raise
        with self.db.transaction() as c:
            c.execute("UPDATE research_task_batches SET state='COMPLETED',result=? WHERE run_id=? AND batch_id=?",
                      (json.dumps(result, ensure_ascii=False), self.run_id, key))
        self.guard()
        return result

    def tool(self, name, payload, operation):
        key = name + '-' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.operation(key, 'tool', payload, operation)

    def read_paper(self, sha):
        from .section_router import select_sections, read_selected_sections, structural_overviews
        from .request_scope import automatic_table_targets
        from .reader_visual import collect_visual_evidence
        from .research_notebook import ResearchNotebook
        task = self.load()
        query = task['question']
        adapter, store = self.runtime.search_adapter, getattr(self.runtime.search_adapter, 'store', None)
        rows = self.tool('paper.search', [sha, query],
                         lambda: adapter.search(query, top_k=4, paper_sha256s={sha}))
        rows = list(rows)
        gaps = []
        if store is not None and hasattr(store, 'paper_outline'):
            outline = self.tool('paper.outline', [sha], lambda: store.paper_outline(sha))
            route = select_sections(query, outline, CheckpointModel(self, 'research_sections'),
                                    original_question=task['question'])
            # Routers turn exceptions into reports; restore budget/unknown semantics.
            if route['status'] == 'FAILED':
                self.check_unknown()
                if route.get('error_type') == 'LLMBudgetExhausted':
                    raise LLMBudgetExhausted('Section navigation paused for budget')
                gaps.append('章节选择未完成，仅保留检索材料。')
            else:
                ids = list(dict.fromkeys(structural_overviews(outline) + route['section_ids']))
                extra, omitted = self.tool('paper.sections', [sha, ids],
                    lambda: read_selected_sections(store, sha, outline, ids))
                rows.extend({**r, 'paper_sha256': sha} for r in extra)
                if omitted:
                    gaps.append(f'{len(omitted)} 个章节片段未进入当前阅读窗口，可按来源重载。')
        if self.db.get_run(self.run_id)['metadata'].get('visual_policy') != 'off':
            targets, target_gaps = automatic_table_targets(query, set(task['papers']),
                lambda text: self.runtime._resolve_explicit_source_scope(text, None, None))
            if target_gaps:
                gaps.append('部分图表身份尚未解析，不能据此比较数值。')
            own = [t for t in targets if t.get('paper_sha256') == sha]
            if own:
                # Tool invocation itself already accounts for its tool use and
                # vision calls. Cache its serializable evidence, not ToolOutcome.
                visual = self.operation('visual-' + sha, 'model', [sha, own], lambda:
                    collect_visual_evidence(self.runtime, self.run_id, query, own, {sha},
                                            lambda name, action: action()))
                rows = list(visual) + rows
                if not visual:
                    gaps.append('指定图表未取得可核对转录。')
        checked = []
        book = ResearchNotebook(self.db, self.run_id)
        seen = set()
        for row in rows:
            uri = row.get('source_uri', '')
            if not uri.startswith(f'paper://{sha}/') or row.get('paper_sha256', sha) != sha:
                raise ValueError('Research retrieval returned a foreign source')
            identity = (uri, row.get('content'))
            if identity in seen:
                continue
            seen.add(identity)
            book.add_source(uri, row['content'])
            checked.append({**row, 'evidence_id': row.get('evidence_id') or 'e_' + hashlib.sha256(
                json.dumps(identity, ensure_ascii=False).encode()).hexdigest()})
        evidence = persist_reader_evidence(self.db, self.run_id, checked)
        return {'evidence': evidence, 'gaps': gaps, 'paper_sha256': sha}

    def check_unknown(self):
        with self.db.connect() as c:
            row = c.execute("SELECT 1 FROM research_task_batches WHERE run_id=? AND kind='model' "
                            "AND state IN ('STARTED','UNKNOWN')", (self.run_id,)).fetchone()
        if row:
            raise UncertainBatch('Model response is unknown; inspect the saved request before continuing')

    def report_groups(self, evidence):
        """Persist bounded originals; do not substitute summaries for long passages."""
        from .context_window import window_limit, split_text
        limit = min(18000, max(512, (window_limit(self.runtime.llm_runtime.model) - 18000) // 4))
        groups, group, size = [], [], 0
        for row in evidence:
            # Split only text sources, preserving exact URI/character provenance.
            pieces = list(split_text(row['content'], max(256, limit // 2)))
            if len(pieces) > 1 and row.get('evidence_type', 'TEXT') != 'TEXT':
                self.save(pause_reason='图表转录大于当前审核窗口，需要缩小图表范围。')
                raise ValueError('Visual transcription cannot be sliced as ordinary text')
            for start, end, text in pieces:
                item = row if len(pieces) == 1 else persist_reader_evidence(self.db, self.run_id, [{
                    **row, 'content': text, 'source_uri': row['source_uri'] + f'#chars={start}:{end}',
                    'provenance': {**(row.get('provenance') or {}), 'original_evidence_id': row['evidence_id'],
                                   'start_char': start, 'end_char': end}}])[0]
                required = len(json.dumps(item, ensure_ascii=False).encode())
                # The Writer supports at most 12 facts; leave room for conditions
                # and limitations instead of asking one response to cover 50 papers.
                if group and (size + required > limit or len(group) >= 6):
                    groups.append(group)
                    group, size = [], 0
                group.append(item)
                size += required
        if group:
            groups.append(group)
        return groups

    def execute(self):
        from .answer_audit import answer_then_audit
        from .reader_recovery import retrieve_gap_evidence, known_revision_spans, revision_feedback
        task = self.load()
        if task.get('report_format') == 'evidence_comparison_v1':
            return self.execute_comparison()
        evidence, coverage = [], []
        for sha in task['papers']:
            self.guard()
            result = self.operation('read-' + sha, 'local', [sha, task['question']], lambda: self.read_paper(sha))
            evidence.extend(result['evidence'])
            coverage.append({**task['coverage'][len(coverage)], 'state': 'READ' if result['evidence'] else 'NO_EVIDENCE',
                             'evidence_count': len(result['evidence']), 'gaps': result['gaps']})
            self.save(coverage=coverage + task['coverage'][len(coverage):])
            self.db.emit_event(self.run_id, 'research.task.progress',
                              {'completed': len(coverage), 'total': len(task['papers'])})
        sections = []
        for dimension in task['dimensions']:
            for index, group in enumerate(self.report_groups(evidence)):
                scope = sorted({r['source_uri'].split('/')[2] for r in group})
                question = (task['question'] + '\n本章节关注：' + dimension +
                    '\n仅对所给原文覆盖的论文作答；不代表其他论文。缺失条件明确保留，不凭摘要补足。')
                model = CheckpointModel(self, 'research_report', 'synthesis')
                report = answer_then_audit(question, group, model, original_question=task['question'],
                    request_history=task['requirements'], relation_policy=True, question_scope_policy=True)
                self.check_unknown()
                if report.get('error_type') == 'LLMBudgetExhausted':
                    raise LLMBudgetExhausted('Report paused for budget; completed responses are retained')
                revision = None
                if known_revision_spans(report, group):
                    first = report
                    revised = answer_then_audit(question, group, model, original_question=task['question'],
                        request_history=task['requirements'], relation_policy=True, question_scope_policy=True,
                        revision_feedback=revision_feedback(first, group))
                    self.check_unknown()
                    if revised.get('error_type') == 'LLMBudgetExhausted':
                        raise LLMBudgetExhausted('Local revision paused; completed responses are retained')
                    accepted = revised.get('status') == 'REVIEW_REQUIRED' and revised.get('direct_answer_supported')
                    revision = {'initial_report': first, 'revision_report': revised, 'accepted': bool(accepted)}
                    if accepted:
                        report = revised
                # One concrete gap lookup per section; no recursive blind research loop.
                gaps = (report.get('audit') or {}).get('missing_aspects') or report.get('limitations') or []
                if revision is None and report.get('answer_scope') != 'MODEL_REVIEWED' and gaps:
                    gap_query = ' '.join(str(x) for x in gaps)[:400]
                    new = self.tool('paper.gap', [scope, gap_query], lambda:
                        retrieve_gap_evidence(self.runtime.search_adapter, question, gap_query, scope, group))
                    if new:
                        originals = persist_reader_evidence(self.db, self.run_id, new)
                        # Further bounded sections retain original evidence, not a synthetic summary.
                        for addition in self.report_groups(originals):
                            extra = answer_then_audit(question, addition, model,
                                original_question=task['original_question'], request_history=task['requirements'],
                                relation_policy=True, question_scope_policy=True)
                            self.check_unknown()
                            if extra.get('error_type') == 'LLMBudgetExhausted':
                                raise LLMBudgetExhausted('Supplement paused for budget')
                            sections.append({'dimension': dimension, 'group': index + 1,
                                             'papers': scope, 'report': extra, 'supplement': True})
                sections.append({'dimension': dimension, 'group': index + 1, 'papers': scope, 'report': report,
                                 **({'local_revision': revision} if revision is not None else {})})
                self.save(sections=sections)
                self.publish(sections, complete=False)
        self.publish(sections, complete=True)
        failed = any(s['report']['status'] in {'FAILED', 'ROUTING_FAILED'} for s in sections)
        self.save(state='FAILED' if failed else 'FINISHED',
                  pause_reason='部分章节生成或审核失败，已保留成功章节。' if failed else None)
        self.db.update_run_status(self.run_id, RunStatus.FAILED if failed else RunStatus.COMPLETED)

    def publish(self, sections, *, complete):
        self.guard()
        task = self.load()
        if task.get('report_format') == 'evidence_comparison_v1':
            return self.publish_comparison(complete=complete)
        texts, limitations = [], []
        for section in sections:
            report = section['report']
            if report.get('answer'):
                title = section['dimension'][:160]
                texts.append(f"### {title} · 材料组 {section['group']}\n\n" + report['answer'])
            limitations.extend(report.get('limitations') or [])
        if not complete:
            limitations.insert(0, '任务尚未结束；以下仅包含已审核的阶段章节。')
        if any(x['state'] != 'READ' or x['gaps'] for x in task['coverage']):
            limitations.append('部分论文或材料仍有缺口，请查看逐篇覆盖表。')
        limitations.append('分组章节只覆盖各自引用的原文；已处理论文不代表所有结论已独立核验。')
        result = {'status': 'REVIEW_REQUIRED' if texts else 'NO_EVIDENCE',
            'answer': '\n\n'.join(texts), 'limitations': list(dict.fromkeys(limitations)),
            'answer_scope': 'PARTIAL', 'direct_answer_supported': bool(texts)}
        publish_reader_answer(self.db, self.run_id, result)
        self.db.update_run_outcome(self.run_id, ResearchOutcome.PARTIAL if texts else ResearchOutcome.UNRESOLVED)

    def execute_comparison(self):
        from .answer_audit import answer_then_audit
        from .paper_comparison import reviewed_records, assemble_comparison
        from .reader_recovery import known_revision_spans, revision_feedback
        task = self.load()
        records, coverage, sections = [], [], []
        for position, sha in enumerate(task['papers']):
            self.guard()
            result = self.operation('read-' + sha, 'local', [sha, task['question']], lambda: self.read_paper(sha))
            paper = {**task['coverage'][position], 'state': 'READ' if result['evidence'] else 'NO_EVIDENCE',
                     'evidence_count': len(result['evidence']), 'gaps': list(result['gaps'])}
            for index, group in enumerate(self.report_groups(result['evidence'])):
                question = (task['question'] + '\n逐篇整理所给论文的实验事实与方法模块。保留不同实验条件，'
                            '不把相关工作写成本文实验，不根据这些局部片段推断全库排名。')
                model = CheckpointModel(self, 'comparison_extract', 'synthesis', [sha, index])
                report = answer_then_audit(question, group, model, original_question=task['question'],
                    request_history=task['requirements'], relation_policy=True, question_scope_policy=True,
                    draft_contract='comparison-v2')
                self.check_unknown()
                if report.get('error_type') == 'LLMBudgetExhausted':
                    raise LLMBudgetExhausted('Comparison paused; completed extraction/audit responses are retained')
                revision = None
                if known_revision_spans(report, group):
                    revised = answer_then_audit(question, group, model, original_question=task['question'],
                        request_history=task['requirements'], relation_policy=True, question_scope_policy=True,
                        draft_contract='comparison-v2', revision_feedback=revision_feedback(report, group))
                    self.check_unknown()
                    if revised.get('error_type') == 'LLMBudgetExhausted':
                        raise LLMBudgetExhausted('Comparison revision paused; responses are retained')
                    accepted = revised.get('status') == 'REVIEW_REQUIRED' and revised.get('direct_answer_supported')
                    revision = {'initial_report': report, 'revision_report': revised, 'accepted': bool(accepted)}
                    if accepted:
                        report = revised
                found = reviewed_records(report, sha, paper['title'])
                records.extend(found)
                paper['gaps'].extend((report.get('draft') or {}).get('comparison_format_gaps', []))
                if not found or not any(r['cells'] for r in found):
                    paper['gaps'].append(f'材料组 {index + 1} 未形成通过字段与联合归属审核的记录。')
                paper['gaps'].extend(report.get('limitations') or [])
                sections.append({'dimension': '逐篇证据记录', 'group': index + 1, 'papers': [sha], 'report': report,
                                 **({'local_revision': revision} if revision else {})})
                # Keep reviewed group results visible even when the next group
                # pauses. Completed model responses remain checkpointed as well.
                pending = coverage + [paper] + task['coverage'][position + 1:]
                self.save(coverage=pending, sections=sections,
                          comparison=assemble_comparison(records, pending))
                self.publish_comparison(complete=False)
            coverage.append(paper)
            pending = coverage + task['coverage'][len(coverage):]
            comparison = assemble_comparison(records, pending)
            self.save(coverage=pending, sections=sections, comparison=comparison)
            self.publish_comparison(complete=False)
            self.db.emit_event(self.run_id, 'research.task.progress',
                               {'completed': len(coverage), 'total': len(task['papers']),
                                'reviewed_records': len(comparison['records'])})
        self.publish_comparison(complete=True)
        failed = any(s['report']['status'] in {'FAILED', 'ROUTING_FAILED'} for s in sections)
        self.save(state='FAILED' if failed else 'FINISHED',
                  pause_reason='部分证据记录提取或审核失败，已保留通过的记录。' if failed else None)
        self.db.update_run_status(self.run_id, RunStatus.FAILED if failed else RunStatus.COMPLETED)

    def publish_comparison(self, *, complete):
        from .paper_comparison import comparison_markdown, assemble_comparison
        self.guard()
        task = self.load()
        comparison = task.get('comparison') or assemble_comparison([], task['coverage'])
        limitations = ['表格字段与联合归属经过模型审核；尚未经过独立事实验收。',
                       '已记录条件相同不保证所有实验条件相同；缺失条件不推断，不自动生成性能排名。']
        if not complete:
            limitations.insert(0, '任务尚未结束；当前比较表只包含已处理材料中的已审核记录。')
        if comparison['conflicts']:
            limitations.append('部分同条件结果存在冲突，已排除出条件对照组。')
        result = {'status': 'REVIEW_REQUIRED' if comparison['records'] else 'NO_EVIDENCE',
                  'answer': comparison_markdown(comparison) if comparison['records'] or complete else '', 'limitations': limitations,
                  'answer_scope': 'PARTIAL', 'direct_answer_supported': bool(comparison['records'])}
        publish_reader_answer(self.db, self.run_id, result)
        self.db.update_run_outcome(self.run_id, ResearchOutcome.PARTIAL if comparison['records'] else ResearchOutcome.UNRESOLVED)


def resume_research_task(runtime, run_id):
    task = ResearchTask(runtime, run_id)
    current = task.load()
    if current is None or current['state'] in {'AWAITING_BUDGET', 'FINISHED', 'PAUSED_BUDGET', 'PAUSED_CANCELLED', 'UNKNOWN'}:
        return runtime.status(run_id)
    with task_lock(runtime.workspace, run_id) as acquired:
        if not acquired:
            return runtime.status(run_id)
        try:
            task.check_unknown()
            task.check_protocol()
            task.guard()
            task.save(state='RUNNING', pause_reason=None)
            runtime.db.update_run_metadata(run_id, {'research_active_since': time.time()})
            runtime.db.update_run_status(run_id, RunStatus.RUNNING)
            task.execute()
        except LLMBudgetExhausted:
            task.save(state='PAUSED_BUDGET', pause_reason='累计预算不足，已保存批次；追加总额度后继续。')
            runtime.db.update_run_status(run_id, RunStatus.BUDGET_EXHAUSTED)
        except (UncertainBatch,):
            task.save(state='UNKNOWN', pause_reason='上次模型请求结果未知，禁止自动重试；已有响应和材料保留。')
            runtime.db.update_run_status(run_id, RunStatus.WAITING_APPROVAL)
        except ProviderRequestRejected as exc:
            task.save(state='FAILED', pause_reason=str(exc))
            runtime.db.update_run_status(run_id, RunStatus.FAILED, error=str(exc))
        except Exception as exc:
            if runtime.db.get_run(run_id)['status'] == 'CANCELLED':
                task.save(state='PAUSED_CANCELLED', pause_reason='任务已停止；未开始的批次保留。')
            else:
                try:
                    task.check_unknown()
                except UncertainBatch:
                    task.save(state='UNKNOWN', pause_reason='模型请求结果未知，未重试。')
                    runtime.db.update_run_status(run_id, RunStatus.WAITING_APPROVAL)
                else:
                    task.save(state='FAILED', pause_reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
                    runtime.db.update_run_status(run_id, RunStatus.FAILED, error=type(exc).__name__)
        finally:
            run = runtime.db.get_run(run_id)
            runtime.db.update_run_metadata(run_id, {'research_active_seconds': elapsed_seconds(run),
                                                   'research_active_since': None})
    return runtime.status(run_id)
