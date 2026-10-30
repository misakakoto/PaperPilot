"""Bounded cross-page synthesis from original evidence, never prior answers."""
import json

from .durable_reader import DurableResearchModel, run_durable_reader
from .models import Budget


def evidence_groups(evidence, max_bytes=24000):
    """Every whole passage occurs once, or no partial plan is returned."""
    groups, current = [], []
    for row in evidence:
        if len(json.dumps([row], ensure_ascii=False).encode('utf-8')) > max_bytes:
            return None
        trial = current + [row]
        if current and len(json.dumps(trial, ensure_ascii=False).encode('utf-8')) > max_bytes:
            groups.append(current)
            current = [row]
        else:
            current = trial
    if current:
        groups.append(current)
    return groups


def _cross_group_review(runtime, run_id, manifest, groups, task_text, guard):
    # Stream the Cartesian comparisons; do not materialize a quadratic plan.
    count = sum(map(len, groups))
    total = (count * (count - 1) - sum(len(g) * (len(g) - 1) for g in groups)) // 2
    results, oversized = [], []
    stopped = None
    pairs = ((a, b) for i, left in enumerate(groups) for right in groups[i + 1:]
             for a in left for b in right)
    for index, pair in enumerate(pairs):
        guard()
        # Fixed safety ceiling bounds work and checkpoint size even with a large budget.
        if index >= 12:
            stopped = 'PAIR_LIMIT'
            break
        ids = [row['evidence_id'] for row in pair]
        if len(json.dumps(pair, ensure_ascii=False).encode('utf-8')) > 24000:
            oversized.append(ids)
            continue
        current = runtime.db.get_run(run_id)
        key = f'workset_cross_group_{index}'
        if (key not in current['metadata'] and current['used_llm_calls'] + 2 >
                Budget.from_dict(current['budget']).max_llm_calls):
            stopped = 'CALL_BUDGET'
            break
        report = run_durable_reader(runtime.db, run_id, list(pair),
            DurableResearchModel(runtime.llm_runtime, run_id, stage='workset_cross_group'),
            protocol_id=manifest['protocol'], finalize=False, checkpoint_key=key,
            question=('对照以下来自不同组的两段原文，分别引用双方证据。'
                      '检查数据集、划分、指标、实验条件和结论是否可比；'
                      '区分真正矛盾、条件不同、信息不足与无关内容，不强行制造冲突。'
                      '仅报告这一对原文的发现，不给全库排名，不声称实验已执行。研究要求：\n'
                      + task_text))
        guard()
        results.append({'evidence_ids': ids, 'report': report})
        if report.get('status') in {'FAILED', 'ROUTING_FAILED'}:
            stopped = report.get('error_type', 'PAIR_FAILED')
            break
    reviewed = sum(row['report'].get('status') not in {'FAILED', 'ROUTING_FAILED'}
                   for row in results)
    return {'total_pairs': total, 'reviewed_pairs': reviewed, 'results': results,
            'oversized_pairs': oversized, 'unreviewed_pairs': total - reviewed,
            'stop_reason': stopped, 'pair_limit': 12,
            'assurance': 'Pairwise model review, not independent factual verification'}


def _grouped_synthesis(runtime, run_id, manifest, evidence, task_text, guard):
    groups = evidence_groups(evidence)
    if groups is None:
        return {'status': 'SKIPPED_EVIDENCE_WINDOW', 'reason': 'INDIVISIBLE_PASSAGE_TOO_LARGE'}
    results = []
    stopped = None
    for index, group in enumerate(groups):
        guard()
        current = runtime.db.get_run(run_id)
        key = f'workset_synthesis_group_{index}'
        if (key not in current['metadata'] and current['used_llm_calls'] + 2 >
                Budget.from_dict(current['budget']).max_llm_calls):
            stopped = 'CALL_BUDGET'
            break
        report = run_durable_reader(runtime.db, run_id, group,
            DurableResearchModel(runtime.llm_runtime, run_id, stage='workset_synthesis_group'),
            protocol_id=manifest['protocol'], finalize=False, checkpoint_key=key,
            question=('仅对当前原文组作局部综合，比较机制、实验条件、支持与反对证据。'
                      '组外证据未提供，不得声称全库一致或给出全局排名。'
                      '不同划分与实验条件的指标不可直接比较；假设不是实验结果。'
                      '不要声称研究目标完成或实验已执行。研究要求：\n' + task_text))
        guard()
        results.append({'group': index, 'evidence_ids': [row['evidence_id'] for row in group],
                        'report': report})
        if report.get('status') in {'FAILED', 'ROUTING_FAILED'}:
            stopped = report.get('error_type', 'GROUP_FAILED')
            break
    usable = [row for row in results if row['report'].get('answer') and
              row['report'].get('direct_answer_supported') is not False and
              row['report'].get('status') not in {'FAILED', 'ROUTING_FAILED'}]
    reviewed_ids = {eid for row in results if row['report'].get('status') not in
                    {'FAILED', 'ROUTING_FAILED'} for eid in row['evidence_ids']}
    missing = [row['evidence_id'] for row in evidence if row['evidence_id'] not in reviewed_ids]
    limitations = [f'这是局部分组综合，不是全局结论；已取得 {len(usable)}/{len(groups)} 组可支持回答。',
                  '尚未核验不同组之间的冲突，不能据此断言所有文献一致或给出全局排名。',
                  f'尚有 {len(missing)} 段原文未完成分组核验；未执行任何实验。']
    if stopped:
        limitations.append('分组综合停止原因：' + stopped)
    cross = None
    cross_usable = []
    if manifest.get('cross_group_review', False) and not stopped:
        cross = _cross_group_review(runtime, run_id, manifest, groups, task_text, guard)
        cross_usable = [row for row in cross['results'] if row['report'].get('answer') and
                        row['report'].get('direct_answer_supported') is not False and
                        row['report'].get('status') not in {'FAILED', 'ROUTING_FAILED'}]
        limitations[1] = (f"不同组之间的冲突已进行 {cross['reviewed_pairs']}/{cross['total_pairs']} "
                          '对原文的模型审阅，非独立事实验证；不能据此断言全库一致或给出全局排名。')
        if cross['unreviewed_pairs']:
            limitations.append(f"尚有 {cross['unreviewed_pairs']} 对跨组原文未完成对照；"
                               f"停止原因：{cross['stop_reason'] or 'EVIDENCE_WINDOW'}。")
    report = {'status': 'REVIEW_REQUIRED', 'answer_scope': 'PARTIAL',
              'direct_answer_supported': bool(usable or cross_usable),
              'answer': '\n\n'.join(f"### 原文组 {row['group'] + 1}/{len(groups)}（局部综合）\n" +
                                      row['report']['answer'] for row in usable),
              'limitations': limitations + [text for row in results
                  for text in row['report'].get('limitations', [])],
              'unverified_limitations': [text for row in results
                  for text in row['report'].get('unverified_limitations', [])]}
    if cross_usable:
        report['answer'] += '\n\n## 跨组原文对照（局部发现）\n' + '\n\n'.join(
            row['report']['answer'] for row in cross_usable)
    if cross:
        for row in cross['results']:
            for field in ('limitations', 'unverified_limitations'):
                report[field].extend(row['report'].get(field, []))
    return {'status': 'REVIEWED_GROUPS' if usable or cross_usable else 'NO_SUPPORTED_SYNTHESIS',
            'cross_group_review': cross,
            'report': report, 'groups': results, 'group_count': len(groups),
            'unreviewed_evidence_ids': missing, 'cross_group_verified': False,
            'stop_reason': stopped, 'evidence_ids': [row['evidence_id'] for row in evidence]}


def synthesize_worksets(runtime, run_id, manifest, reports, *, guard):
    db = runtime.db
    guard()
    if len(reports) != len(manifest['pages']):
        raise ValueError('Synthesis requires every page report')
    # Full requirements are included or synthesis is skipped, never silently cut.
    # Drop bookkeeping, not requirement text or activity bindings. Repeated
    # global restrictions are included once, and must agree across every page.
    tasks = {'global_conditions': None, 'pages': []}
    for page in manifest['pages']:
        conditions = {key: {'text': row['text'], 'applies_to': row.get('applies_to')}
                      for key, row in page['conditions'].items()}
        global_conditions = {k: v for k, v in conditions.items() if v['applies_to'] is None}
        if tasks['global_conditions'] is None:
            tasks['global_conditions'] = global_conditions
        elif tasks['global_conditions'] != global_conditions:
            raise ValueError('Global requirements differ between pages')
        tasks['pages'].append({'label': page['label'],
            'conditions': {k: v for k, v in conditions.items() if v['applies_to'] is not None},
            'activities': {k: v['text'] for k, v in page['pending_actions'].items()}})
    task_text = json.dumps(tasks, ensure_ascii=False)
    if len(task_text.encode('utf-8')) > 16000:
        return {'status': 'SKIPPED_REQUIREMENTS_WINDOW'}
    identifiers = list(dict.fromkeys(eid for report in reports
        for eid in report['workset_context']['selected_evidence_ids']))
    owned = {row['evidence_id']: row for row in db.list_evidence(run_id)}
    if any(eid not in owned for eid in identifiers):
        raise ValueError('Synthesis evidence is missing or outside the run')
    evidence = [{key: owned[eid].get(key) for key in
                 ('evidence_id', 'content', 'source_uri', 'page')} for eid in identifiers]
    if not evidence:
        return {'status': 'SKIPPED_NO_EVIDENCE'}
    if len(json.dumps(evidence, ensure_ascii=False).encode('utf-8')) > 24000:
        if manifest.get('group_large_evidence', False):
            return _grouped_synthesis(runtime, run_id, manifest, evidence, task_text, guard)
        return {'status': 'SKIPPED_EVIDENCE_WINDOW'}
    current = db.get_run(run_id)
    limits = Budget.from_dict(current['budget'])
    if ('workset_synthesis_checkpoint' not in current['metadata'] and
            current['used_llm_calls'] + 2 > limits.max_llm_calls):
        return {'status': 'SKIPPED_CALL_BUDGET'}
    question = ('综合比较以下研究活动的共同机制、差异、冲突及实验条件。'
                '只使用提供的原文证据；条件不一致的指标不可直接排名。'
                '区分文献事实、推断与待验证假设；不要声称研究目标已完成或实验已执行。'
                '证据覆盖不足时仅回答可支持的部分并明确缺口。\n' + task_text)
    report = run_durable_reader(db, run_id, evidence,
        DurableResearchModel(runtime.llm_runtime, run_id, stage='workset_synthesis'),
        protocol_id=manifest['protocol'], question=question, finalize=False,
        checkpoint_key='workset_synthesis_checkpoint')
    guard()
    return {'status': 'REVIEWED' if report.get('answer') and
            report.get('direct_answer_supported') is not False and
            report.get('status') not in {'FAILED', 'ROUTING_FAILED'} else 'NO_SUPPORTED_SYNTHESIS',
            'evidence_ids': identifiers, 'report': report,
            'assurance': 'Model-reviewed synthesis, not independent factual validation'}
