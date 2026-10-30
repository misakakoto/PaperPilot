"""Read-only historical lineage and supplemental acceptance projections."""
import hashlib
import json
from pathlib import Path
import sqlite3

from .budget import canonical, save
from .checks import structural_observations

REVIEW_DIMENSIONS = {'factual_correctness', 'completeness', 'citation_ownership',
                     'conclusion_scope', 'presentation'}


def historical_totals(snapshot):
    """Aggregate ancestry for display, without granting any new allowance."""
    total = {key: snapshot[key] for key in
             ('actual_requests', 'reported_tokens', 'unsettled_upper_bound')}
    ancestor = snapshot.get('ancestor_cost')
    if ancestor:
        previous = historical_totals(ancestor)
        for key in total:
            total[key] += previous[key]
    return total


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def historical_snapshot(folder):
    folder = Path(folder).resolve()
    ledger = folder/'receipts'/'spending.sqlite3'
    before = sha(ledger)
    with sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute('SELECT * FROM attempts ORDER BY id')]
        limits = dict(conn.execute('SELECT * FROM campaign').fetchone())
        approvals = [json.loads(row[0]) for row in conn.execute('SELECT authorization FROM continuations')]
    bounds = {}
    for row in rows:
        if row['state'] in {'RESERVED', 'STARTED', 'UNKNOWN', 'FAILED'}:
            reviewed = [entry['unknown_bounds'][str(row['id'])] for entry in approvals
                        if str(row['id']) in entry.get('unknown_bounds', {})]
            bounds[str(row['id'])] = max(reviewed) if reviewed else row['reservation']
    if sha(ledger) != before:
        raise ValueError('Historical ledger changed during inspection')
    revisions = sorted((folder/'revisions').glob('*/protocol.json'))
    protocol_path = revisions[-1] if revisions else folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    return {'folder': str(folder), 'ledger_sha256': before,
        'protocol_path': str(protocol_path), 'protocol_sha256': sha(protocol_path),
        'actual_requests': sum(row['state'] != 'NOT_SENT' for row in rows),
        'reported_tokens': sum(row['tokens'] for row in rows),
        'unsettled_upper_bound': sum(bounds.values()), 'unknown_bounds': bounds,
        'states': [{'id': row['id'], 'state': row['state']} for row in rows],
        'max_requests': limits['requests'], 'max_tokens': limits['tokens'],
        'stopped': (folder/'receipts'/'stopped.json').exists(),
        'ancestor_cost': protocol.get('historical_cost'),
        'protocol': protocol}


def backup_parent(parent, destination):
    """Copy corpus and cache-bearing DB; historical answers stay historical."""
    destination = Path(destination)
    source = Path(parent['protocol_path']).parent
    lineage = []
    for name in ('corpus.sqlite3', 'runtime.sqlite3'):
        original, target = source/name, destination/name
        if target.exists():
            raise ValueError('Cannot replace an existing acceptance database')
        before = sha(original)
        with sqlite3.connect(original.as_uri()+'?mode=ro', uri=True) as old:
            with sqlite3.connect(target) as new:
                old.backup(new)
        if sha(original) != before:
            raise ValueError('Parent database changed while taking a read-only backup')
        lineage.append({'source': str(original), 'source_sha256': before,
                        'target': str(target), 'target_sha256': sha(target)})
    save(destination/'cache-lineage.json', {'backups': lineage,
        'assurance': 'Only source-identical parser caches may be reaudited; old answers are not evidence.'})


def cost_estimate(parent_folder):
    """Retained request shapes, not invented future usage or a tokenizer."""
    parent = historical_snapshot(parent_folder)
    samples = []
    for path in sorted((Path(parent_folder)/'receipts').glob('request-*.json')):
        request = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(request.get('max_tokens'), int) or not request.get('messages'):
            continue
        samples.append({'receipt': path.name, 'phase': request.get('phase'),
            'input_utf8_bytes': len(canonical(request['messages']).encode()),
            'output_ceiling': request['max_tokens'],
            'reservation': len(canonical(request['messages']).encode())+request['max_tokens']+2048})
    return {'historical': {k: v for k, v in parent.items() if k != 'protocol'},
            'retained_request_estimates': samples,
            'assurance': 'Each new complete request still needs admission; no guaranteed nine-case cost.'}


def report(folder, budget):
    folder = Path(folder)
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    snapshot = budget.snapshot()
    stop_path = budget.folder/'stopped.json'
    stop = json.loads(stop_path.read_text(encoding='utf-8')) if stop_path.exists() else {}
    not_run_reason = ('BUDGET_EXHAUSTED' if stop.get('reason') == 'budget_exhausted' else
                      'CAMPAIGN_STOPPED' if snapshot['blocked'] else 'NOT_SCHEDULED')
    rows, results = [], {}
    for case in protocol['cases']:
        path = folder/(case['id']+'-result.json')
        if not path.exists():
            rows.append({'case': case['id'], 'status': 'NOT_RUN', 'reason': not_run_reason,
                         'source_review': 'NOT_RUN', 'live_pass': False})
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        results[case['id']] = result
        observations = structural_observations(case, result, results.get(case.get('parent_scenario_id')))
        review_path = folder/'source-reviews'/(case['id']+'.json')
        review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else None
        valid = bool(review and review.get('case') == case['id'] and
                     review.get('run_id') == result['run']['run_id'] and review.get('result_sha256') == sha(path))
        source_pass = bool(valid and review.get('verdict') == 'PASS' and
                           {check.get('dimension') for check in review.get('checks', [])} == REVIEW_DIMENSIONS and
                           review.get('source_artifacts') and
                           all(check.get('result') == 'PASS' for check in review['checks']))
        outcome_ok = not case.get('original_demo') or result['run'].get('research_outcome') == 'ANSWERED'
        from .semantic import dual_observations
        semantic_path = folder/'source-reviews-semantic-v2'/(case['id']+'.json')
        semantic_review = json.loads(semantic_path.read_text(encoding='utf-8')) if semantic_path.exists() else None
        dual = dual_observations(case, result, result_sha256=sha(path),
            parent=results.get(case.get('parent_scenario_id')), review=semantic_review)
        rows.append({'case': case['id'], 'run_id': result['run']['run_id'],
            'status': result['run']['status'], 'source_review': review.get('verdict') if valid else 'PENDING',
            'research_outcome': result['run'].get('research_outcome'),
            'live_pass': bool(source_pass and outcome_ok and not observations['structural_issues']),
            'elapsed_seconds': result.get('acceptance_elapsed_seconds'),
            'structural': observations, 'result_sha256': sha(path), 'dual_scoring': dual})
    historical = protocol.get('historical_cost')
    if historical:
        current = historical_snapshot(historical['folder'])
        if current['ledger_sha256'] != historical['ledger_sha256']:
            raise ValueError('Historical budget changed; do not silently refresh the baseline')
    costs_by_phase = {}
    for attempt in snapshot['attempts']:
        phase = costs_by_phase.setdefault(attempt['phase'], {'actual_requests': 0, 'reported_tokens': 0})
        phase['actual_requests'] += attempt['state'] != 'NOT_SENT'
        phase['reported_tokens'] += attempt['tokens']
    total = {'actual_requests': snapshot['actual_requests'], 'reported_tokens': snapshot['reported_tokens'],
             'unsettled_upper_bound': snapshot['unsettled_upper_bound']}
    if historical:
        previous = historical_totals(historical)
        for key in total:
            total[key] += previous[key]
    total['occupied_tokens'] = total['reported_tokens']+total['unsettled_upper_bound']
    data = {'identity': protocol['identity'], 'rows': rows, 'budget': snapshot,
        'costs_by_phase': costs_by_phase, 'stop_reason': stop.get('reason'),
        'historical_cost': historical, 'aggregate_cost': total,
        'original_demo_passes': sum(row['live_pass'] for row in rows[:3]),
        'original_demo_total': 3, 'same_frozen_version': True,
        'semantic_v2_original_demo_passes': sum(
            row.get('dual_scoring', {}).get('semantic_v2', {}).get('live_pass', False) for row in rows[:3]),
        'context': json.loads((folder/'context-result.json').read_text(encoding='utf-8'))
                   if (folder/'context-result.json').exists() else {'status': 'NOT_RUN'}}
    (folder/'acceptance-report.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 默认问答有界验收', '', f"原三题：{data['original_demo_passes']}/3；同一冻结版本。",
        '过程完成、模型审核、原文复核分别记录；未运行项不从分母移除。', '',
        '| 用例 | 执行状态 | 原文复核 | 实时通过 |', '| --- | --- | --- | --- |']
    lines += [f"| {r['case']} | {r['status']} | {r['source_review']} | {r['live_pass']} |" for r in rows]
    lines += ['', '本轮与历史成本（UNKNOWN 不是已结算用量）：', '',
        f"- 本轮请求 {snapshot['actual_requests']}/{snapshot['max_requests']}，已确认 {snapshot['reported_tokens']} token，未知占额 {snapshot['unsettled_upper_bound']}。",
        f"- 合计请求 {total['actual_requests']}，已确认 {total['reported_tokens']} token，未知占额 {total['unsettled_upper_bound']}，合计占额 {total['occupied_tokens']}。",
        '- 上下文结果见 JSON；所有编译与对照调用使用同一账本。',
        '- 表格 3/10、公式 2/10 仍为历史固定集实验结果，不作为本轮整体正确率。']
    (folder/'acceptance-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return data
