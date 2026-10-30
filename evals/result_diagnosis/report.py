"""Export unchanged answers and separately authored source reviews, never score by status."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def costs(folder):
    path = folder/'receipts/spending.sqlite3'
    if not path.exists():
        return {'requests': 0, 'confirmed_tokens': 0, 'unresolved': []}
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY id')]
    return {'requests': len(rows), 'confirmed_tokens': sum(r['tokens'] or 0 for r in rows),
        'unresolved': [{k: r[k] for k in ('id', 'state', 'reservation', 'phase')}
                       for r in rows if r['state'] not in {'RETURNED', 'NOT_SENT'}]}


def export_qa(folder, review_path, target):
    protocol = json.loads((folder/'protocol.json').read_bytes())
    reviews = json.loads(review_path.read_bytes()) if review_path.exists() else {}
    text = ['# 本地问答：原问题、原答案与独立原文复核', '',
        '运行状态、在线模型审核、Codex AI 原文复核分别记录。没有人工专家复核。',
        '旧轮与本轮分开，未执行项不补造答案；内容通过不自动升级为正式验收。', '',
        '源码身份：`' + protocol['identity']['source_sha256'] + '`', '',
        '实际成本：`' + json.dumps(costs(folder), ensure_ascii=False) + '`', '']
    for case in protocol['cases']:
        path = folder/(case['id']+'-result.json')
        text += ['## ' + case['id'], '', '**原问题：** ' + case['question'], '']
        if not path.exists():
            text += ['NOT_RUN：本轮尚无执行回执。', '']
            continue
        raw = path.read_bytes()
        result = json.loads(raw)
        report = result['run']['metadata'].get('reader_report', {})
        review = reviews.get(case['id'])
        digest = hashlib.sha256(raw).hexdigest()
        if review and review['result_sha256'] != digest:
            raise ValueError('Review does not belong to this unchanged result: ' + case['id'])
        text += [f"执行：{result['run']['status']}；模型审核：{report.get('status')}；"
                 f"覆盖：{report.get('final_request_coverage', {}).get('coverage', report.get('audit', {}).get('coverage'))}", '',
                 '**原样最终答案：**', '', (result.get('final_answer') or {}).get('answer') or '（没有发布最终答案）', '',
                 '**保留缺口：** ' + '；'.join((result.get('final_answer') or {}).get('limitations') or []), '',
                 '**AI 原文复核：** ' + (review['verdict'] + '。' + review['reason'] if review else 'PENDING'), '']
        if review:
            text += ['五维复核：`' + json.dumps(review['dimensions'], ensure_ascii=False) + '`', '']
        text += ['原始结果 SHA：`' + digest + '`', '']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(text), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_qa(args.folder, args.reviews, args.output)
