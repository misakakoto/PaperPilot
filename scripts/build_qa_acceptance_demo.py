"""Build a local, immutable replay from actual frozen results; never call an LLM."""
import argparse
import html
import json
from pathlib import Path


def build(campaign, revision, output):
    folder = campaign/'revisions'/f'{revision:04}' if revision else campaign
    report = json.loads((folder/'acceptance-report.json').read_text(encoding='utf-8'))
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    sections = []
    for case, row in zip(protocol['cases'], report['rows'], strict=True):
        path = folder/(case['id']+'-result.json')
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        review_path = folder/'source-reviews'/(case['id']+'.json')
        review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
        answer = (result.get('final_answer') or {}).get('answer') or '没有可发布答案。'
        checks = ''.join('<li><b>'+html.escape(c['dimension'])+'</b> — '+html.escape(c['result'])+
            '：'+html.escape(c.get('reason', ''))+'</li>' for c in review.get('checks', []))
        sections.append('<details><summary>'+html.escape(case['id'])+' · '+html.escape(row['status'])+
            ' · 原文 '+html.escape(row['source_review'])+'</summary><p>'+html.escape(case['question'])+
            '</p><pre>'+html.escape(answer)+'</pre><ul>'+checks+'</ul></details>')
    budget, total = report['budget'], report['aggregate_cost']
    attempts = report['rows']
    recorded = sum(row['status'] != 'NOT_RUN' for row in attempts)
    passes = report['original_demo_passes']
    history = {key: total[key] - budget[key] for key in
               ('actual_requests', 'reported_tokens', 'unsettled_upper_bound')}
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PaperPilot 有界问答验收回放</title>
<style>body{font:16px/1.6 system-ui;background:#f2f5fa;color:#243248;max-width:1100px;margin:40px auto;padding:0 24px}
header,details,.card{background:white;border:1px solid #d6deeb;border-radius:12px;padding:20px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card strong{font-size:26px;display:block}
.warn{background:#fff2d8;padding:14px;border-radius:8px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere}
code{overflow-wrap:anywhere}@media(max-width:700px){.cards{grid-template-columns:1fr}}</style>
<header><small>2026-10-05 · 保存的真实请求回放 · 不发送请求</small>
<h1>PaperPilot：运行完成 ≠ 答案正确</h1><p>证据驱动的论文理解与研发 Agent</p>
<p class="warn">原三题实时验收 PASSES/3。RECORDED 题有执行记录，NOT_RUN_COUNT 题未运行；上下文 CONTEXT_STATUS。表格与公式仍为实验功能。</p></header>
<div class="cards"><div class="card"><strong>PASSES/3</strong>原题满足冻结标准并通过原文复核</div>
<div class="card"><strong>REQUESTS</strong>本轮实际尝试／REQUEST_CEILING</div>
<div class="card"><strong>TOKENS</strong>本轮确认 token／TOKEN_CEILING</div></div>
<h2>验收边界</h2><p>过程完成、模型审核、来源与结构校验、原文复核分开记录。
只展示本冻结版本保留的原答案；不改写回答，不把未审核草案或离线回归计为实时通过。
逐项缺口见下方原文复核记录。</p>
<p>所有预检、视觉、生成、审核和修订共用一个持久化账本；刷新或重复提交复用已有结果。
历史 UNKNOWN 保持原样；本轮 UNKNOWN_COUNT 次，UNKNOWN_TOKENS token 未知占额。
停止原因 STOP_REASON；未知占额不是服务端已结算用量。</p><h2>冻结九题逐项回放</h2>SECTIONS
<h2>成本与边界</h2><p>历史：HISTORY_REQUESTS 次尝试、HISTORY_TOKENS 确认 token、HISTORY_UNKNOWN UNKNOWN 占额。
合计：TOTAL_REQUESTS 次尝试、TOTAL_TOKENS 确认 token、TOTAL_UNKNOWN 未知占额、TOTAL_OCCUPIED 总占额。独立预算分别核算。</p>
<p>真实模型版本指纹：<code>FINGERPRINT</code>。实时成绩绑定此冻结版本。
真实上下文遗漏率和 token 节省均未测量。既有 Recall@8 77.72%→88.74% 是固定子集检索组件结果。</p></html>'''
    values = {'REQUESTS': str(budget['actual_requests']), 'TOKENS': f"{budget['reported_tokens']:,}",
              'PASSES': str(passes), 'RECORDED': str(recorded), 'NOT_RUN_COUNT': str(len(attempts)-recorded),
              'CONTEXT_STATUS': html.escape(report['context']['status']),
              'REQUEST_CEILING': str(budget['max_requests']), 'TOKEN_CEILING': f"{budget['max_tokens']:,}",
              'UNKNOWN_COUNT': str(sum(a['state'] == 'UNKNOWN' for a in budget['attempts'])),
              'UNKNOWN_TOKENS': f"{budget['unsettled_upper_bound']:,}",
              'STOP_REASON': html.escape(report.get('stop_reason') or '未停止'),
              'HISTORY_REQUESTS': str(history['actual_requests']),
              'HISTORY_TOKENS': f"{history['reported_tokens']:,}",
              'HISTORY_UNKNOWN': f"{history['unsettled_upper_bound']:,}",
              'TOTAL_OCCUPIED': f"{total['occupied_tokens']:,}",
              'TOTAL_UNKNOWN': f"{total['unsettled_upper_bound']:,}",
              'TOTAL_REQUESTS': str(total['actual_requests']), 'TOTAL_TOKENS': f"{total['reported_tokens']:,}",
              'FINGERPRINT': html.escape(protocol['identity']['source_sha256']), 'SECTIONS': ''.join(sections)}
    # Substitute longer placeholders first; replacement values are escaped or numeric.
    for key in sorted(values, key=len, reverse=True):
        page = page.replace(key, values[key])
    with output.open('x', encoding='utf-8') as stream:
        stream.write(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--revision', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.campaign_dir, args.revision, args.output)
