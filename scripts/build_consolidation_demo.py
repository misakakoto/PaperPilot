"""Build a private, self-contained replay gallery. Reads local receipts; no provider calls."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT))
from evals.structured_pdf import quality_summary, structured_result_is_bound  # noqa: E402
from paperpilot_agentic.structured_extraction import config_sha, file_sha  # noqa: E402


def build(folder: Path, output: Path, review_path: Path | None = None):
    cases = json.loads((folder.parent / 'cases.json').read_text(encoding='utf-8'))['cases']
    records = {case['id']: json.loads((folder / f'{case["id"]}.json').read_text(encoding='utf-8')) for case in cases}
    review_document = json.loads((review_path or folder / 'reviews.json').read_text(encoding='utf-8'))
    reviews = review_document['reviews']
    review_revision = review_document.get('review_revision', 'v1')
    summary = quality_summary(cases, records, reviews)
    cards = []
    for case in cases:
        image = Path(case['crop'])
        if file_sha(image) != case['source']['crop_sha256']:
            raise ValueError('Frozen source crop changed')
        record, review = records[case['id']], reviews[case['id']]
        artifact = ((record.get('result') or {}).get('metadata') or {}).get('structured_extraction')
        # Never show an unbound external PASS as a trusted original-image review.
        counted = review['verdict'] == 'PASS' and structured_result_is_bound(case, record)
        if review['result_sha256'] != config_sha(record) or review['crop_sha256'] != case['source']['crop_sha256']:
            raise ValueError('Review does not belong to the frozen result/crop')
        image_uri = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()
        details = json.dumps(artifact, ensure_ascii=False, indent=2) if artifact else record.get('error', record['status'])
        title = f"{case['id']} · {case['source']['label']} · p{case['source']['page']}"
        cards.append(f'''<article data-kind="{case['kind']}" data-verdict="{'pass' if counted else 'fail'}">
<h2>{html.escape(title)}</h2><p class="badges"><b>{'原图复核 PASS' if counted else '原图复核 FAIL'}</b>
 · {html.escape(record['status'])} · {'已知失败' if case['known_failure'] else '留出样本'}
 · {record['elapsed_seconds']:.2f} 秒</p><p>{html.escape(review['note'])}</p>
<div class="pair"><div><h3>冻结原图</h3><img src="{image_uri}" alt="{html.escape(title, quote=True)}"></div>
<div><h3>保存的转写与状态</h3><pre>{html.escape(details)}</pre></div></div></article>''')
    groups = summary['groups']
    result = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaperPilot · 离线证据诊断</title><style>
body{{margin:0;background:#f4f6f9;color:#182a40;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1260px;margin:auto;padding:32px}}h1{{font-size:30px}}h2{{font-size:20px}}h3{{font-size:16px}}
header,article{{padding:24px;background:white;border:1px solid #dbe2eb;border-radius:14px;margin-bottom:20px}}
.notice{{border-left:4px solid #b64b23;padding:12px;background:#fff5ee}}.metrics{{display:flex;gap:20px;flex-wrap:wrap}}
.metrics strong{{font-size:26px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}img{{width:100%;height:auto;max-height:480px;object-fit:contain;object-position:top}}
pre{{background:#f4f6fa;font-size:12px;line-height:1.5;padding:16px;max-height:480px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}}
select{{padding:10px;border-radius:7px;border:1px solid #adb9ca;margin:0 12px 20px 0}}article[data-verdict=fail]{{border-top:4px solid #b64b23}}
article[data-verdict=pass]{{border-top:4px solid #218266}}.badges{{color:#526278}}[hidden]{{display:none}}@media(max-width:850px){{.pair{{grid-template-columns:1fr}}}}
</style><main><header><h1>PaperPilot · 证据与解析失败诊断</h1>
<p class="notice">离线回放 · 0 次云端调用 · 新版九题 NOT_RUN/BUDGET_EXHAUSTED · 三道实时演示仍未通过</p>
<div class="metrics"><div><strong>{groups['table']['passed']}/{groups['table']['total']}</strong><br>表格原图完全正确 · 留出 {groups['table']['holdout_passed']}/{groups['table']['holdout_total']}</div>
<div><strong>{groups['equation']['passed']}/{groups['equation']['total']}</strong><br>公式原图完全正确 · 留出 {groups['equation']['holdout_passed']}/{groups['equation']['holdout_total']}</div>
<div><strong>{'升级' if summary['promote_default'] else '不升级'}</strong><br>Docling 默认路径门槛</div></div>
<p>区分解析成功、来源/结构校验和事实复核。Docling 识别模型为第三方；个人工程贡献在来源绑定、缓存重审、失败检测、预算与恢复。</p>
<p>复核版本：{html.escape(review_revision)}。Codex 直接对照冻结原图，无人工签名。当前冻结集未建立完整 Qwen 配对基线。失败和超时保留在分母内。</p></header>
<label>类型 <select id="kind"><option value="all">全部 20 区域</option><option value="table">表格</option><option value="equation">公式</option></select></label>
<label>原图复核 <select id="verdict"><option value="all">全部结果</option><option value="fail">失败</option><option value="pass">通过</option></select></label>
{''.join(cards)}<footer>材料来自不可覆盖的原始回执和冻结裁剪；这是诊断演示，不是一次新 Agent 回答。</footer></main>
<script>function filter(){{document.querySelectorAll('article').forEach(card=>{{card.hidden=(kind.value!=='all'&&card.dataset.kind!==kind.value)||(verdict.value!=='all'&&card.dataset.verdict!==verdict.value)}})}}
const kind=document.getElementById('kind'),verdict=document.getElementById('verdict');kind.addEventListener('change',filter);verdict.addEventListener('change',filter);</script></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/structured-pdf-20261002/run-bb5b358ea62a50df')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/consolidation-demo-20261003.html')
    parser.add_argument('--reviews', type=Path, help='Separate crop/result-bound review revision')
    args = parser.parse_args()
    print(build(args.run, args.output, args.reviews))


if __name__ == '__main__':
    main()
