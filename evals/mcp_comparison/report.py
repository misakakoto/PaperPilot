"""Export the unchanged questions/answers and measured costs for source review."""
import argparse
import hashlib
import json
from pathlib import Path

from .cases import cases
from .inspect import inspect

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/mcp-comparison-20261017'


def export(round_folder, destination):
    selection_path=round_folder/'selection.json'
    selected=set(json.loads(selection_path.read_text(encoding='utf-8'))['case_ids']) if selection_path.exists() else None
    lines=['# PaperPilot 本地方法与官方 MCP 真实测试',
        '', '原始回答逐字保留。原文复核者为 AI，不是人工专家。未执行项不算失败，也不算通过。',
        '', '该对照测量“第三方 Markdown 语料接入现有问答流程”的效果；检索切块、页码可用性也会改变。',
        '不能将差异全部归因于解析引擎，更不能归因于 MCP 协议。两臂保留相同的视觉补读和既有复核资产。', '']
    observations={}
    for arm in ['local','mcp']:
        folder=round_folder/arm
        data=inspect(folder)
        observations[arm]=data
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        lines += [f'## {arm}：{protocol["identity"]["model"]}', '',
            f'实际模型请求：{data["actual_attempts"]}；已确认 token：{data["confirmed_tokens"]:,}。',
            f'失败/未决请求及其保守占额：`{json.dumps(data["unresolved"],ensure_ascii=False)}`。',
            '供应商没有回报用量的失败请求不能当成免费；预留不是实测 token。', '']
        by_case={r['case']:r for r in data['rows']}
        for case in cases(ROOT):
            row=by_case.get(case['id'])
            lines += ['### '+case['id'], '', '**原问题：** '+case['question'],'']
            if not row:
                lines += [('状态：NOT_SELECTED（本补测轮未选择）。' if selected is not None and case['id'] not in selected else '状态：NOT_RUN（停止原因见本轮账本）。'),'']
                continue
            result_path=folder/(case['id']+'-result.json')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            answer=(result.get('final_answer') or {}).get('answer') or ''
            lines += [f'执行：{row["run_status"]}；系统审核：{row["reader_status"]}；系统覆盖：{row["coverage"]}。',
                      f'耗时 {row["elapsed_seconds"]} 秒；{row["calls"]} 次；{row["tokens"]:,} token。', '',
                      '**实际最终回答：**', '', answer or '（没有发布最终答案）','']
            review_path=folder/'source-reviews-v2'/(case['id']+'.json')
            if not review_path.exists():
                review_path=folder/'source-reviews'/(case['id']+'.json')
            review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else None
            if review:
                if review['result_sha256']!=hashlib.sha256(result_path.read_bytes()).hexdigest():
                    raise ValueError('Source review no longer matches the immutable result')
                lines += ['**AI 原文复核：** '+review['verdict']+'。'+review['reason'], '']
                lines += ['逐维：`'+json.dumps(review.get('dimensions',{}),ensure_ascii=False)+'`。','']
            else:
                lines += ['**AI 原文复核：** PENDING。', '']
            if case.get('criteria'):
                lines += ['**预先冻结的复核要点（不发送给模型）：**', '']
                lines += ['- '+criterion for criterion in case['criteria']]
                lines.append('')
            report=result['run']['metadata'].get('reader_report') or {}
            if report.get('table_projection_errors'):
                lines += ['**宿主数值契约拦截：**', '', '```json',
                    json.dumps(report['table_projection_errors'],ensure_ascii=False,indent=2),'```','']
            lines += [f'原始结果：[JSON](<{result_path.as_posix()}>)；SHA256 `{row["result_sha256"]}`。','']
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(lines),encoding='utf-8')
    (round_folder/'observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2),encoding='utf-8')
    return destination


def export_direct(folder,destination):
    lines=['# 探索性 MCP 文本直接回答：原问题、原草案与事后复核','',
        '此臂没有在线审核/数值投影/视觉补读，输出不能视为正式Agent已审核答案。',
        '相同检索引擎每篇取前2个完整MCP块；这也改变了检索策略，不是单变量解析器对照。','']
    for case in cases(ROOT):
        path=folder/(case['id']+'-result.json')
        lines += ['## '+case['id'],'','**原问题：** '+case['question'],'']
        if not path.exists():
            lines += ['NOT_RUN',''];continue
        result=json.loads(path.read_text(encoding='utf-8'))
        lines += [f'状态：{result["status"]}；{result["response"]["prompt_tokens"]+result["response"]["completion_tokens"]:,} token；{result["elapsed_seconds"]}秒。',
                  '',result['answer'],'']
        review_path=folder/'source-reviews-v2'/(case['id']+'.json')
        if not review_path.exists():
            review_path=folder/'source-reviews'/(case['id']+'.json')
        review=json.loads(review_path.read_text(encoding='utf-8'))
        if review['result_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Direct review/result mismatch')
        lines += ['**AI 原文复核：** '+review['verdict']+'。'+review['reason'],'',
                  '逐维：`'+json.dumps(review.get('dimensions',{}),ensure_ascii=False)+'`。','']
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(lines),encoding='utf-8')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--round-folder',type=Path,default=BASE/'round2')
    parser.add_argument('--output',type=Path,default=BASE/'问题回答与逐项复核.md')
    opt=parser.parse_args()
    print(export(opt.round_folder.resolve(),opt.output.resolve()))
