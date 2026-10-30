"""Export unchanged oracle-reader questions, answers and references for local review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.qasper.score_oracle import load_evaluator, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    dataset = json.loads(args.dataset.read_text(encoding='utf-8-sig'))
    evaluator = load_evaluator(args.evaluator)
    result = score(report, dataset, evaluator)
    gold = evaluator.get_answers_and_evidence(dataset, False)
    lines = ['# QASPER dev 阅读测试：原问题与未经修改的回答', '',
             '本文件为真实调用结果的直接导出，不是助手代写的答案。参考答案只在调用后评分/展示；模型输入使用标注证据。', '',
             f"- 模型：{report.get('model')}",
             f"- 请求尝试：{report['provider_attempts']}",
             f"- Answer F1：{result['answer_f1']:.2%}；EM：{result['exact_match']:.2%}",
             '- 仅为小样本 oracle Reader 诊断，不是完整 Agent 或公开榜单分数。', '']
    for i, (case, row) in enumerate(zip(report['cases'], result['rows']), 1):
        lines += [f'## {i}. {case["question"]}', '',
                  f"论文：{case['paper_id']}；问题ID：{case['question_id']}", '',
                  '**模型原回答**', '', str(case.get('prediction') or '(未返回)'), '',
                  '**数据集参考答案（可能存在标注分歧）**', '']
        lines += [f"- {ref['answer']}" for ref in gold[case['question_id']]]
        lines += ['', f"Answer F1：{row['answer_f1']:.2%}；EM：{row['exact_match']:.0%}", '',
                  '<details><summary>实际发送的标注证据</summary>', '']
        lines += [f'{j}. {evidence}' for j, evidence in enumerate(case['evidence'], 1)]
        lines += ['', '</details>', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
