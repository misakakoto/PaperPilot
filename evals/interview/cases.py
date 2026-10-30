"""Frozen development acceptance questions; never pass the checks to the model."""
import json
from pathlib import Path

CHANGE = 'bc6159dbfcd9fba2abfceccfbcb4e13cd816abbb07d490743c6958f4bbeaf1b9'
FLOW = '7ff8e41d335452c27e8debfbef517061729fcd1a09595aa7c7c4e58606dcbf20'


def acceptance_cases(root):
    original = json.loads((Path(root) / 'demo/interview_scenarios.json').read_text(encoding='utf-8'))
    cases = [{**case, 'original_demo': True,
              'paper_ids': case['answer_checks']['allowed_citation_source_sha256s']}
             for case in original['scenarios']]
    cases.extend([
        {'id': 'new-mechanism', 'paper_ids': [CHANGE], 'dimension': 'mechanism',
         'question': '仅依据 ChangeMamba 原文，Parallel 机制在哪里连接双时相特征，它与 Sequential 的组织方式有什么具体不同？将原文事实和你的解释分开。'},
        {'id': 'new-numeric-owner', 'paper_ids': [FLOW], 'dimension': 'numeric_ownership',
         'question': '仅依据 FlowMamba 的 Table VII，在不使用 CDA 的条件下加入 MFA，overall F1 从多少变到多少？写清两行各自的 MFA/CDA 条件及变化的百分点，所有数值绑定原表引用。'},
        {'id': 'new-comparability', 'paper_ids': [CHANGE, FLOW], 'dimension': 'comparability',
         'question': 'ChangeMamba 的 Table XII 与 FlowMamba 的 Table VII 能直接比较 F1 高低并判定哪个方法更好吗？请核对任务、数据集和指标含义，明确事实、推断和尚未确认的实验条件。'},
        {'id': 'new-source-exclusion', 'paper_ids': [CHANGE], 'dimension': 'scope',
         'parent_scenario_id': 'multi_change_flow_comparison',
         'question': '从现在起排除 FlowMamba，只依据 ChangeMamba 说明 Cross 机制；不要沿用上一问中 FlowMamba 的事实或引用。'},
        {'id': 'new-missing-evidence', 'paper_ids': [CHANGE], 'dimension': 'missing_evidence',
         'question': '仅依据 ChangeMamba 原文，完整组合在 Table XII 实验的单卡峰值显存精确是多少 GB？如果没有这个记录，明确回答证据不足，不要拿参数量或推理速度换算。'},
        {'id': 'new-history-retraction', 'paper_ids': [CHANGE], 'dimension': 'history',
         'parent_scenario_id': 'new-source-exclusion',
         'question': '延续上一问的论文范围。撤销只讨论 Cross 的要求，现在比较 Table XII 的基线与完整组合，给出 F1 和差值，并保留数值引用。'},
    ])
    return cases
