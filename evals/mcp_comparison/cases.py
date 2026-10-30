"""Frozen questions and AI-reviewed criteria. Criteria never enter model requests."""
from evals.interview.cases import CHANGE, FLOW, acceptance_cases


def cases(root):
    original = [c for c in acceptance_cases(root) if c.get('original_demo')]
    additions = [
        ('backbone_tradeoff', [CHANGE], '仅依据 ChangeMamba 的 Table XIII，比较 VMamba-Small 与 EfficientNet-B5 的 F1、GFLOPs 和参数量。给出 F1 的百分点差，并说明能否称 VMamba 在精度和计算量上同时占优。', [17], ['VMamba F1 82.83, GFLOPs 114.82, Param 49.94M', 'EfficientNet F1 82.26, GFLOPs 76.88, Param 31.63M', '+0.57 pp; not both better']),
        ('backbone_budget', [CHANGE], '若只允许选择 ChangeMamba Table XIII 中 GFLOPs 不超过 80 的骨干，哪一行 F1 最高？写出所有符合预算的行及其 F1 和 GFLOPs，区分该表内结论与其他输入尺寸下的结论。', [17], ['EfficientNet-B5 82.26/76.88', 'Swin-Small 82.22/97.99 is INELIGIBLE', 'EfficientNet only eligible; no extrapolation']),
        ('disaster_missing_class', [FLOW], 'FlowMamba 的 Table VIII 中 Nairobi flood 与 PNG landslide 分别哪种损伤类别没有标注样本？表中的特殊记号是否表示 F1 为零？请给出事件和类别的对应关系。', [11], ['Nairobi Minor Dmg absent', 'PNG Major Dmg absent', 'mark means absent annotated samples, not zero']),
        ('disaster_metric', [FLOW], '依据 FlowMamba Table VIII，PNG landslide 上两种方法的 damage F1 与 mIoU 各是多少，FlowMamba 分别提高多少个百分点？这能否视为 xBD 测试集 overall F1 的提升？', [11], ['ChangeMamba 84.53 F1dmg, 74.03 mIoU', 'FlowMamba 94.35 F1dmg, 89.44 mIoU', '+9.82 pp, +15.41 pp', 'real-world scenario not xBD overall F1']),
        ('alignment_loss', [FLOW], '请依据 FlowMamba 式(4)写出 mask-guided alignment loss，解释 m 的含义和本研究取值、被比较的两种特征，以及 mask 排除了什么。不要把 m 解释为像素数。', [4], ['1/m sum_i=1..m [y_u elementwise (fa[i]-f1[i])]^2', 'm feature stages=4', 'aligned fa versus reference f1', 'unchanged building mask excludes background and changed areas']),
        ('ssm_discretization', [CHANGE], '依据 ChangeMamba 式(2)转录零阶保持离散化的 A-bar 与 B-bar，保留指数、逆矩阵与乘法顺序，并说明 Delta 的作用。不要自行省略 B-bar 的因子。', [4], ['Abar=exp(Delta A)', 'Bbar=(Delta A)^(-1)(exp(Delta A)-I)Delta B', 'Delta timescale parameter']),
        ('deployment_limits', [FLOW], 'FlowMamba 已有光流对齐模块，是否就能保证洪灾厚云、暴雨和树木遮挡场景中的建筑评估可靠？只依据论文 Limitations 解释两种限制，不要把错位鲁棒性扩展成万能保证。', [11], ['cloud/heavy rain reduce image quality and inundated building detection', 'trees occlude buildings and localization', 'alignment not universal guarantee']),
        ('robustness_conditions', [CHANGE], 'ChangeMamba 在 Fig.14 的高斯模糊和高斯噪声测试分别报告了多少 F1 下降、对应什么强度条件？这些数字能否解释成在任意退化或所有数据集上都只下降相同比例？请区分比例与百分点。', [17], ['blur sigma 0 to 2: 8.73% decline', 'noise std20: 8.26% decline', 'LEVIR-CD+ test, ratios not claimed pp; no universal extrapolation']),
    ]
    return original + [dict(id='unseen-'+i, paper_ids=ids, question=q, truth_pages=pages,
                            criteria=truth, original_demo=False)
                       for i, ids, q, pages, truth in additions]
