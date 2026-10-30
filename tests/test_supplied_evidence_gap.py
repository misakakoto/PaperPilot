from paperpilot_agentic.answer_boundaries import safe_limitations
import pytest


def test_supplied_evidence_gap_keeps_question_but_not_unreviewed_explanation():
    original = ('所提供证据无法确定 RadarNet 与 OpticNet 谁的推理更快：'
                'RadarNet 的论文未报告延迟，OpticNet 的论文也未报告实测推理延迟。')
    visible, withheld = safe_limitations([original])
    assert visible == ['当前证据无法确定RadarNet 与 OpticNet 谁的推理更快。']
    assert withheld == [original]


def test_unverified_causal_tail_does_not_pass_with_supplied_prefix():
    original = '所提供的证据无法确定模型是否安全：但它肯定适合自动驾驶。'
    visible, withheld = safe_limitations([original])
    assert visible == ['当前证据无法确定模型是否安全。']
    assert withheld == [original]


@pytest.mark.parametrize('scope', [
    'agreement metrics (e.g., kappa) were measured',
    'the reported metric (i.e., F1) exceeds 0.8',
])
def test_abbreviations_and_decimal_do_not_truncate_uncertainty_target(scope):
    original = f'The supplied passages do not establish whether {scope}. The system is certainly safe.'
    visible, withheld = safe_limitations([original])
    assert visible == [f'Cannot determine from the supplied passages whether {scope}.']
    assert withheld == [original]
