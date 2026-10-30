"""Reviewed pure inference boundaries survive display without claiming truth."""
import pytest

from paperpilot_agentic.answer_boundaries import safe_limitations, reviewed_epistemic_boundary
from paperpilot_agentic.answer_completion import retained_boundary_indices


@pytest.mark.parametrize('value', [
    '该比较不能证明一种方法在实际任务中的精度必然更优。',
    '该机制层面的差异分析不能替代实验结果，无法断言哪种设计在噪声条件下更具鲁棒性。',
    'This mechanism comparison cannot establish superior accuracy.',
])
def test_reviewed_boundary_is_retained_whole(value):
    visible, withheld = safe_limitations([value], reviewed_values=[value])
    assert visible == [value] and withheld == []
    draft = {'limitations': [value]}
    audit = {'limitation_assessments': [{'limitation_index': 0, 'role': 'SCOPE_BOUNDARY'}]}
    assert retained_boundary_indices(draft, audit) == {0}


@pytest.mark.parametrize('value', [
    '该比较不能证明性能更优，因为模型没有任何对齐模块。',
    '该比较不能证明性能更优，但模型在所有数据上都准确。',
    '该比较表明一种方法的精度更高，但不能证明泛化。',
    '该比较不能证明性能更优，模型从未接受训练。',
    'This comparison cannot establish superiority because the baseline has no alignment.',
])
def test_mixed_positive_or_causal_statements_do_not_gain_a_boundary_waiver(value):
    assert not reviewed_epistemic_boundary(value)


def test_unreviewed_pure_boundary_gets_no_new_rendering_waiver():
    value = '该比较不能证明精度必然更优。'
    visible, withheld = safe_limitations([value])
    assert value not in visible and value in withheld
