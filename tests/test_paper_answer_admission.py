from decimal import Decimal

import pytest

from paperpilot_agentic.paper_numbers import numeric_values
from paperpilot_agentic.paper_answer_admission import admit_answer
from paperpilot_agentic.prototype_paper_evidence import validate_analysis


@pytest.mark.parametrize('text,expected', [
    ('作者报告F1为93.5%。', {'93.5'}), ('推理耗时35ms。', {'35'}),
    ('F1 and ResNet18 and Python3.11', set()),
    ('The result is 93.5.', {'93.5'}), ('1e-3与0.001', {'0.001'}),
    ('误差为−1.2，样本为1,000。', {'-1.2', '1000'}),
])
def test_values_do_not_hide_next_to_chinese_or_units(text, expected):
    assert numeric_values(text) == {Decimal(v) for v in expected}


def source(kind='TEXT', quality=None):
    return {'source_id': 's', 'paper_id': 'paper-version', 'source_kind': kind,
            'source_quality': quality or {}, 'content': 'The author reports F1 of 93.50%.'}


def answer():
    return {'answer': '作者报告F1为93.5%。', 'verdict': 'ANSWERED', 'gaps': [],
            'claims': [{'fact': '作者报告F1为93.5%。', 'source_id': 's', 'quote': 'F1 of 93.50%'}]}


def test_admission_allows_decimal_formatting_but_not_hidden_summary_numbers():
    value = answer()
    assert admit_answer(value, [source()])['admitted']
    value['answer'] = '作者报告F1为99.9%。'
    with pytest.raises(ValueError, match='summary'):
        admit_answer(value, [source()])
    value['claims'][0]['fact'] = '作者报告F1为99.9%。'
    with pytest.raises(ValueError, match='Numeric claim'):
        admit_answer(value, [source()])


def test_failed_grid_cannot_supply_a_numeric_claim_even_when_the_text_contains_it():
    with pytest.raises(ValueError, match='Failed table'):
        admit_answer(answer(), [source('TABLE', {'table_structure': 'INVALID_STRUCTURE'})])
    assert admit_answer(answer(), [source('TABLE', {'table_structure': 'GRID_VALIDATED_NOT_FACT_VERIFIED'})])['admitted']


def test_refusal_can_reference_requested_ci_without_presenting_it_as_a_result():
    value = {'answer': '所选PDF不足以报告95%置信区间。', 'verdict': 'INSUFFICIENT',
             'claims': [], 'gaps': ['没有相应实验依据']}
    assert admit_answer(value, [], '请给出95%置信区间')['admitted']
    with pytest.raises(ValueError, match='summary'):
        admit_answer(value, [], '请给出置信区间')


def test_prototype_numeric_source_check_also_handles_chinese_without_spaces():
    content = {'findings': [{'interpretation': '作者报告F1为99.9%。', 'quote': 'F1:93.5%'}]}
    with pytest.raises(ValueError, match='numeric literal'):
        validate_analysis(content, {'tasks': {}, 'artifacts': {}}, {'depends_on': []})
