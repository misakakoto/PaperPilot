import pytest
from paperpilot_agentic.answer_layout import apply_requested_layout


def presentation():
    return {'answer':'事实一 [e1]\n\n事实二 [e2]', 'answer_units':['事实一 [e1]','事实二 [e2]','当前证据无法确定内存需求。'],
            'limitations':['当前证据无法确定内存需求。'], 'answer_scope':'PARTIAL'}


def test_groups_preserve_every_approved_fact_and_gap_verbatim():
    before = presentation()
    seen = []
    def call(messages, tokens):
        seen.append(messages)
        return {'groups':[['u1','u2'], ['u3']]}
    result = apply_requested_layout('请用两点说明依据与缺口', before, call)
    assert result['answer'] == '1. 事实一 [e1]\n   事实二 [e2]\n\n2. 当前证据无法确定内存需求。'
    assert result['limitations_in_answer']
    assert result['presentation_contract']['status'] == 'SATISFIED'
    assert before == presentation()
    assert len(seen) == 1


@pytest.mark.parametrize('plan', [
    {'groups':[['u1'],['u2']]}, {'groups':[['u1','u2'],['u2']]},
    {'groups':[['u1','u2'],['invented']]}, {'groups':[['u1','u2','u3'],[]]},
    {'groups':[['u1','u2'],['u3']], 'heading':'unverified claim'},
])
def test_invalid_formatter_cannot_delete_or_invent_content(plan):
    result = apply_requested_layout('分成两点', presentation(), lambda *args:plan)
    assert result['answer'] == presentation()['answer']
    assert result['presentation_contract']['status'] == 'UNFULFILLED'
    assert result['limitations'][-1].startswith('本轮未能')


def test_formatter_budget_failure_preserves_audited_answer():
    def call(*args):
        raise RuntimeError('Simulated exhausted budget')
    result = apply_requested_layout('用两点说明', presentation(), call)
    assert result['answer'] == presentation()['answer']
    assert result['presentation_contract']['status'] == 'UNFULFILLED'


@pytest.mark.parametrize('question', ['不要用三点说明', '如果用三点会怎样？', '分析这三个模型'])
def test_no_grouping_without_positive_format_request(question):
    assert apply_requested_layout(question, presentation(), lambda *args:pytest.fail('Unexpected call')) == presentation()


def test_insufficient_units_do_not_invent_content_to_fill_count():
    result = apply_requested_layout('用五点说明', presentation(), lambda *args:pytest.fail('Unexpected call'))
    assert result['presentation_contract']['status'] == 'UNFULFILLED'


def test_exact_user_section_titles_group_only_approved_units():
    before = presentation()
    result = apply_requested_layout('请将方法机制和实验结论分开回答。', before,
        lambda *args: {'groups': [['u1'], ['u2', 'u3']]})
    assert result['answer'].startswith('### 方法机制\n\n事实一 [e1]')
    assert '### 实验结论\n\n事实二 [e2]' in result['answer']
    assert '当前证据无法确定内存需求。' in result['answer']
    assert result['presentation_contract']['section_titles'] == ['方法机制', '实验结论']
    assert before == presentation()


def test_named_section_formatter_cannot_supply_its_own_headings():
    result = apply_requested_layout('将方法机制和实验结论分开回答', presentation(),
        lambda *args: {'groups': [['u1'], ['u2', 'u3']], 'heading': 'New conclusion'})
    assert result['answer'] == presentation()['answer']
    assert result['presentation_contract']['status'] == 'UNFULFILLED'


def test_unrelated_negation_does_not_cancel_explicit_section_request():
    result = apply_requested_layout('请将方法机制和实验结论分开回答；不要混用来源。', presentation(),
        lambda *args: {'groups': [['u1'], ['u2', 'u3']]})
    assert result['presentation_contract']['section_titles'] == ['方法机制', '实验结论']
    assert result['answer'].startswith('### 方法机制')


def test_three_quoted_titles_preserve_order_and_all_content():
    result = apply_requested_layout('明确分成“论文事实”“比较性推断”“结论边界”，说明不能证明什么。',
        presentation(), lambda *args: {'groups': [['u1'], ['u2'], ['u3']]})
    assert result['presentation_contract']['section_titles'] == ['论文事实', '比较性推断', '结论边界']
    for text in presentation()['answer_units']:
        assert text in result['answer']


def test_assignment_plan_preserves_every_unit_and_user_title():
    before = presentation()
    result = apply_requested_layout('将设计依据和实验结果分开回答', before,
        lambda *args: {'assignments': {'u1': 1, 'u2': 2, 'u3': 2}})
    assert result['presentation_contract']['status'] == 'SATISFIED'
    assert result['presentation_contract']['groups'] == [['u1'], ['u2', 'u3']]
    assert '### 设计依据' in result['answer'] and '### 实验结果' in result['answer']
    assert all(text in result['answer'] for text in before['answer_units'])


@pytest.mark.parametrize('assignments', [
    {'u2': 1, 'u3': 2}, {'u1': 1, 'u2': 1, 'u3': 1},
    {'u1': True, 'u2': 2, 'u3': 2}, {'u1': 0, 'u2': 2, 'u3': 2},
    {'u1': 1, 'u2': 2, 'u3': 3}, {'u1': 1, 'u2': 2, 'u3': 2, 'invented': 1},
])
def test_assignment_plan_cannot_omit_units_or_invent_groups(assignments):
    result = apply_requested_layout('分成两点', presentation(), lambda *args: {'assignments': assignments})
    assert result['presentation_contract']['status'] == 'UNFULFILLED'
    assert result['answer'] == presentation()['answer']
