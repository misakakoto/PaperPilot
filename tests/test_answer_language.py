import pytest
from paperpilot_agentic.answer_audit import requires_chinese_answer, validate_draft, draft_repair_diagnostics


@pytest.mark.parametrize('question,history,expected', [
    ('解释这个方法。Please answer in English.', [], False),
    ('解释这个方法。请用英文回答。', [], False),
    ('Continue.', [{'user_request': '请用中文回答。'}], True),
    ('继续。', [{'user_request': '请用英文回答。'}], False),
    ('这次请用中文回答。', [{'user_request': '请用英文回答。'}], True),
    ('不要用英文回答。解释方法。', [], True),
    ('如果用英文回答会怎样？解释方法。', [], True),
    ('“请用英文回答”是什么意思？', [], True),
    ('英文回答的优点是什么？', [], True),
    ('解释短语 answer in English 的含义。', [], True),
    ('继续。', [{'user_request': '请用英文回答。'}, {'user_request': '请用中文回答。'}], True),
])
def test_language_edits(question, history, expected):
    assert requires_chinese_answer(question, history) is expected


def test_explicit_english_does_not_trigger_false_repair():
    evidence = [{'evidence_id': 'e', 'content': 'A method.'}]
    draft = {'facts': [{'id':'f','text':'A method.','evidence_ids':['e']}], 'direct_answer_id':'f'}
    question = '请解释方法。Please answer in English.'
    assert validate_draft(draft, evidence, question=question) == draft
    assert 'Chinese' not in draft_repair_diagnostics(draft, question=question, primary=ValueError('other error'))
    with pytest.raises(ValueError, match='language'):
        validate_draft(draft, evidence, question='请用中文回答。')


def test_english_override_keeps_citation_validation():
    with pytest.raises(ValueError, match='cite supplied evidence'):
        validate_draft({'facts':[{'id':'f','text':'A method.','evidence_ids':['invented']}],
                        'direct_answer_id':'f'}, [], question='请用英文回答。')


@pytest.mark.parametrize('question,history', [
    ('请用英文回答。这个方法使用什么分类器？', []),
    ('继续解释分类器。', [{'user_request':'后续请用英文回答。'}]),
])
def test_answer_and_audit_accept_owned_english_requirement_without_repair(question, history):
    from test_answer_audit import Model, DRAFT, EVIDENCE, audit
    from paperpilot_agentic.answer_audit import answer_then_audit
    result = answer_then_audit('What classifier?', EVIDENCE, Model([DRAFT, audit()]),
                              original_question=question, request_history=history)
    assert result['status'] == 'REVIEW_REQUIRED'
    assert result['provider_attempts'] == 2
    assert 'draft_repair' not in result
    assert 'Uses a linear classifier.' in result['answer']
