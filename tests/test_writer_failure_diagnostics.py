import json

from paperpilot_agentic.answer_audit import answer_then_audit, draft_repair_diagnostics
from paperpilot_agentic.providers import LLMResponse


def test_chinese_limitation_does_not_hide_english_fact_language_failure():
    # The real writer mixed uncited SOURCE_FACT and Chinese LIMITATION entries.
    draft = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'A uses temporal order.', 'evidence_ids': ['e1']},
        {'id': 'f2', 'kind': 'SOURCE_FACT', 'text': 'No table was retrieved.', 'evidence_ids': []},
        {'id': 'f3', 'kind': 'LIMITATION', 'text': '当前证据无法核实表格。', 'evidence_ids': []},
    ], 'direct_answer_id': 'f1', 'limitations': ['当前证据无法核实表格。']}
    diagnostic = draft_repair_diagnostics(
        draft, question='请解释机制和实验。', primary=ValueError('Fact must cite supplied evidence: f2'))
    assert 'Answer language must match the Chinese original question' in diagnostic
    assert 'f3' in diagnostic and 'SOURCE_FACT or INFERENCE' in diagnostic
    assert 'f2' in diagnostic and 'limitations' in diagnostic


def test_repair_receives_explicit_language_and_kind_contract():
    calls = []
    bad = {'facts': [
        {'id': 'f1', 'text': 'Uses temporal order.', 'evidence_ids': ['e1']},
        {'id': 'f2', 'kind': 'LIMITATION', 'text': '当前证据不足。', 'evidence_ids': []},
    ], 'direct_answer_id': 'f1', 'limitations': ['当前证据不足。']}
    good = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': '该机制按时间顺序组织。',
                       'evidence_ids': ['e1']}], 'direct_answer_id': 'f1', 'limitations': []}
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                              'span_ids': ['s0']}], 'coverage': 'COMPLETE'}

    class Model:
        def complete(self, messages, **kwargs):
            calls.append(messages)
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps([bad, good, audit][len(calls)-1]), finish_reason='stop')

    report = answer_then_audit('请解释机制。',
        [{'evidence_id': 'e1', 'content': 'The mechanism uses temporal order.'}], Model())
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['raw_draft'] == bad
    assert report['repaired_draft'] == good
    assert '请用中文' in calls[1][-1]['content']
    assert 'LIMITATION' in calls[1][-1]['content']
    assert report['provider_attempts'] == 3


def test_owned_english_language_edit_reaches_writer_contract():
    calls = []

    class EmptyModel:
        def complete(self, messages, **kwargs):
            calls.append(messages)
            kwargs['before_attempt']()
            payload = ({'facts': [], 'limitations': ['No supplied evidence.']} if len(calls) == 1
                       else {'assessments': [], 'coverage': 'UNANSWERABLE'})
            return LLMResponse(text=json.dumps(payload), finish_reason='stop')

    answer_then_audit('请解释机制。', [{'evidence_id': 'e1', 'content': 'Source.'}], EmptyModel(),
                      request_history=[{'user_request': 'Please answer in English.'}])
    payload = json.loads(calls[0][-1]['content'])
    assert payload['output_contract']['prose_language'] == 'English'
    assert payload['question'] == '请解释机制。'


def test_only_exact_duplicate_uncited_limitation_can_be_removed():
    from paperpilot_agentic.answer_audit import normalize_duplicate_limitations, validate_draft
    limitation = '当前证据无法核实表格。'
    draft = {'facts': [
        {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': '该机制按时间顺序组织。', 'evidence_ids': ['e1']},
        {'id': 'f2', 'kind': 'LIMITATION', 'text': limitation, 'evidence_ids': []},
    ], 'direct_answer_id': 'f1', 'limitations': [limitation]}
    normalized, changes = normalize_duplicate_limitations(draft)
    assert len(normalized['facts']) == 1
    assert changes == ['f2']
    assert len(draft['facts']) == 2
    assert normalized['limitations'] == draft['limitations']
    validate_draft(normalized, [{'evidence_id': 'e1'}], question='解释机制。')
    for override in ({'kind': 'SOURCE_FACT'}, {'evidence_ids': ['e1']},
                     {'text': '论文没有任何实验。'}):
        candidate = {**draft, 'facts': [draft['facts'][0], {**draft['facts'][1], **override}]}
        assert normalize_duplicate_limitations(candidate)[1] == []
    for override in ({'direct_answer_id': 'f2'},
                     {'facts': [{**draft['facts'][0], 'depends_on': ['f2']}, draft['facts'][1]]}):
        assert normalize_duplicate_limitations({**draft, **override})[1] == []


def test_unrequested_ocr_formula_is_not_published_as_mechanism_prose():
    import pytest
    from paperpilot_agentic.answer_audit import validate_draft
    draft = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'evidence_ids': ['e1'],
              'text': '该机制先排列第一时相，再排列第二时相，形成序列 F = [T1(1), T1(HW/(21+j)), T2(1)]。'}],
             'direct_answer_id': 'f1'}
    with pytest.raises(ValueError, match='formula'):
        validate_draft(draft, [{'evidence_id': 'e1'}], question='解释 token 组织机制。')
    assert validate_draft(draft, [{'evidence_id': 'e1'}], question='列出该机制的公式。')
    # An offline audit of a historical draft retains its original text.
    assert validate_draft(draft, [{'evidence_id': 'e1'}]) == draft
