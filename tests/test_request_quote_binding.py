"""Request display anchors cannot erase independently source-audited facts."""
import json
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


def test_bad_request_quote_after_writer_repair_keeps_audited_fact_without_extra_call():
    calls = []
    fact = {'id': 'f', 'text': '该方法使用线性分类器。', 'evidence_ids': ['e']}

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) <= 2:
                value = {'facts': [fact], 'direct_answer_id': 'f',
                         'limitations': ['Unknown runtime.'] if len(calls) == 1 else []}
            else:
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                    'span_ids': ['s0'], 'relevance': 'DIRECT'}], 'coverage': 'COMPLETE',
                    'requirement_assessments': [{'requirement': '说明分类器', 'fact_ids': ['f'],
                        'gap': '', 'request_quote': '请说明...分类器'}]}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')

    report = answer_then_audit('请说明分类器。',
        [{'evidence_id': 'e', 'content': 'The method uses a linear classifier.'}], Model())
    assert fact['text'] in report.get('answer', '')
    assert report['answer_scope'] == 'PARTIAL'
    assert report['provider_attempts'] == len(calls) == 3
    assert report['audit']['requirement_contract_errors']


def test_repeated_display_contract_repair_error_keeps_independently_audited_fact():
    calls = []
    fact = {'id': 'f', 'text': '该方法使用线性分类器。', 'evidence_ids': ['e']}
    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) == 1:
                value = {'facts': [fact], 'direct_answer_id': 'f',
                         'limitations': ['当前证据无法证明该方法在所有任务上更优。']}
            else:
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                    'span_ids': ['s0'], 'relevance': 'DIRECT'}], 'coverage': 'PARTIAL',
                    'requirement_assessments': [{'requirement':'说明分类器', 'fact_ids':['f'],
                        'gap':'', 'request_quote':'说明分类器。'}],
                    'limitation_assessments': [{'limitation_index':0, 'role':'REQUEST_GAP',
                        'requirement':'说明分类器', 'reason':'This already satisfied request has a gap.'}]}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')
    report = answer_then_audit('说明分类器。',
        [{'evidence_id':'e','content':'The method uses a linear classifier.'}], Model())
    assert fact['text'] in report.get('answer', '')
    assert report['answer_scope'] == 'PARTIAL'
    assert len(calls) == report['provider_attempts'] == 3
    assert report['audit']['limitation_contract_errors']
    assert '当前证据无法证明该方法在所有任务上更优。' not in report['answer']


@pytest.mark.parametrize('indices', [[1], [0, 0], ['0']])
def test_repaired_unowned_limitation_mapping_keeps_audited_fact_as_partial(indices):
    """A format repair must not erase facts by inventing a caveat index."""
    calls = []
    fact = {'id': 'f', 'text': '该方法使用线性分类器。', 'evidence_ids': ['e']}

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) == 1:
                value = {'facts': [fact], 'direct_answer_id': 'f',
                         'limitations': ['当前证据不能保证跨数据集性能。']}
            else:
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                    'span_ids': ['s0'], 'relevance': 'DIRECT'}], 'coverage': 'COMPLETE',
                    'requirement_assessments': [{'requirement': '说明分类器', 'fact_ids': ['f'],
                        'gap': '', 'request_quote': '说明分类器。', 'limitation_indices': indices}],
                    'limitation_assessments': [{'limitation_index': 0, 'role': 'REQUEST_GAP',
                        'requirement': '说明分类器', 'reason': 'Wrong gap attribution.'}]}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')

    report = answer_then_audit('说明分类器。',
        [{'evidence_id': 'e', 'content': 'The method uses a linear classifier.'}], Model())
    assert fact['text'] in report.get('answer', '')
    assert report['answer_scope'] == 'PARTIAL'
    assert report['provider_attempts'] == len(calls) == 3
    assert report['audit']['requirement_contract_errors']
    assert report['audit']['limitation_contract_errors']
    assert report['audit']['requirement_assessments'] == []
    assert '当前证据不能保证跨数据集性能。' not in report['answer']


def test_native_request_selection_projects_exact_host_quote_through_default_pipeline():
    calls = []
    question = '说明分类器，并说明不能证明什么。'

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) == 1:
                value = {'facts': [{'id': 'f', 'text': 'Uses a linear classifier.',
                                   'evidence_ids': ['e']}], 'direct_answer_id': 'f', 'limitations': []}
                value['facts'][0]['text'] = '该方法使用线性分类器。'
            else:
                payload = json.loads(messages[-1]['content'])
                anchor = next(u for u in payload['request_quote_units'] if u['text'].startswith('说明分类器'))
                assert anchor['text'] in question
                value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                    'relevance': 'DIRECT', 'span_ids': ['s0']}], 'coverage': 'PARTIAL',
                    'requirement_assessments': [{'requirement': '说明分类器', 'fact_ids': ['f'],
                        'gap': '', 'request_ref': anchor['request_ref']}],
                    'missing_aspects': ['结论边界未完成。']}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')

    report = answer_then_audit(question,
        [{'evidence_id': 'e', 'content': 'The method uses a linear classifier.'}], Model())
    assert '该方法使用线性分类器。' in report['answer']
    assert report['audit']['requirement_assessments'][0]['request_quote'] == '说明分类器，'
    assert 'request_quote' not in report['raw_audit']['requirement_assessments'][0]
    assert report['provider_attempts'] == 2
    assert report['audit']['coverage'] == 'PARTIAL'  # A valid display ref does not supply missing work.


@pytest.mark.parametrize('defect', ['unknown_ref', 'foreign_quote', 'conflicting_quote'])
def test_reference_resolution_cannot_invent_or_silently_rewrite_user_requirements(defect):
    from paperpilot_agentic.answer_gap_display import request_quote_units, resolve_request_refs
    question = '说明分类器。'
    ref = request_quote_units(question)[0]['request_ref']
    row = {'requirement': '说明分类器', 'fact_ids': ['f'], 'gap': '', 'request_ref': ref}
    if defect == 'unknown_ref':
        row['request_ref'] = 'request_9_0'
    else:
        row['request_quote'] = '启动训练。' if defect == 'foreign_quote' else '说明...分类器'
    audit = {'requirement_assessments': [row]}
    original = deepcopy(audit)
    with pytest.raises(ValueError, match='original user request'):
        resolve_request_refs(audit, question)
    assert audit == original


def test_long_and_historical_requests_keep_exact_owned_bounded_anchors():
    from paperpilot_agentic.answer_gap_display import request_quote_units
    question = '机制' * 180 + '，再比较结果。'
    history = [{'user_request': '旧要求已撤销；保留论文范围。'}]
    units = request_quote_units(question, history)
    assert all(0 < len(u['text']) <= 256 and
               any(u['text'] in text for text in (question, history[0]['user_request'])) for u in units)
    assert len({u['request_ref'] for u in units}) == len(units)
    assert question == '机制' * 180 + '，再比较结果。'  # Cataloguing never changes the raw request.


def test_truncated_audit_after_writer_repair_never_publishes_a_draft():
    calls = []

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) <= 2:
                value = {'facts': [{'id': 'f', 'text': '该方法使用线性分类器。',
                                   'evidence_ids': ['e']}], 'direct_answer_id': 'f',
                         'limitations': ['English caveat.'] if len(calls) == 1 else []}
                return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')
            return LLMResponse(text='{"assessments":', finish_reason='length')

    report = answer_then_audit('说明分类器。',
        [{'evidence_id': 'e', 'content': 'The method uses a linear classifier.'}], Model())
    assert report['status'] == 'FAILED' and not report.get('answer')
    assert len(calls) == 3


def test_display_fallback_does_not_approve_an_unknown_scientific_source_span():
    calls = []

    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            calls.append(messages)
            if len(calls) <= 2:
                value = {'facts': [
                    {'id': 'good', 'text': '该方法使用线性分类器。', 'evidence_ids': ['e']},
                    {'id': 'bad', 'text': '该方法使用随机森林。', 'evidence_ids': ['e']}],
                    'direct_answer_id': 'good',
                    'limitations': ['English caveat.'] if len(calls) == 1 else []}
            else:
                value = {'assessments': [
                    {'fact_id': 'good', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT', 'span_ids': ['s0']},
                    {'fact_id': 'bad', 'verdict': 'SUPPORTED', 'relevance': 'DIRECT', 'span_ids': ['fake_span']}],
                    'coverage': 'COMPLETE', 'requirement_assessments': [
                        {'requirement': '说明分类器', 'fact_ids': ['good', 'bad'], 'gap': '',
                         'request_quote': '说明...分类器'}]}
            return LLMResponse(text=json.dumps(value, ensure_ascii=False), finish_reason='stop')

    report = answer_then_audit('说明分类器。',
        [{'evidence_id': 'e', 'content': 'The method uses a linear classifier.'}], Model())
    assert '该方法使用线性分类器。' in report['answer']
    assert '随机森林' not in report['answer']
    assert report['audit']['assessments'][1]['verdict'] == 'UNCERTAIN'
    assert report['answer_scope'] == 'PARTIAL' and report['provider_attempts'] == 3
