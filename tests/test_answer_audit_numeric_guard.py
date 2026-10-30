"""The deterministic scope guard is a necessary condition, not a truth oracle."""
import json

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


EVIDENCE = [
    {'evidence_id': 'e1', 'content':
     'TABLE IV\nPERFORMANCES ON THE BRIGHT TEST SET.\n'
     'The proposed method improves these results to 96.68% and 68.41%, respectively.'}
]
WRONG = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                    'text': 'FlowMamba 在 xBD 上报告的总体准确率为 96.68%',
                    'evidence_ids': ['e1']}],
         'direct_answer_id': 'f1', 'limitations': []}
RIGHT = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT',
                    'text': 'FlowMamba 在 BRIGHT 上报告的总体准确率为 96.68%',
                    'evidence_ids': ['e1']}],
         'direct_answer_id': 'f1', 'limitations': []}


def model_for(draft):
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'SUPPORTED',
                              'relevance': 'DIRECT', 'span_ids': ['s0'],
                              'quotes': [{'evidence_id': 'e1', 'quote': EVIDENCE[0]['content']}]}],
             'coverage': 'COMPLETE', 'missing_aspects': []}

    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(text=json.dumps(draft if self.calls == 1 else audit),
                               finish_reason='stop')
    return Model()


def test_default_guard_withholds_wrong_dataset_binding():
    report = answer_then_audit('What is the accuracy on xBD?', EVIDENCE, model_for(WRONG))
    assert report['model_audit']['assessments'][0]['verdict'] == 'SUPPORTED'
    assert report['audit']['assessments'][0]['verdict'] == 'UNSUPPORTED'
    assert report['numeric_scope']['changed_fact_ids'] == ['f1']
    assert report['answer'] == ''
    assert report['answer_scope'] == 'NO_SUPPORTED_ANSWER'


def test_correct_dataset_binding_survives_guard():
    report = answer_then_audit('What is the accuracy on BRIGHT?', EVIDENCE, model_for(RIGHT))
    assert report['audit']['assessments'][0]['verdict'] == 'SUPPORTED'
    assert report['numeric_scope']['changed_fact_ids'] == []
    assert '96.68%' in report['answer']


def test_guard_is_ablated_for_unbiased_model_baseline():
    report = answer_then_audit('What is the accuracy on xBD?', EVIDENCE, model_for(WRONG),
                              numeric_scope_mode='off')
    assert report['audit']['assessments'][0]['verdict'] == 'SUPPORTED'
    assert report['numeric_scope']['mode'] == 'off'


def test_unresolved_scope_downgrades_to_uncertain_not_false_rejection():
    evidence = [{'evidence_id': 'e1', 'content': 'The method reaches 96.68%.'}]
    draft = {'facts': [{'id': 'f1', 'text': 'Method on xBD reaches 96.68%', 'evidence_ids': ['e1']}],
             'direct_answer_id': 'f1', 'limitations': []}
    report = answer_then_audit('What is the accuracy?', evidence, model_for(draft))
    assert report['audit']['assessments'][0]['verdict'] == 'UNCERTAIN'
    assert report['numeric_scope']['facts'][0]['checks'][0]['status'] == 'UNRESOLVED'


def test_invalid_scope_mode_fails_before_provider_call():
    model = model_for(WRONG)
    try:
        answer_then_audit('Question', EVIDENCE, model, numeric_scope_mode='unknown')
    except ValueError as error:
        assert 'scope mode' in str(error)
    else:
        raise AssertionError('invalid scope mode should fail')
    assert model.calls == 0
