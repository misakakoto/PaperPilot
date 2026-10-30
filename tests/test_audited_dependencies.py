"""A supported model verdict cannot survive invalidated source premises."""
import json
from copy import deepcopy
from types import SimpleNamespace

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.answer_pipeline import auditor_messages
from paperpilot_agentic.providers import LLMResponse


def test_numeric_guard_rejection_propagates_to_inference_and_preserves_mechanism():
    sources = [{'evidence_id': 'method', 'content': 'Uses a local encoder.'},
               {'evidence_id': 'score', 'content': 'TABLE IV\nPERFORMANCES ON THE BRIGHT TEST SET.\n'
                'The method reaches 96.68% accuracy.'}]
    draft = {'facts': [
        {'id': 'method', 'kind': 'SOURCE_FACT', 'text': 'Uses a local encoder.', 'evidence_ids': ['method']},
        {'id': 'score', 'kind': 'SOURCE_FACT', 'text': '该方法在 xBD 上报告的总体准确率为 96.68%',
         'evidence_ids': ['score']},
        {'id': 'inference', 'kind': 'INFERENCE', 'text': 'This supports the proposed design.',
         'evidence_ids': ['score'], 'depends_on': ['score']}],
        'direct_answer_id': 'method', 'limitations': []}
    audit = {'assessments': [{'fact_id': f['id'], 'verdict': 'SUPPORTED', 'relevance': 'DIRECT',
                             'span_ids': ['s0' if f['id'] == 'method' else 's1']} for f in draft['facts']],
             'coverage': 'COMPLETE', 'missing_aspects': []}

    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            self.calls += 1
            return LLMResponse(json.dumps(draft if self.calls == 1 else audit), finish_reason='stop')

    report = answer_then_audit('Describe the encoder and its measured F1.', sources, Model())
    assert report['audit']['assessments'][1]['verdict'] != 'SUPPORTED'
    assert report['audit']['assessments'][2]['verdict'] == 'UNCERTAIN'
    assert 'local encoder' in report['answer']
    assert 'supports the proposed design' not in report['answer']
    assert report['audit']['coverage'] == 'PARTIAL'


def test_output_boundary_allows_existing_conflict_and_revision_contracts():
    options = SimpleNamespace(audit_cited_only=False, relation_policy=False,
                              question_scope_policy=False, draft_contract='flat')
    context = SimpleNamespace(options=options, comparison_contract=False, report={})
    payload = {'question': 'Q', 'facts': [], 'sources': []}
    before = deepcopy(payload)
    system = auditor_messages(context, payload, [{'previous_fact_id': 'old'}])[0]['content']
    boundary = system.split('AUDIT OUTPUT BOUNDARY:')[1]
    assert 'answer_conflicts' in boundary
    assert 'revision_resolutions' in boundary
    assert payload == before
