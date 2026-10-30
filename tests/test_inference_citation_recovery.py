from copy import deepcopy
import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.providers import LLMResponse


EVIDENCE = [{'evidence_id': 'left', 'content': 'Alpha uses temporal tokens.'},
            {'evidence_id': 'right', 'content': 'Beta estimates spatial displacement.'}]
DRAFT = {'facts': [
    {'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Alpha uses temporal tokens.',
     'evidence_ids': ['left'], 'depends_on': []},
    {'id': 'f2', 'kind': 'SOURCE_FACT', 'text': 'Beta estimates spatial displacement.',
     'evidence_ids': ['right'], 'depends_on': []},
    {'id': 'f3', 'kind': 'INFERENCE', 'text': 'The two designs operate on different representations.',
     'depends_on': ['f1', 'f2']}], 'direct_answer_id': 'f3', 'limitations': []}


class Model:
    def __init__(self, draft, verdict):
        self.draft, self.verdict, self.messages = draft, verdict, []

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        self.messages.append(messages)
        if len(self.messages) == 1:
            value = self.draft
        else:
            value = {'assessments': [
                {'fact_id': 'f1', 'verdict': 'SUPPORTED', 'relevance': 'NECESSARY_CONTEXT', 'span_ids': ['s0']},
                {'fact_id': 'f2', 'verdict': 'SUPPORTED', 'relevance': 'NECESSARY_CONTEXT', 'span_ids': ['s1']},
                {'fact_id': 'f3', 'verdict': self.verdict, 'relevance': 'DIRECT',
                 'span_ids': ['s0', 's1'] if self.verdict == 'SUPPORTED' else []}],
                'coverage': 'COMPLETE' if self.verdict == 'SUPPORTED' else 'PARTIAL',
                'missing_aspects': []}
        return LLMResponse(text=json.dumps(value), finish_reason='stop')


@pytest.mark.parametrize('citations', [None, [], ['f1', 'f2']])
@pytest.mark.parametrize('verdict', ['SUPPORTED', 'UNSUPPORTED'])
def test_explicit_source_dependencies_restore_citations_but_auditor_still_decides(citations, verdict):
    draft = deepcopy(DRAFT)
    if citations is not None:
        draft['facts'][2]['evidence_ids'] = citations
    original = deepcopy(draft)
    model = Model(draft, verdict)
    report = answer_then_audit('Compare the representations.', EVIDENCE, model, allow_repair=False)
    assert len(model.messages) == report['provider_attempts'] == 2
    assert report['direct_answer_supported'] == (verdict == 'SUPPORTED')
    assert report['raw_draft'] == original == draft
    assert 'f3' in report['draft_citation_normalization']
    audit_input = '\n'.join(m['content'] for m in model.messages[1])
    assert 'f3' in audit_input
    if verdict == 'UNSUPPORTED':
        assert DRAFT['facts'][2]['text'] not in report['answer']


@pytest.mark.parametrize('mutation', ['unknown_source', 'mixed_namespace', 'unknown_dep',
                                     'inference_dep', 'duplicate_id', 'uncited_premise'])
def test_ambiguous_or_unowned_references_are_not_recovered(mutation):
    from paperpilot_agentic.answer_contracts import normalize_inference_citations
    draft = deepcopy(DRAFT)
    inference = draft['facts'][2]
    inference['evidence_ids'] = ['f1', 'f2']
    if mutation == 'unknown_source':
        inference['evidence_ids'] = ['invented']
    elif mutation == 'mixed_namespace':
        inference['evidence_ids'] = ['left', 'f2']
    elif mutation == 'unknown_dep':
        inference['depends_on'].append('missing')
    elif mutation == 'inference_dep':
        draft['facts'][0]['kind'] = 'INFERENCE'
    elif mutation == 'duplicate_id':
        draft['facts'][1]['id'] = 'f1'
    else:
        draft['facts'][0]['evidence_ids'] = []
    original = deepcopy(draft)
    normalized, changes = normalize_inference_citations(draft, EVIDENCE)
    assert normalized == draft == original
    assert changes == []


def test_explicit_mechanism_followup_supersedes_expanded_ablation_question():
    model = Model(deepcopy(DRAFT), 'SUPPORTED')
    original = '请只作机制层面的比较，保留结论边界。'
    expanded = '比较两种机制并引用 Table IV 的消融结果。'
    # Use a valid source-cited draft so this tests the question seam, not recovery.
    model.draft['facts'][2]['evidence_ids'] = ['left', 'right']
    history = [{'user_request': '请用中文，并且不要启动训练。', 'role': 'initial_request'}]
    # Chinese source statements are required by this owned request.
    for fact in model.draft['facts']:
        fact['text'] = '机制比较：' + fact['text']
    answer_then_audit(expanded, EVIDENCE, model, original_question=original, request_history=history)
    first = json.loads(model.messages[0][-1]['content'])
    assert first['question'] == original
    assert first['request_history'] == history
