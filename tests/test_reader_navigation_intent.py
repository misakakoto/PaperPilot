"""Navigation must receive the same owned request context as answer generation."""
import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.models import RunStatus
from paperpilot_agentic.providers import LLMResponse
from test_reader_pipeline import factory


class NavigationProvider:
    def __init__(self):
        self.inputs = {}

    def complete(self, messages, *, before_attempt, **kwargs):
        before_attempt()
        payload = json.loads(messages[-1]['content'])
        if 'current_question' in payload:
            value = {'question': 'Explain Alpha.', 'use_prior_sources': True,
                     'needs_clarification': False, 'reason': 'Same paper'}
        elif 'outline' in payload:
            self.inputs['section'] = payload
            value = {'section_ids': ['method'], 'reason': 'Inspect method'}
        elif 'unread_candidates' in payload:
            self.inputs['recovery'] = payload
            value = {'action': 'STOP', 'reason': 'Synthetic navigation capture'}
        elif 'evidence' in payload:
            self.inputs['writer'] = payload
            value = {'facts': [{'id': 'f', 'text': 'Alpha uses shared weights.',
                               'evidence_ids': [payload['evidence'][0]['evidence_id']]}],
                     'direct_answer_id': 'f', 'limitations': ['The requested scores remain uncertain.']}
        else:
            value = {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                'relevance': 'NECESSARY_CONTEXT', 'span_ids': [payload['facts'][0]['allowed_span_ids'][0]]}],
                'coverage': 'PARTIAL', 'missing_aspects': ['The requested scores remain uncertain.']}
        return LLMResponse(text=json.dumps(value), finish_reason='stop', prompt_tokens=10, completion_tokens=10)


class NavigationSearch:
    def __init__(self):
        self.store = self

    def paper_outline(self, sha):
        return [{'section_id': 'method', 'title': 'Method', 'chars': 40, 'chunk_ids': ['method']}]

    def get_chunks(self, ids):
        return [SimpleNamespace(paper_sha256='sha', stable_id='method', title='Alpha',
                                section='Method', text='Alpha uses shared weights.', page_start=1)]

    def search(self, query, **kwargs):
        return [{'stable_id': 'method', 'paper_sha256': 'sha', 'source_uri': 'paper://sha/method',
                 'content': 'Alpha\nMethod\nAlpha uses shared weights.', 'page': 1}]


@pytest.mark.parametrize('stage', ['section', 'recovery'])
def test_follow_up_navigation_preserves_original_correction_and_owned_history(tmp_path, stage):
    provider = NavigationProvider()
    runtime = factory(tmp_path, provider)
    runtime.search_adapter = NavigationSearch()
    initial = 'Compare Alpha accuracy and speed on the test split.'
    latest = 'Use the validation split instead; include baseline and method F1 and their difference.'
    root = runtime.enqueue(question=initial, metadata={
        'engine': 'answer_first_text', 'paper_sha256s': ['sha']})['run']['run_id']
    runtime.db.update_run_status(root, RunStatus.COMPLETED)
    rid = runtime.enqueue_follow_up(parent_run_id=root, question=latest)['run']['run_id']
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run']['metadata']
    # Writer already sees the request; navigation must not decide from only a
    # lossy retrieval rewrite and a model-generated description of the gaps.
    assert provider.inputs['writer']['original_question'] == latest
    payload = provider.inputs[stage]
    assert payload.get('original_question') == latest
    assert payload.get('request_history') == provider.inputs['writer']['request_history']
    assert initial in str(payload['request_history'])
    assert payload['question'] == 'Explain Alpha.'
