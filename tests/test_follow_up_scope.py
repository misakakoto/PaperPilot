import pytest
import json

from paperpilot_agentic.follow_up_reader import prior_source_scope
from paperpilot_agentic.orchestrator import ResearchAgentRuntime


def test_resolution_preserves_new_question_attributes_without_history_evidence():
    from paperpilot_agentic.follow_up_reader import resolve_follow_up
    from paperpilot_agentic.providers import LLMResponse
    class Model:
        def complete(self, messages, **kwargs):
            prompt = messages[0]['content']
            assert 'NOT answerability' in prompt
            assert 'ordered entities in prior_question' in prompt
            payload = json.loads(messages[1]['content'])
            assert 'offset' in payload['current_question']
            assert 'prior_answer' not in payload['history']
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps({
                'question': 'What offsets were tested in Beta Table II?',
                'use_prior_sources': True, 'needs_clarification': False,
                'reason': 'Beta is the second paper in the prior question'}), finish_reason='stop')
    result = resolve_follow_up('What offsets were tested in the latter table?', {
        'prior_question': 'Compare Alpha Table I and Beta Table II.',
        'prior_answer': 'They use different datasets.'}, Model())
    assert result['question'] == 'What offsets were tested in Beta Table II?'
    assert result['history_is_evidence'] is False
    assert result['provider_attempts'] == 1


def test_explicit_multi_paper_scope_survives_single_paper_answer():
    parent = {'run': {'metadata': {'paper_sha256s': ['a', 'b']}},
              'final_answer': {'citation_ids': ['e1']},
              'evidence': [{'evidence_id': 'e1', 'source_uri': 'paper://a/chunk', 'page': 1}]}
    context = ResearchAgentRuntime._follow_up_context(parent)
    assert prior_source_scope(context) == ['a', 'b']
    assert 'content' not in context['cited_sources'][0]


def test_resolved_new_subject_overrides_older_declared_scope():
    context = ResearchAgentRuntime._follow_up_context({'run': {'metadata': {
        'paper_sha256s': ['old'], 'resolved_paper_sha256s': ['new']}}})
    assert prior_source_scope(context) == ['new']


def test_legacy_context_falls_back_to_citation_uris_only():
    assert prior_source_scope({'cited_sources': [
        {'source_uri': 'paper://a/chunk'}, {'source_uri': 'https://outside.example'},
        {'source_uri': 'paper:///invalid'}]}) == ['a']
    with pytest.raises(ValueError):
        prior_source_scope({'prior_paper_sha256s': 'not-a-list'})


def test_explicitly_cleared_scope_does_not_restore_older_filter():
    context = ResearchAgentRuntime._follow_up_context({'run': {'metadata': {
        'paper_sha256s': ['old'], 'resolved_paper_sha256s': []}}})
    assert prior_source_scope(context) == []
