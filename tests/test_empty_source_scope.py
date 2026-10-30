"""A deliberately emptied source set is not an unrestricted corpus query."""
import json
import pytest

from paperpilot_agentic.models import Budget, RunStatus
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.research_notebook import ResearchNotebook


@pytest.mark.parametrize('engine', ['answer_first_text', 'research_notebook'])
def test_exclude_all_continue_and_explicit_restore(tmp_path, monkeypatch, engine):
    seen = []
    class Resolver:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            # The rewrite omits the source edit. Original user text must win.
            return LLMResponse(text=json.dumps({'question':'Continue analysis.',
                'use_prior_sources':True, 'needs_clarification':False,
                'reason':'Synthetic lossy rewrite'}), finish_reason='stop')
    class Search:
        def resolve_explicit_title_scope(self, queries, **kwargs):
            text = ' '.join(queries)
            return {sha for name, sha in [('Alpha','a'),('Beta','b')] if name in text}
        def search(self, query, *, top_k, paper_sha256s):
            seen.append(paper_sha256s)
            raise RuntimeError('Stop after scope capture, not an answer-quality test')
    def notes(self, question, **kwargs):
        seen.append(kwargs.get('paper_sha256s'))
        raise RuntimeError('Stop after notebook scope capture')
    monkeypatch.setattr(ResearchNotebook, 'retrieve_sources', notes)
    rt = ResearchAgentRuntime(db_path=tmp_path/'scope.db', workspace=tmp_path/'work',
        llm=Resolver(), search_adapter=Search(), text_reader_protocol='scope-test-v1')
    root = rt.db.create_run('Analyze Alpha and Beta', Budget(), metadata={
        'engine':engine, 'conversation_id':'owned', 'resolved_paper_sha256s':['a','b']})
    rt.db.update_run_status(root, RunStatus.COMPLETED)
    previous = root
    for question in ['Exclude Alpha and Beta.', 'Continue analysis.']:
        rid = rt.enqueue_follow_up(parent_run_id=previous, question=question)['run']['run_id']
        result = rt.resume(rid)
        assert seen == []
        assert result['run']['research_outcome'] == 'UNRESOLVED'
        assert result['run']['metadata']['resolved_paper_sha256s'] == []
        assert result['final_answer']['answer'] == ''
        previous = rid
    assert result['run']['metadata']['source_scope_explicitly_empty'] is True
    restored = rt.enqueue_follow_up(parent_run_id=previous, question='Also include Beta.')['run']['run_id']
    result = rt.resume(restored)
    assert seen == [{'b'}]
    assert result['run']['status'] == 'FAILED'  # Deliberate retrieval stop, not success.
    assert result['run']['metadata']['source_scope_explicitly_empty'] is False
