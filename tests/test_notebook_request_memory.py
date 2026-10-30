"""The real notebook entry shares bounded request memory, not its paper facts."""
import pytest

from paperpilot_agentic.models import RunStatus
from paperpilot_agentic.research_notebook import ResearchNotebook, resume_notebook_query
from paperpilot_agentic.reader_request_memory import prepare_reader_request_memory
from paperpilot_agentic.durable_reader import ReaderAlreadyStarted
from test_reader_pipeline import factory
from test_reader_request_memory import MemoryProvider
from test_research_notebook import Extractor


def setup(tmp_path, provider, large=True):
    rt = factory(tmp_path, provider)
    question = 'What fact? Answer briefly. ' + ('Background note. ' * 800 if large else '')
    root = rt.enqueue(question=question, metadata={'engine': 'research_notebook',
        'resolved_paper_sha256s': ['sha']})['run']['run_id']
    book = ResearchNotebook(rt.db, root)
    book.add_source('paper://sha/chunk', 'A source fact.')
    book.process_next(Extractor())
    rt.db.update_run_status(root, RunStatus.COMPLETED)
    child = rt.enqueue_follow_up(parent_run_id=root, question='Continue.')['run']['run_id']
    return rt, root, child


def test_notebook_default_entry_compacts_then_answers_from_original_sources(tmp_path):
    provider = MemoryProvider()
    rt, root, child = setup(tmp_path, provider)
    result = rt.resume(child)
    assert result['run']['status'] == 'COMPLETED', result['run']
    assert result['run']['metadata']['request_memory_status'] == 'READY'
    assert result['final_answer']['answer'].startswith('A source fact.')
    assert provider.calls == 5
    assert provider.answer_inputs[0]['request_history'][0]['role'] == 'derived_request_state'
    assert all(e['content'] == 'A source fact.' for e in result['evidence'])
    assert rt.resume(child)['final_answer'] == result['final_answer']
    assert provider.calls == 5
    next_id = rt.enqueue_follow_up(parent_run_id=child, question='Continue.')['run']['run_id']
    next_result = rt.resume(next_id)
    assert next_result['run']['metadata']['request_state_root'] == root
    assert len(provider.memory_inputs[-1]['sources']) == 1


def test_notebook_short_history_needs_no_extra_model_calls(tmp_path):
    provider = MemoryProvider()
    rt, _, child = setup(tmp_path, provider, large=False)
    assert rt.resume(child)['run']['status'] == 'COMPLETED'
    assert provider.calls == 3 and not provider.memory_inputs


def test_notebook_rejected_memory_cannot_retrieve_or_answer(tmp_path, monkeypatch):
    provider = MemoryProvider(reject=True)
    rt, _, child = setup(tmp_path, provider)
    monkeypatch.setattr(ResearchNotebook, 'retrieve_sources', lambda *a, **kw: pytest.fail('retrieved'))
    result = rt.resume(child)
    assert result['run']['metadata']['clarification_required'] is True
    assert result['final_answer']['answer'] == ''
    assert provider.calls == 2 and not provider.answer_inputs


def test_finished_memory_stage_reused_when_resolution_is_interrupted(tmp_path, monkeypatch):
    provider = MemoryProvider()
    rt, root, child = setup(tmp_path, provider)
    import paperpilot_agentic.follow_up_reader as module
    original = module.resolve_follow_up
    def crash(*args, **kwargs):
        raise OSError('lost resolution')
    monkeypatch.setattr(module, 'resolve_follow_up', crash)
    with pytest.raises(OSError, match='lost resolution'):
        resume_notebook_query(rt, child)
    assert provider.calls == 2
    monkeypatch.setattr(module, 'resolve_follow_up', original)
    from paperpilot_agentic.research_notebook import NotebookBusy
    with pytest.raises(NotebookBusy):
        resume_notebook_query(rt, child)
    assert provider.calls == 2
    cached = prepare_reader_request_memory(rt, rt.db.get_run(child), rt.db.get_run(root))
    assert cached['status'] == 'READY' and provider.calls == 2


def test_interrupted_memory_stage_not_automatically_retried(tmp_path, monkeypatch):
    provider = MemoryProvider()
    rt, root, child = setup(tmp_path, provider)
    import paperpilot_agentic.reader_request_memory as module
    def crash(*args, **kwargs):
        raise OSError('lost compile')
    monkeypatch.setattr(module, '_prepare_reader_request_memory', crash)
    with pytest.raises(OSError):
        prepare_reader_request_memory(rt, rt.db.get_run(child), rt.db.get_run(root))
    with pytest.raises(ReaderAlreadyStarted):
        prepare_reader_request_memory(rt, rt.db.get_run(child), rt.db.get_run(root))
    assert provider.calls == 0
