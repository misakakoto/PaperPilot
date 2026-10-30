import json

import pytest

from evals.qasper.agent_case import paper_records
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.section_router import read_selected_sections, select_sections, structural_overviews
from paperpilot_retrieval import PaperKnowledgeStore


def test_outline_is_namespace_bound_and_read_returns_original_chunks(tmp_path):
    store = PaperKnowledgeStore(tmp_path/'papers.sqlite3', namespace='one')
    _, doc, chunks = paper_records({'title': 'Paper', 'abstract': 'abstract', 'full_text': [
        {'section_name': 'Methods', 'paragraphs': ['Original method paragraph.']}]})
    store.sync_documents([doc], chunks)
    outline = store.paper_outline(doc.sha256)
    assert [r['title'] for r in outline] == ['Abstract', 'Methods']
    assert not PaperKnowledgeStore(store.path, namespace='two').paper_outline(doc.sha256)
    result, omitted = read_selected_sections(store, doc.sha256, outline, ['section_1'])
    assert result[0]['content'].endswith('Original method paragraph.')
    assert not omitted
    assert read_selected_sections(store, doc.sha256, outline, ['section_1'], max_chars=1)[1]
    with pytest.raises(ValueError, match='scope'):
        read_selected_sections(store, 'wrong-sha', outline, ['section_1'])


@pytest.mark.parametrize('ids,expected', [(['section_0'], 'SELECTED'), (['invented'], 'FAILED')])
def test_router_cannot_create_nonexistent_sections(ids, expected):
    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text=json.dumps({'section_ids': ids}), finish_reason='stop')
    report = select_sections('Question?', [{'section_id': 'section_0', 'title': 'Methods', 'chars': 80}], Model())
    assert report['status'] == expected
    assert report['provider_attempts'] == 1


def test_structural_overview_is_hierarchical_budgeted_and_not_leaf_guessing():
    rows = [{'section_id': 'p', 'title': 'Approach', 'chars': 100},
            {'section_id': 'c', 'title': 'Approach ::: Design', 'chars': 80},
            {'section_id': 'u', 'title': 'Approaching unrelated topics', 'chars': 20}]
    assert structural_overviews(rows) == ['p']
    assert structural_overviews(rows, max_chars=99) == []
    assert structural_overviews(rows[1:]) == []


def test_empty_selection_is_not_reported_as_successful_navigation():
    class Model:
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            return LLMResponse(text='{"section_ids":[],"reason":"No factual excerpts"}', finish_reason='stop')
    value = select_sections('Explain grouping', [{'section_id': 'm', 'title': 'Methods', 'chars': 40}], Model())
    assert value['status'] == 'EMPTY_SELECTION'
    assert value['provider_attempts'] == 1


@pytest.mark.parametrize('wrong_source', [False, True])
def test_fallback_ranks_bounded_originals_and_checks_identity(wrong_source):
    from types import SimpleNamespace
    from paperpilot_agentic.section_router import read_outline_fallback
    class Store:
        def get_chunks(self, ids):
            assert ids == ['b', 'm']  # the third chunk is outside the scan cap
            return [SimpleNamespace(paper_sha256='outside' if wrong_source else 'sha',
                stable_id=cid, title='Paper', section='REFERENCES' if cid == 'b' else 'broken heading',
                text='Grouping alternates positions.', page_start=1) for cid in ids]
    outline = [{'section_id': 'm', 'chunk_ids': ['b', 'm', 'omitted']}]
    if wrong_source:
        with pytest.raises(ValueError, match='scope'):
            read_outline_fallback(Store(), 'sha', outline, 'Explain grouping', max_chunks=2)
    else:
        rows, meta = read_outline_fallback(Store(), 'sha', outline, 'Explain grouping', max_chunks=2)
        assert [row['evidence_id'] for row in rows] == ['e_m']
        assert meta['unscanned_chunk_count'] == 1
        assert meta['provider_attempts'] == 0 and meta['semantic_coverage'] == 'NOT_PROVEN'
        assert read_outline_fallback(Store(), 'sha', outline, 'Explain grouping', max_chunks=2,
                                     max_chars=1)[0] == []
