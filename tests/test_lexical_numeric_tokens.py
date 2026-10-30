import sqlite3

import pytest

from evals.qasper.agent_case import paper_records
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_retrieval.store import _fts_query


def seed(path, namespace):
    store = PaperKnowledgeStore(path, namespace=namespace)
    _, document, chunks = paper_records({'title': 'Layers', 'abstract': 'Layer 1 baseline.',
        'full_text': [{'section_name': 'Results', 'paragraphs': ['Layer 2 alternative.']}]})
    store.sync_documents([document], chunks)
    return store


@pytest.mark.parametrize('number', ['0', '1', '2', '5', '9'])
def test_single_numeric_query_is_not_erased(number):
    assert _fts_query(number) == f'"{number}"'
    assert _fts_query('a I') == ''


def test_real_fts_distinguishes_single_digit_queries(tmp_path):
    store = seed(tmp_path/'index.sqlite3', 'one')
    assert [h.text for h in store.search_chunks('1', mode='bm25')] == ['Layer 1 baseline.']
    assert [h.text for h in store.search_chunks('2', mode='bm25')] == ['Layer 2 alternative.']
    assert store.search_chunks('5', mode='bm25') == []


def test_explicit_refresh_is_dry_run_idempotent_and_namespace_local(tmp_path):
    path = tmp_path/'index.sqlite3'
    store = seed(path, 'one')
    seed(path, 'two')
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE chunks SET search_text = replace(replace(search_text, '1', ''), '2', '')")
        conn.execute("INSERT INTO chunk_embeddings SELECT row_id, 'fixture', 1, X'0000803F', 'fixture' FROM chunks")
        originals = conn.execute('SELECT row_id, namespace, stable_id, section, text, text_sha256 FROM chunks').fetchall()
        embeddings = conn.execute('SELECT * FROM chunk_embeddings').fetchall()
    assert store.search_chunks('1', mode='bm25') == []
    assert store.refresh_lexical_index() == {
        'namespace': 'one', 'applied': False, 'changed_rows': {'chunks': 2, 'memories': 0}}
    assert store.search_chunks('1', mode='bm25') == []
    assert store.refresh_lexical_index(apply=True)['changed_rows']['chunks'] == 2
    assert len(store.search_chunks('1', mode='bm25')) == 1
    assert PaperKnowledgeStore(path, namespace='two').search_chunks('1', mode='bm25') == []
    assert store.refresh_lexical_index(apply=True)['changed_rows'] == {'chunks': 0, 'memories': 0}
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT row_id, namespace, stable_id, section, text, text_sha256 FROM chunks').fetchall() == originals
        assert conn.execute('SELECT * FROM chunk_embeddings').fetchall() == embeddings


def test_refresh_never_reactivates_or_reindexes_forgotten_memories(tmp_path):
    path = tmp_path/'index.sqlite3'
    store = seed(path, 'one')
    with sqlite3.connect(path) as conn:
        for mid, status in [('active-memory', 'active'), ('forgotten-memory', 'forgotten')]:
            conn.execute('INSERT INTO memories(namespace,memory_id,kind,content,search_text,normalized_sha256,'
                         'confidence,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)',
                         ('one', mid, 'qa_answer', 'Layer 1 result', '', 'fixture', 'medium', status, 't', 't'))
    assert store.refresh_lexical_index(apply=True)['changed_rows']['memories'] == 1
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT memory_id,search_text,status FROM memories ORDER BY memory_id').fetchall()
    assert rows == [('active-memory', 'layer 1 result', 'active'), ('forgotten-memory', '', 'forgotten')]


def test_refresh_failure_rolls_back_prior_chunk_updates(tmp_path):
    path = tmp_path/'index.sqlite3'
    store = seed(path, 'one')
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE chunks SET search_text = replace(replace(search_text, '1', ''), '2', '')")
        # Force the second row to fail after an earlier row has been updated.
        conn.execute("CREATE TRIGGER reject_second BEFORE UPDATE OF search_text ON chunks "
                     "WHEN new.text = 'Layer 2 alternative.' BEGIN SELECT RAISE(ABORT, 'fixture'); END")
        before = conn.execute('SELECT row_id,search_text FROM chunks').fetchall()
    with pytest.raises(sqlite3.IntegrityError):
        store.refresh_lexical_index(apply=True)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT row_id,search_text FROM chunks').fetchall() == before
    assert store.search_chunks('1', mode='bm25') == []
