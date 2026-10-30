from evals.qasper.agent_case import paper_records
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore


def test_corpus_adapter_never_indexes_questions_or_gold_annotations():
    paper = {'title': 'A public paper', 'abstract': 'abstract text',
             'full_text': [{'section_name': 'Method', 'paragraphs': ['method text']}],
             'figures_and_tables': [], 'qas': [{'question': 'SECRET QUESTION', 'answers': 'SECRET GOLD'}]}
    public, document, chunks = paper_records(paper)
    assert 'qas' not in public
    assert all('SECRET' not in c.text for c in chunks)
    assert [p.number for p in document.pages] == [1, 2]
    paper['qas'] = []
    assert paper_records(paper)[1].sha256 == document.sha256
    assert document.warnings
    _, identified, _ = paper_records(paper, source_id='1234.56789')
    assert identified.file_name == 'qasper-1234.56789.json'
    assert identified.sha256 == document.sha256


def test_long_paragraphs_preserved_in_chunks():
    text = 'abcde' * 3000
    _, _, chunks = paper_records({'title': 'Paper', 'abstract': '', 'full_text': [
        {'section_name': 'Section', 'paragraphs': [text]}]})
    assert ''.join(c.text for c in chunks) == text


def test_public_captions_indexed_without_loading_images_or_gold():
    paper = {'title': 'Paper', 'abstract': 'Summary',
             'full_text': [{'section_name': 'Results', 'paragraphs': ['Body text']}],
             'figures_and_tables': [{'file': 'unread-private-path.png', 'caption': 'Table 1: model comparison'},
                                    {'caption': ''}],
             'qas': [{'answers': 'HIDDEN GOLD'}]}
    _, document, chunks = paper_records(paper)
    assert document.pages[1].text == 'Body text'
    assert document.pages[2].text == 'Table 1: model comparison'
    assert any(c.section == 'Figure/table caption 1 (text only)' for c in chunks)
    assert all('HIDDEN GOLD' not in c.text and 'unread-private-path' not in c.text for c in chunks)


def test_diversity_backfills_single_paper_without_changing_multi_paper_priority():
    adapter = PaperPilotKnowledgeSearchAdapter(None, max_chunks_per_paper=3)
    items = [{'paper_sha256': 'one', 'content': f'passage {i}', 'score': 1-i*.01}
             for i in range(10)]
    assert len(adapter._diversify(items, top_k=8)) == 8
    other = {'paper_sha256': 'two', 'content': 'different paper', 'score': .01}
    selected = adapter._diversify(items + [other], top_k=8)
    assert len(selected) == 8
    assert selected.index(next(x for x in selected if x['paper_sha256'] == 'two')) < 4


def test_real_store_candidate_cap_and_single_paper_search(tmp_path):
    _, doc, chunks = paper_records({'title': 'Scientific baselines', 'abstract': '', 'full_text': [
        {'section_name': 'Results', 'paragraphs': [f'baseline method experiment {i}' for i in range(12)]}]})
    store = PaperKnowledgeStore(tmp_path/'index.sqlite3')
    store.sync_documents([doc], chunks)
    adapter = PaperPilotKnowledgeSearchAdapter(store)
    hits = adapter.search('baseline method', top_k=8)
    assert len(hits) == 8


def test_mmr_is_invariant_to_positive_relevance_score_scaling():
    adapter = PaperPilotKnowledgeSearchAdapter(None)
    items = [{'stable_id': str(i), 'paper_sha256': 'one', 'content': text, 'score': score}
             for i, (text, score) in enumerate([
                 ('contextual representations method model', .05),
                 ('contextual representations method definition', .045),
                 ('unrelated copyright notice', .001)])]
    selected = adapter._diversify(items, top_k=2)
    scaled = adapter._diversify([{**x, 'score': x['score'] * 100 + 8} for x in items], top_k=2)
    assert [x['stable_id'] for x in selected] == [x['stable_id'] for x in scaled] == ['0', '1']
    legacy = adapter._diversify(items, top_k=2, normalize_relevance=False)
    assert [x['stable_id'] for x in legacy] == ['0', '2']
