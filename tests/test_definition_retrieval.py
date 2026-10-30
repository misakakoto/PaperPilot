from evals.qasper.agent_case import paper_records
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore


def test_definition_question_does_not_lose_expansion_among_short_mentions(tmp_path):
    definition = ('Zero-shot Evidence Transfer Alignment (ZETA) is the proposed method. '
                  + 'The architecture processes representations of documents and queries. ' * 15)
    mentions = [f'ZETA improves scores for task {i}.' for i in range(8)]
    _, document, chunks = paper_records({'title': 'A study', 'abstract': '',
        'full_text': [{'section_name': 'Introduction', 'paragraphs': [definition, *mentions]}]}, source_id='synthetic')
    store = PaperKnowledgeStore(tmp_path/'index.sqlite3', namespace='definition')
    store.sync_documents([document], chunks)
    hits = PaperPilotKnowledgeSearchAdapter(store).search('What does ZETA abbreviate for',
                                                        top_k=8, paper_sha256s={document.sha256})
    originals = {c.stable_id: c.text for c in store.get_chunks([h['stable_id'] for h in hits])}
    assert definition in [originals[h['stable_id']] for h in hits]
