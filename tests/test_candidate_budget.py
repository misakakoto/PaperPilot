"""Candidate budget is not the answer-context budget; exercise the real adapter."""
from evals.qasper.agent_case import paper_records
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import PaperKnowledgeStore
import pytest


class CaptureRanker:
    def __init__(self):
        self.passages = []
    def score(self, query, passages):
        self.passages = list(passages)
        return list(range(len(passages), 0, -1))


def test_scoped_reranker_can_inspect_more_than_twenty_candidates(tmp_path):
    paper = dict(title='Synthetic corpus', abstract='abstract', figures_and_tables=[],
        full_text=[dict(section_name='Experiments', paragraphs=[
            f'experiment evidence description number {n}' for n in range(40)])])
    _, doc, chunks = paper_records(paper, source_id='budget-unit')
    store = PaperKnowledgeStore(tmp_path/'index.sqlite3')
    store.sync_documents([doc], chunks)
    ranker = CaptureRanker()
    adapter = PaperPilotKnowledgeSearchAdapter(store, reranker=ranker, rerank_stage='final', scoped_candidate_k=64)
    hits = adapter.search('experiment evidence', top_k=8, paper_sha256s={doc.sha256})
    assert len(hits) == 8
    assert len(ranker.passages) > 20, 'Evidence beyond rank 20 cannot reach the reranker'
    assert len(ranker.passages) <= 64
    assert adapter.search('experiment evidence', top_k=8, paper_sha256s=set()) == []


@pytest.mark.parametrize('size', [0, 7, 101])
def test_invalid_candidate_budget_is_rejected(size):
    with pytest.raises(ValueError, match='scoped_candidate_k'):
        PaperPilotKnowledgeSearchAdapter(None, scoped_candidate_k=size)


def test_expansion_does_not_change_unscoped_retrieval_contract(tmp_path):
    paper = dict(title='Synthetic corpus', abstract='experiment evidence', figures_and_tables=[],
        full_text=[dict(section_name='Experiments', paragraphs=[
            f'experiment evidence description number {n}' for n in range(40)])])
    _, doc, chunks = paper_records(paper, source_id='unscoped')
    store = PaperKnowledgeStore(tmp_path/'index.sqlite3')
    store.sync_documents([doc], chunks)
    old = PaperPilotKnowledgeSearchAdapter(store, reranker=CaptureRanker(), rerank_stage='final')
    ranker = CaptureRanker()
    new = PaperPilotKnowledgeSearchAdapter(store, reranker=ranker, rerank_stage='final', scoped_candidate_k=64)
    assert [h['stable_id'] for h in old.search('experiment',top_k=8)] == [
        h['stable_id'] for h in new.search('experiment',top_k=8)]
    assert len(ranker.passages) <= 20
