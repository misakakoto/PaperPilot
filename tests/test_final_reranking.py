import pytest
from evals.qasper.agent_case import paper_records
from paperpilot_retrieval import PaperKnowledgeStore
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter


class Ranker:
    def __init__(self, invalid=None):
        self.calls = []
        self.invalid = invalid
    def score(self, query, passages):
        self.calls.append((query, passages))
        if self.invalid == 'count':
            return []
        if self.invalid == 'finite':
            return [float('nan')]*len(passages)
        return [100 if 'target evidence' in p else 0 for p in passages]


def corpus(tmp_path):
    paper = dict(title='Distinctive analysis', abstract='experiment abstract',
        full_text=[dict(section_name='Results', paragraphs=[
            'experiment target evidence with detailed measurements',
            *[f'experiment irrelevant description {i}' for i in range(15)]])], figures_and_tables=[])
    _, doc, chunks = paper_records(paper, source_id='unit')
    store = PaperKnowledgeStore(tmp_path/'index.sqlite3')
    store.sync_documents([doc], chunks)
    return store, doc


def test_final_stage_ranks_pool_once_and_keeps_source_scope(tmp_path):
    store, doc = corpus(tmp_path)
    ranker = Ranker()
    adapter = PaperPilotKnowledgeSearchAdapter(store, reranker=ranker, rerank_stage='final')
    hits = adapter.search('experiment', top_k=8, paper_sha256s={doc.sha256})
    assert len(ranker.calls) == 1
    assert len(ranker.calls[0][1]) > 8
    assert len(hits) == 8
    assert 'target evidence' in hits[0]['content']
    assert all(h['paper_sha256'] == doc.sha256 for h in hits)
    assert all('final_cross_encoder' in h['channels'] for h in hits)
    assert adapter.search('experiment', top_k=8, paper_sha256s=set()) == []
    assert len(ranker.calls) == 1


@pytest.mark.parametrize('invalid', ['count', 'finite'])
def test_invalid_reranker_never_publishes_unvalidated_rankings(tmp_path, invalid):
    store, doc = corpus(tmp_path)
    with pytest.raises(ValueError, match='reranker'):
        PaperPilotKnowledgeSearchAdapter(store, reranker=Ranker(invalid), rerank_stage='final').search(
            'experiment', paper_sha256s={doc.sha256})


def test_unknown_stage_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='rerank_stage'):
        PaperPilotKnowledgeSearchAdapter(None, rerank_stage='typo')


def test_runtime_composition_selects_validated_stage_and_model_defaults(tmp_path, monkeypatch):
    from paperpilot_agentic import bootstrap
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_retrieval.reranking import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_FILE
    store, _ = corpus(tmp_path)
    seen = {}
    def make_ranker(**kwargs):
        seen.update(kwargs)
        return Ranker()
    monkeypatch.setattr(bootstrap, 'OnnxCrossEncoderReranker', make_ranker)
    monkeypatch.setenv('PAPERPILOT_ENABLE_RERANKER', 'true')
    monkeypatch.setenv('PAPERPILOT_ENABLE_DENSE', 'false')
    for name in ('PAPERPILOT_RERANKER_MODEL','PAPERPILOT_RERANKER_FILE','PAPERPILOT_RERANK_STAGE'):
        monkeypatch.delenv(name, raising=False)
    settings = RuntimeSettings('deepseek', None, 'https://api.deepseek.com', 'unused', False,
        0.0, 100, 10, tmp_path, tmp_path/'runtime.sqlite3', tmp_path/'index.sqlite3', 'default', ())
    runtime = bootstrap.build_runtime(settings)
    assert runtime.search_adapter.rerank_stage == 'final'
    assert seen['model_name'] == DEFAULT_RERANKER_MODEL
    assert seen['model_file'] == DEFAULT_RERANKER_FILE
