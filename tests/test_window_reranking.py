import pytest
from paperpilot_retrieval.window_reranking import WindowReranker


class Ranker:
    name = 'unit'
    def __init__(self):
        self.inputs = []
    def score(self, query, passages):
        self.inputs.extend(passages)
        return [9.0 if 'target' in p else 0.0 for p in passages]


def test_window_score_maps_back_to_original_paragraph_and_covers_tail():
    inner = Ranker()
    ranker = WindowReranker(inner, words=10, overlap=2, max_windows=3)
    values = ranker.score('question', ['short paragraph', ' '.join(['padding']*99+['target'])])
    assert values == [0.0, 9.0]
    assert len(inner.inputs) == 4
    assert all(len(p.split()) <= 10 for p in inner.inputs)
    assert ranker.score('question', []) == []


def test_single_short_paragraph_is_unchanged():
    inner = Ranker()
    assert WindowReranker(inner).score('question', ['some target phrase']) == [9.0]
    assert inner.inputs == ['some target phrase']


def test_bad_window_score_dimensions_rejected():
    class Bad:
        def score(self, query, passages):
            return []
    with pytest.raises(ValueError, match='scores'):
        WindowReranker(Bad()).score('q', ['passage'])
