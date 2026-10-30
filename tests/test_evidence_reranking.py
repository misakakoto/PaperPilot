import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.evidence_reranking import rerank_evidence


class RankingModel:
    def __init__(self, transform=lambda ids: ids, finish_reason='stop'):
        self.transform = transform
        self.finish_reason = finish_reason
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = json.loads(messages[1]['content'])
        ids = [p['id'] for p in payload['passages']]
        return SimpleNamespace(text=json.dumps({'ranking': self.transform(ids)}),
                               finish_reason=self.finish_reason)


def passages(n=24):
    return [dict(stable_id=f'original-{i}', content=f'original text {i}',
                 paper_sha256='source', gold='must never leave process') for i in range(n)]


def test_ranking_returns_original_records_without_exposing_metadata():
    model = RankingModel(lambda ids: ids[::-1])
    candidates = passages()
    result = rerank_evidence('What evidence?', candidates, model)
    assert len(result['hits']) == 8
    assert len(set(result['ranked_ids'])) == 24
    assert all(hit in candidates for hit in result['hits'])
    payload = json.loads(model.calls[0][0][1]['content'])
    assert set(payload) == {'question', 'passages'}
    assert all(set(p) == {'id', 'text'} for p in payload['passages'])
    assert 'must never leave process' not in model.calls[0][0][1]['content']
    assert model.calls[0][1] == dict(max_tokens=512, json_mode=True, thinking_enabled=False)


@pytest.mark.parametrize('transform', [lambda ids: ids[:-1], lambda ids: ids+[ids[0]],
    lambda ids: ['invented']+ids[1:], lambda ids: [None]+ids[1:]])
def test_invalid_permutations_are_not_repaired_or_silently_used(transform):
    with pytest.raises(ValueError, match='exact permutation'):
        rerank_evidence('question', passages(), RankingModel(transform))


def test_truncated_response_rejected():
    with pytest.raises(ValueError, match='incomplete'):
        rerank_evidence('question', passages(), RankingModel(finish_reason='length'))


def test_character_cap_and_order_are_deterministic():
    candidates = passages(2)
    candidates[0]['content'] = 'x'*4000
    a, b = RankingModel(), RankingModel()
    rerank_evidence('question', candidates, a)
    rerank_evidence('question', candidates[::-1], b)
    assert a.calls == b.calls
    assert max(len(p['text']) for p in json.loads(a.calls[0][0][1]['content'])['passages']) == 3500


def test_budget_errors_propagate_without_retry():
    class BudgetModel:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError('budget exhausted')
    model = BudgetModel()
    with pytest.raises(RuntimeError, match='budget exhausted'):
        rerank_evidence('question', passages(), model)
    assert model.calls == 1


@pytest.mark.parametrize('candidates', [[], passages(25), passages(1)*2])
def test_bad_candidate_sets_never_call_provider(candidates):
    model = RankingModel()
    with pytest.raises(ValueError):
        rerank_evidence('question', candidates, model)
    assert model.calls == []


def test_durable_budget_counts_rerank_and_blocks_a_second_attempt(tmp_path):
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.durable_reader import DurableResearchModel
    from paperpilot_agentic.llm_runtime import PersistentLLMRuntime, LLMBudgetExhausted
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.providers import LLMResponse

    class Provider:
        calls = 0
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            self.calls += 1
            payload = json.loads(next(m['content'] for m in messages if m['role']=='user'))
            ids = [p['id'] for p in payload['passages']]
            return LLMResponse(text=json.dumps({'ranking':ids}), prompt_tokens=100,
                               completion_tokens=30, finish_reason='stop')
    db = RuntimeDatabase(tmp_path/'budget.sqlite3')
    # One general call, plus the two reserved synthesis/repair calls.
    run_id = db.create_run('question', Budget(max_llm_calls=3))
    provider = Provider()
    model = DurableResearchModel(PersistentLLMRuntime(db, provider),run_id,stage='evidence_rerank')
    assert len(rerank_evidence('question',passages(),model)['hits']) == 8
    assert db.get_run(run_id)['used_llm_calls'] == 1
    assert db.get_run(run_id)['used_tokens'] == 130
    assert db.list_llm_calls(run_id)[0]['stage'] == 'evidence_rerank'
    with pytest.raises(LLMBudgetExhausted):
        rerank_evidence('question',passages(),model)
    assert provider.calls == 1
