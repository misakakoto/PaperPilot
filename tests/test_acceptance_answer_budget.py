import pytest

from evals.interview.run import answer_budget
from evals.interview.budget import CampaignBudget, RecordedProvider
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime, LLMBudgetExhausted
from test_interview_campaign import Provider


def test_legacy_protocol_keeps_its_original_per_run_ceiling():
    assert answer_budget({'max_tokens': 300000}).max_tokens == 150000
    assert answer_budget({'max_tokens': 300000, 'answer_run_max_tokens': 300000}).max_tokens == 300000


@pytest.mark.parametrize('tokens', [True, 0, -1, 300001, '300000'])
def test_run_allowance_cannot_exceed_the_authorized_campaign(tokens):
    with pytest.raises(ValueError):
        answer_budget({'max_tokens': 300000, 'answer_run_max_tokens': tokens})


def requests():
    # Held-out ASCII sources: preserve full messages rather than trimming
    # evidence to obtain budget admission. No actual provider dispatch here.
    return [{'messages': [{'role': 'user', 'content': 'source ' * 9200}], 'max_tokens': 4000,
             'reserve_class': 'repair', 'stage': 'json_repair'},
            {'messages': [{'role': 'user', 'content': 'source ' * 9200}], 'max_tokens': 5000,
             'reserve_class': 'synthesis', 'stage': 'answer_first'}]


def test_known_repair_and_audit_fit_frozen_run_allowance_without_spending(tmp_path):
    provider = Provider()
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    for cap, fits in [(150000, False), (300000, True)]:
        rid = db.create_run('Original unchanged request', answer_budget({
            'max_tokens': 300000, 'answer_run_max_tokens': cap}))
        with db.transaction() as conn:
            conn.execute('UPDATE runs SET used_tokens=?,used_llm_calls=? WHERE run_id=?',
                         (19532, 2, rid))
        runtime = PersistentLLMRuntime(db, provider)
        before = runtime.budget_snapshot(rid)
        if fits:
            runtime.ensure_capacity(rid, requests(), reserve_class='synthesis')
        else:
            with pytest.raises(LLMBudgetExhausted):
                runtime.ensure_capacity(rid, requests(), reserve_class='synthesis')
        assert runtime.budget_snapshot(rid) == before
    assert provider.calls == 0


def test_larger_frozen_run_never_bypasses_shared_campaign_budget(tmp_path):
    campaign = CampaignBudget(tmp_path / 'receipts', {'model': 'fixed'},
                              max_requests=30, max_tokens=300000)
    with campaign.transaction() as conn:
        conn.execute("INSERT INTO attempts VALUES(1,'RETURNED',200000,200000,'fixed','previous',1,NULL)")
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Unchanged question', answer_budget({'max_tokens': 300000,
                                                          'answer_run_max_tokens': 300000}))
    provider = Provider()
    bridge = PersistentLLMRuntime(db, RecordedProvider(provider, campaign, phase='bounded-repair'))
    before = campaign.snapshot()
    with pytest.raises(LLMBudgetExhausted):
        bridge.ensure_capacity(rid, requests(), reserve_class='synthesis')
    assert campaign.snapshot() == before
    assert provider.calls == db.get_run(rid)['used_llm_calls'] == 0
