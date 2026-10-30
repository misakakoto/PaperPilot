import copy
import json

import pytest

from paperpilot_agentic.prototype_model_migration import migrate_quota_model
from test_prototype_budget_recovery import blocked


def fixture(tmp_path, monkeypatch):
    rt, h, rid = blocked(tmp_path, monkeypatch)
    with rt.db.transaction() as c:
        c.execute('''INSERT INTO llm_attempts
            (attempt_id,run_id,ordinal,stage,model,state,reserved_tokens,usage_known,
             request_bytes,request_chars,request_sha256,max_output_tokens,failure_json,created_at)
             VALUES(?,?,1,'research.lead.1','old','FAILED',100,0,1,1,'original-request',10,?,'fixture')''',
             ('attempt-fixture', rid, json.dumps({'category': 'BILLING_OR_QUOTA', 'http_status': 403,
                                                'provider_code': 'AllocationQuota.FreeTierOnly'})))
    monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda _: {'unknown_results': 0})
    rt.llm_runtime.model.model = 'replacement'
    return rt, h, rid


def test_migration_preserves_work_budget_and_original_failed_request(tmp_path, monkeypatch):
    rt, h, rid = fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(h._load(rid))
    run = rt.db.get_run(rid)
    migrate_quota_model(h, rid)
    after = h._load(rid)
    assert after['work'] == before['work'] and after['turn'] == 1 and after['phase'] == 'READY'
    assert after['model_protocol'] != before['model_protocol']
    assert rt.db.get_run(rid)['budget'] == run['budget']
    assert rt.db.get_run(rid)['used_llm_calls'] == run['used_llm_calls']
    with rt.db.connect() as c:
        assert c.execute('select state from research_episode_turns where run_id=?', (rid,)).fetchone()[0] == 'PROVIDER_REJECTED'
        assert c.execute('select request_sha256 from llm_attempts where run_id=?', (rid,)).fetchone()[0] == 'original-request'


@pytest.mark.parametrize('case', ['unknown', 'nonquota', 'returned'])
def test_uncertain_or_successful_operations_cannot_be_replayed(tmp_path, monkeypatch, case):
    rt, h, rid = fixture(tmp_path, monkeypatch)
    if case == 'unknown':
        monkeypatch.setattr(rt.llm_runtime, 'budget_snapshot', lambda _: {'unknown_results': 1})
    elif case == 'nonquota':
        with rt.db.transaction() as c:
            c.execute("update llm_attempts set failure_json='{}' where run_id=?", (rid,))
    else:
        with rt.db.transaction() as c:
            c.execute("update research_episode_turns set raw_json='{}' where run_id=?", (rid,))
    before = h._load(rid)
    with pytest.raises(ValueError):
        migrate_quota_model(h, rid)
    assert h._load(rid) == before


def test_real_provider_checkpoint_migrates_with_research_identity(tmp_path, monkeypatch):
    from dataclasses import replace
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    rt, h, rid = fixture(tmp_path, monkeypatch)
    settings = replace(RuntimeSettings.from_env(), api_key='fixture-not-a-real-key', model='old',fallback_models=())
    provider = OpenAICompatibleLLM(settings)
    old_policy = provider.policy_identity(rt.db.get_run(rid)['metadata'].get('execution_profile','frozen_acceptance'))
    rt.db.update_run_metadata(rid, {'model_policy': old_policy})
    provider.model = 'replacement'
    rt.llm_runtime.model = provider
    migrate_quota_model(h,rid)
    assert provider.for_run(rt.db,rid).model == 'replacement'
    assert rt.db.get_run(rid)['metadata']['model_policy_history'] == [old_policy]
