import copy
import json

import httpcore
import httpx
from openai import APIConnectionError
import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from scripts.reconcile_prototype_connect_denial import reconcile


class Denied:
    model = 'local-connect-denial-fixture'

    def complete(self, messages, *, before_attempt=None, **kwargs):
        if before_attempt:
            before_attempt()
        os_error = PermissionError(13, 'fixture local denial')
        os_error.winerror = 10013
        core = httpcore.ConnectError('fixture')
        core.__cause__ = os_error
        transport = httpx.ConnectError('fixture')
        transport.__cause__ = core
        error = APIConnectionError(request=httpx.Request('POST', 'https://example.test'))
        error.__cause__ = transport
        raise error


def failed(tmp_path, monkeypatch):
    rt = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Denied())
    monkeypatch.setattr(rt.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'baseline'}])
    h = PrototypeHarness(rt)
    rid = h.start('goal', 'local-network-denial-001', 'parent', ['source'])['run']['run_id']
    h.advance(rid)
    assert h._load(rid)['phase'] == 'UNKNOWN'
    return rt, h, rid


def test_narrow_local_denial_reconciliation_preserves_attempt_and_reservation(tmp_path, monkeypatch):
    rt, h, rid = failed(tmp_path, monkeypatch)
    before = copy.deepcopy(rt.db.get_run(rid))
    ledger = rt.llm_runtime.budget_snapshot(rid)
    record = reconcile(h, rid)
    after = rt.db.get_run(rid)
    assert h._load(rid)['phase'] == 'READY' and h._load(rid)['turn'] == 0
    assert after['used_llm_calls'] == before['used_llm_calls'] == 1
    assert after['budget'] == before['budget'] and after['created_at'] == before['created_at']
    current = rt.llm_runtime.budget_snapshot(rid)
    assert current['unknown_results'] == 0
    assert current['reserved_unknown_tokens'] == ledger['reserved_unknown_tokens']
    assert record['reservation_retained'] and not record['budget_reset']
    with rt.db.connect() as conn:
        row = conn.execute('SELECT * FROM research_episode_turns WHERE run_id=?', (rid,)).fetchone()
    assert row['state'] == 'NOT_DISPATCHED' and row['raw_json'] is None
    with pytest.raises(ValueError):
        reconcile(h, rid)


@pytest.mark.parametrize('condition', ['read', 'other_os_error', 'short_chain', 'http_response', 'saved_response', 'wrong_stage'])
def test_ambiguous_or_responded_requests_remain_unknown(tmp_path, monkeypatch, condition):
    rt, h, rid = failed(tmp_path, monkeypatch)
    with rt.db.transaction() as conn:
        row = conn.execute('SELECT * FROM llm_attempts WHERE run_id=?', (rid,)).fetchone()
        failure = json.loads(row['failure_json'])
        if condition == 'read':
            failure['transport_phase'] = 'read'
        elif condition == 'other_os_error':
            failure['cause_chain'][-1]['winerror'] = 10061
        elif condition == 'short_chain':
            failure['cause_chain'] = failure['cause_chain'][:-1]
        elif condition == 'http_response':
            failure['http_status'] = 403
        elif condition == 'saved_response':
            conn.execute("UPDATE research_episode_turns SET raw_json='{}' WHERE run_id=?", (rid,))
        else:
            conn.execute("UPDATE llm_attempts SET stage='other' WHERE run_id=?", (rid,))
        conn.execute('UPDATE llm_attempts SET failure_json=? WHERE attempt_id=?', (json.dumps(failure), row['attempt_id']))
    before = copy.deepcopy(h._load(rid))
    with pytest.raises(ValueError):
        reconcile(h, rid)
    assert h._load(rid) == before and rt.llm_runtime.budget_snapshot(rid)['unknown_results'] == 1
