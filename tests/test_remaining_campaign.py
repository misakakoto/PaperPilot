import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from evals.interview.budget import CampaignBudget, save, canonical
from evals.interview.remaining import grant_remaining, RemainingCampaignBudget


def parent(tmp_path):
    folder = tmp_path/'parent'; folder.mkdir()
    identity = {'model': 'm', 'provider': 'qwen'}
    save(folder/'protocol.json', {'identity': identity})
    budget = CampaignBudget(folder/'receipts', identity, max_requests=30, max_tokens=300000)
    with budget.transaction() as conn:
        conn.execute('INSERT INTO attempts VALUES(1,?,?,?,?,?,?,?)',
                     ('RETURNED', 190000, 185105, 'm', 'old', 1, None))
        conn.execute('INSERT INTO attempts VALUES(2,?,?,?,?,?,?,?)',
                     ('FAILED', 28618, 0, 'm', 'old', 1, '{}'))
    return folder


def test_grant_is_unique_idempotent_and_carries_failed_reservation(tmp_path):
    folder = parent(tmp_path)
    before = (folder/'receipts/spending.sqlite3').read_bytes()
    grant = grant_remaining(folder, tmp_path/'child', 'User authorizes remaining balance')
    assert grant['max_requests'] == 28 and grant['max_tokens'] == 86277
    assert grant == grant_remaining(folder, tmp_path/'child', 'same authorized balance')
    with pytest.raises(ValueError): grant_remaining(folder, tmp_path/'other', 'same parent')
    assert before == (folder/'receipts/spending.sqlite3').read_bytes()


def test_parent_change_blocks_child_and_child_refresh_keeps_cost(tmp_path):
    folder = parent(tmp_path)
    grant = grant_remaining(folder, tmp_path/'child', 'remaining')
    budget = RemainingCampaignBudget(tmp_path/'child/receipts', {'model': 'm'}, grant=grant)
    budget.dispatch(model='m', phase='answer', request={'messages': [], 'max_tokens': 128},
        operation=lambda: {'text': 'ok', 'prompt_tokens': 10, 'completion_tokens': 20}, receipt=lambda r: r)
    restored = RemainingCampaignBudget(tmp_path/'child/receipts', {'model': 'm'}, grant=grant)
    assert restored.snapshot()['reported_tokens'] == 30 and restored.snapshot()['actual_requests'] == 1
    with CampaignBudget(folder/'receipts', {'model': 'm', 'provider': 'qwen'}, max_requests=30, max_tokens=300000).transaction() as conn:
        conn.execute('UPDATE attempts SET tokens=185106 WHERE id=1')
    with pytest.raises(ValueError): restored._reserve('m', 'answer', {'messages': [], 'max_tokens': 128})


def test_explicit_older_model_recovery_counts_combined_probe_and_never_falls_back(tmp_path):
    from evals.interview.selection_preflight import selection_preflight
    from paperpilot_agentic.providers import LLMResponse
    from paperpilot_agentic.model_policy import PRIMARY_MODEL
    folder = parent(tmp_path)
    child = tmp_path/'child'
    grant = grant_remaining(folder, child, 'User delegates selection of an older free model')
    identity = {'model': 'qwen3-vl-plus', 'provider': 'qwen',
                'selection_models': ['qwen3-vl-plus']}
    budget = RemainingCampaignBudget(child/'receipts', identity, grant=grant)
    sent = []
    class Probe:
        model = 'qwen3-vl-plus'
        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            sent.append(messages)
            return LLMResponse('{"color":"red"}', 100, 10, finish_reason='stop')
    from paperpilot_agentic.config import RuntimeSettings
    # Construct without loading user secrets or .env.
    settings = RuntimeSettings(provider='qwen', api_key='offline-fixture',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', model='qwen3-vl-plus',
        thinking_enabled=False, temperature=0, max_tokens=128, timeout_seconds=60,
        workspace=child, runtime_db=child/'runtime.sqlite3', memory_db=child/'corpus.sqlite3',
        memory_namespace='offline', paper_roots=())
    assert selection_preflight(settings, budget, child, factory=lambda settings: Probe())
    assert budget.snapshot()['actual_requests'] == 1
    assert budget.snapshot()['reported_tokens'] == 110
    assert sent[0][0]['role'] == 'system'
    assert sent[0][1]['content'][0]['type'] == 'image_url'
    assert selection_preflight(settings, budget, child, factory=lambda settings: pytest.fail('No resend'))
    with pytest.raises(ValueError): budget._validate_model(PRIMARY_MODEL, phase='answer')
    budget._validate_model('qwen3-vl-plus', phase='answer')
    assert grant['max_tokens'] == 86277
    restored = RemainingCampaignBudget(child/'receipts', identity, grant=grant)
    assert restored.snapshot()['actual_requests'] == 1


def test_explicit_single_model_quota_failure_stops_without_switch(tmp_path):
    folder = parent(tmp_path)
    child = tmp_path/'child'
    grant = grant_remaining(folder, child, 'Older model only')
    identity = {'model': 'qwen3-vl-plus', 'selection_models': ['qwen3-vl-plus']}
    budget = RemainingCampaignBudget(child/'receipts', identity, grant=grant)
    with budget.transaction() as conn:
        conn.execute('INSERT INTO attempts VALUES(1,?,?,?,?,?,?,?)',
            ('FAILED', 1234, 0, 'qwen3-vl-plus', 'capability:selection', 1,
             canonical({'provider_code': 'AllocationQuota.FreeTierOnly', 'http_status':403})))
    assert budget.snapshot()['blocked']
    assert budget.snapshot()['unsettled_upper_bound'] == 1234
    with pytest.raises(ValueError): budget._validate_model('qwen3-vl-plus', phase='capability:selection')


@pytest.mark.parametrize('deadline', [None, 90])
def test_freeze_revision_keeps_unique_grant_and_same_ledger(tmp_path, monkeypatch, deadline):
    import sqlite3
    from dataclasses import replace
    from evals.interview import run
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.reviewed_sources import ReviewedSources
    original = parent(tmp_path)
    child = tmp_path/'child'; child.mkdir()
    grant = grant_remaining(original, child, 'Continue only within the remainder')
    identity = {'model': 'm', 'provider': 'qwen', 'source_sha256': 'old', 'selection_models': ['m']}
    old = RemainingCampaignBudget(child/'receipts', identity, grant=grant)
    old.dispatch(model='m', phase='answer', request={'messages': [], 'max_tokens': 128},
        operation=lambda: {'text': 'ok', 'prompt_tokens': 10, 'completion_tokens': 20}, receipt=lambda r: r)
    cases = [{'id': 'unchanged', 'question': 'unchanged'}]
    save(child/'protocol.json', {'identity': identity, 'cases': cases,
        'remaining_grant': grant, 'max_requests': 28, 'max_tokens': 86277,
        'historical_cost': {'protocol_path': str(original/'protocol.json')},
        'supplemental_authorization': {'human_instruction': 'Original remainder authorization'}})
    if deadline is not None:
        path = child/'protocol.json'
        parent_protocol = json.loads(path.read_text(encoding='utf-8'))
        parent_protocol['timeout_seconds'] = deadline
        path.write_text(json.dumps(parent_protocol), encoding='utf-8')
    for name in ('corpus.sqlite3', 'runtime.sqlite3'):
        with sqlite3.connect(child/name) as conn: conn.execute('CREATE TABLE placeholder (id INTEGER)')
    revision = child/'revisions/0001'
    # Do not load the user's .env or secret bundle into the test process.
    monkeypatch.setattr('paperpilot_agentic.config.load_dotenv', lambda **kwargs: False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'offline-fixture')
    settings = replace(RuntimeSettings.from_env(), model='m', provider='qwen',
                       memory_db=child/'corpus.sqlite3', workspace=revision, thinking_enabled=False)
    monkeypatch.setattr(RuntimeSettings, 'from_env', lambda: settings)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', child)
    monkeypatch.setattr(run, 'FOLDER', revision)
    monkeypatch.setattr(run, 'local_configuration', lambda: settings)
    monkeypatch.setattr(run, 'acceptance_cases', lambda root: cases)
    monkeypatch.setattr(run, 'source_fingerprint', lambda root: 'repaired')
    monkeypatch.setattr(ReviewedSources, 'configured', lambda: ReviewedSources(tmp_path/'assets'))
    store = SimpleNamespace(paper_metadata=lambda ids: [{'sha256': s} for s in (run.CHANGE, run.FLOW)],
                            paper_outline=lambda sha: [], get_chunks=lambda ids: [])
    monkeypatch.setattr('paperpilot_agentic.bootstrap.build_runtime', lambda settings:
                        SimpleNamespace(search_adapter=SimpleNamespace(store=store), text_reader_protocol='fixed'))
    run.freeze(continuation=True, continuation_note='User requested the source repair and continuation')
    protocol = json.loads((revision/'protocol.json').read_text(encoding='utf-8'))
    refreshed = run.campaign_budget(protocol)
    assert isinstance(refreshed, RemainingCampaignBudget)
    assert protocol['remaining_grant'] == grant
    assert protocol['identity']['selection_models'] == ['m']
    assert refreshed.path == old.path
    assert refreshed.snapshot()['actual_requests'] == 1
    assert refreshed.snapshot()['reported_tokens'] == 30
    assert refreshed.snapshot()['max_tokens'] == 86277
    assert protocol['timeout_seconds'] == (60 if deadline is None else deadline)
    assert protocol['request_timeout']['vision_seconds'] == 60
    assert protocol['request_timeout'] == protocol['identity']['request_timeout']


@pytest.mark.parametrize('code,status,backup_allowed', [('model_not_found',404,True),
    ('AllocationQuota.FreeTierOnly',403,True),('access_denied',403,False),(None,None,False)])
def test_selection_only_switch_and_failed_cost_retention(tmp_path, code, status, backup_allowed):
    from paperpilot_agentic.model_policy import PRIMARY_MODEL, BACKUP_MODEL
    from evals.interview.budget import UncertainRequest
    folder = parent(tmp_path)
    grant = grant_remaining(folder, tmp_path/'child', 'remaining')
    identity = {'model':PRIMARY_MODEL,'provider':'qwen'}
    budget = RemainingCampaignBudget(tmp_path/'child/receipts', identity, grant=grant)
    with budget.transaction() as conn:
        conn.execute('INSERT INTO attempts VALUES(1,?,?,?,?,?,?,?)',
            ('FAILED' if status else 'UNKNOWN',1000,0,PRIMARY_MODEL,'capability:selection',1,
             canonical({'provider_code':code,'http_status':status})))
    assert budget.snapshot()['unsettled_upper_bound'] == 1000
    if backup_allowed:
        budget._validate_model(BACKUP_MODEL,phase='capability:selection')
        assert not budget.snapshot()['blocked']
        # Ordinary answer fallback is never allowed during this selection gap.
        with pytest.raises(ValueError): budget._validate_model(BACKUP_MODEL,phase='answer')
        save(tmp_path/'child/selected-model.json',{'model':BACKUP_MODEL,'identity':identity})
        budget._validate_model(BACKUP_MODEL,phase='answer')
        with pytest.raises(ValueError): budget._validate_model(PRIMARY_MODEL,phase='answer')
    else:
        assert budget.snapshot()['blocked']
        with pytest.raises(ValueError): budget._validate_model(BACKUP_MODEL,phase='capability:selection')
