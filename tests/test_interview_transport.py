"""Local transport failures must not become billable model attempts."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from evals.interview import run, transport
from evals.interview.budget import save


@pytest.mark.parametrize('reachable', [False, True])
def test_transport_probe_is_zero_call_and_keeps_distinct_receipts(tmp_path, monkeypatch, reachable):
    settings = SimpleNamespace(base_url='https://example.invalid/v1')
    monkeypatch.setattr(transport, 'probe_provider_connectivity', lambda *args, **kwargs: {
        'reachable': reachable, 'error_type': None if reachable else 'PermissionError'})
    first = transport.transport_preflight(settings, tmp_path)
    second = transport.transport_preflight(settings, tmp_path)
    assert first == second
    assert first['reachable'] is reachable
    assert first['actual_requests'] == first['billable_requests'] == 0
    assert first['model_availability'] == 'NOT_TESTED'
    assert len(list(tmp_path.glob('transport-probe-*.json'))) == 2
    assert not (tmp_path/'preflight-result.json').exists()
    assert not (tmp_path/'receipts').exists()


def execution_fixture(tmp_path, monkeypatch):
    settings = SimpleNamespace(provider='qwen', model='fixed', api_key='fixture',
        base_url='https://example.invalid/v1', temperature=0, thinking_enabled=False, max_tokens=100)
    save(tmp_path/'original-sources.json', {})
    protocol = {'identity': {'model': 'fixed', 'provider': 'qwen', 'source_sha256': 'fixed'},
        'temperature': 0, 'thinking_enabled': False, 'max_output': 100,
        'base_url_sha256': hashlib.sha256(settings.base_url.encode()).hexdigest(),
        'source_snapshot_sha256': hashlib.sha256((tmp_path/'original-sources.json').read_bytes()).hexdigest(),
        'cases': [{'id': 'first', 'question': 'original question', 'paper_ids': ['paper']}]}
    save(tmp_path/'protocol.json', protocol)
    monkeypatch.setattr(run, 'FOLDER', tmp_path)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path)
    monkeypatch.setattr(run, 'local_configuration', lambda: settings)
    monkeypatch.setattr(run, 'source_fingerprint', lambda root: 'fixed')
    monkeypatch.setattr('paperpilot_agentic.providers.OpenAICompatibleLLM',
                        lambda *args, **kwargs: pytest.fail('Model must not be constructed'))
    monkeypatch.setattr('paperpilot_agentic.bootstrap.build_runtime',
                        lambda *args, **kwargs: pytest.fail('Runtime must not enqueue'))
    return run.campaign_budget(protocol)


def test_transport_failure_stops_before_provider_and_preserves_budget_on_refresh(tmp_path, monkeypatch):
    budget = execution_fixture(tmp_path, monkeypatch)
    before = budget.snapshot()
    monkeypatch.setattr(transport, 'probe_provider_connectivity', lambda *args, **kwargs: {
        'reachable': False, 'error_type': 'PermissionError'})
    for _ in range(2):
        run.execute(case_ids=['first'], stage='answers')
        assert budget.snapshot() == before
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['independent_source_review'] == 'NOT_RUN_TRANSPORT_UNAVAILABLE'
    assert summary['not_run'] == ['first']
    assert len(list(tmp_path.glob('transport-probe-*.json'))) == 2
    assert not (tmp_path/'first-submission.json').exists()
    assert not (tmp_path/'preflight-result.json').exists()
    assert not (budget.folder/'stopped.json').exists()


def test_existing_unknown_remains_blocked_without_even_a_transport_probe(tmp_path, monkeypatch):
    budget = execution_fixture(tmp_path, monkeypatch)
    with budget.transaction() as conn:
        conn.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',
                     (1, 'UNKNOWN', 2128, 0, 'fixed', 'preflight', 0, None))
    before = budget.snapshot()
    monkeypatch.setattr(transport, 'transport_preflight',
                        lambda *args: pytest.fail('Transport cannot reopen UNKNOWN'))
    run.execute(case_ids=['first'], stage='answers')
    assert budget.snapshot() == before
    assert before['actual_requests'] == 1 and before['blocked']


def test_connectivity_cli_does_not_initialize_campaign_or_execute_models(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.argv', ['run', '--stage', 'connectivity', '--campaign-dir', str(tmp_path/'outputs/probe')])
    monkeypatch.setattr('paperpilot_agentic.config.RuntimeSettings.from_env',
                        lambda **kwargs: SimpleNamespace(base_url='https://example.invalid/v1'))
    monkeypatch.setattr(transport, 'probe_provider_connectivity', lambda *args, **kwargs: {'reachable': True})
    monkeypatch.setattr(run, 'campaign_budget', lambda *args: pytest.fail('No campaign initialization'))
    monkeypatch.setattr(run, 'execute', lambda **kwargs: pytest.fail('No paid execution'))
    run.main()
    assert len(list((tmp_path/'outputs/probe').glob('transport-probe-*.json'))) == 1


def test_connectivity_cli_rejects_combined_paid_execution(monkeypatch):
    monkeypatch.setattr('sys.argv', ['run', '--stage', 'connectivity', '--execute-paid'])
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
