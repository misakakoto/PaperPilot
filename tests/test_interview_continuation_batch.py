"""New revisions carry all unknown costs; bounded batches keep frozen cases."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from evals.interview.budget import CampaignBudget, save
from evals.interview.run import retained_unknown_bounds
from evals.interview import run


def insert_unknown(budget, number, reservation=4000, state='UNKNOWN'):
    with budget.transaction() as conn:
        conn.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',
            (number, state, reservation, 0, 'fixed', 'fixture', 0, None))


def request(budget, number, **changes):
    save(budget.folder/f'request-{number:03}.json', {
        'model': 'fixed', 'messages': [{'role': 'user', 'content': 'question'}],
        'max_tokens': 100, 'thinking_enabled': False, **changes})


def test_later_continuation_carries_old_reviewed_bounds_and_new_complete_request(tmp_path):
    previous = CampaignBudget(tmp_path, {'model': 'fixed'})
    insert_unknown(previous, 1, 3000)
    insert_unknown(previous, 2, 592725)
    request(previous, 1)
    previous.authorize_continuation({'model': 'fixed', 'revision': '0001'},
        unknown_bounds={1: 8000, 2: 10000}, evidence={'human_instruction': 'continue'})
    insert_unknown(previous, 6)
    request(previous, 6)
    before = previous.snapshot()['attempts']
    assert retained_unknown_bounds(previous) == {1: 8000, 2: 10000, 6: 4000}
    assert previous.snapshot()['attempts'] == before


@pytest.mark.parametrize('change', [{'thinking_enabled': True}, {'model': 'other'}, {'max_tokens': 0}])
def test_incomplete_or_unbounded_request_cannot_be_carried(tmp_path, change):
    previous = CampaignBudget(tmp_path, {'model': 'fixed'})
    insert_unknown(previous, 1)
    request(previous, 1, **change)
    with pytest.raises(ValueError, match='complete bounded non-thinking'):
        retained_unknown_bounds(previous)


def test_new_missing_request_cannot_reuse_an_unreviewed_reservation(tmp_path):
    previous = CampaignBudget(tmp_path, {'model': 'fixed'})
    insert_unknown(previous, 6)
    with pytest.raises(ValueError, match='independently reviewed'):
        retained_unknown_bounds(previous)


@pytest.mark.parametrize('scheduled', [False, True])
def test_batched_execution_preserves_all_cases_and_does_not_run_context_early(tmp_path, monkeypatch, scheduled):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'CAMPAIGN_FOLDER', tmp_path)
    monkeypatch.setattr(run, 'FOLDER', tmp_path/'revisions/0002')
    run.FOLDER.mkdir(parents=True)
    source = run.FOLDER/'original-sources.json'
    source.write_text('{}')
    settings = SimpleNamespace(provider='qwen', model='fixed', api_key='fixture',
        base_url='http://example.invalid', temperature=0, thinking_enabled=False,
        max_tokens=100, timeout_seconds=60)
    protocol = {'identity': {'model': 'fixed', 'provider': 'qwen'}, 'temperature': 0, 'thinking_enabled': False,
        'max_output': 100, 'base_url_sha256': hashlib.sha256(settings.base_url.encode()).hexdigest(),
        'source_snapshot_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'reader_protocol': 'fixed', 'cases': [
            {'id': str(i), 'question': f'Q{i}', 'paper_ids': ['paper']} for i in range(3)]}
    protocol['identity']['source_sha256'] = 'fixed'
    save(run.FOLDER/'protocol.json', protocol)
    monkeypatch.setattr(run, 'local_configuration', lambda: settings)
    monkeypatch.setattr(run, 'source_fingerprint', lambda root: 'fixed')
    monkeypatch.setattr('evals.interview.transport.transport_preflight',
                        lambda settings, folder: {'reachable': True, 'actual_requests': 0})
    monkeypatch.setattr(run, 'model_preflight', lambda provider, folder: True)
    monkeypatch.setattr(run, 'structural_observations', lambda *args: {})
    monkeypatch.setattr('paperpilot_agentic.providers.OpenAICompatibleLLM',
                        lambda *args, **kwargs: SimpleNamespace(model='fixed'))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None)))
    monkeypatch.setattr('paperpilot_agentic.qwen_vision.QwenVisionEvidenceAdapter',
                        lambda **kwargs: SimpleNamespace(client=client))
    queued, resumed, contexts = [], [], []
    class Runtime:
        text_reader_protocol = 'fixed'
        def enqueue(self, *, question, **kwargs):
            queued.append(question)
            return {'run': {'run_id': question}}
        def resume(self, run_id):
            resumed.append(run_id)
        def status(self, run_id):
            return {'run': {'run_id': run_id, 'status': 'COMPLETED',
                            'used_llm_calls': 0, 'used_tokens': 0}}
    monkeypatch.setattr('paperpilot_agentic.bootstrap.build_runtime', lambda *args, **kwargs: Runtime())
    monkeypatch.setattr('evals.interview.context.live_context', lambda *args: contexts.append(True))
    order = ['2', '0', '1'] if scheduled else ['0', '1', '2']
    run.execute(max_new_cases=1, case_ids=[order[0]] if scheduled else None)
    assert queued == resumed == ['Q'+order[0]]
    assert not contexts
    summary = json.loads((run.FOLDER/'summary.json').read_text())
    assert summary['not_run'] == [str(i) for i in range(3) if str(i) != order[0]]
    assert summary['new_cases_executed'] == 1
    run.execute(max_new_cases=1, case_ids=[order[1]] if scheduled else None)
    assert queued == resumed == ['Q'+i for i in order[:2]]
    assert not contexts
    run.execute(max_new_cases=1, case_ids=[order[2]] if scheduled else None)
    assert queued == resumed == ['Q'+i for i in order]
    assert contexts == [True]
    assert json.loads((run.FOLDER/'protocol.json').read_text()) == protocol
    with pytest.raises(ValueError, match='frozen case IDs'):
        run.execute(case_ids=['not_frozen'])
    protocol['cases'].append({'id': 'child', 'parent_scenario_id': 'missing_parent'})
    (run.FOLDER/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='same-revision parent'):
        run.execute(case_ids=['child'])
