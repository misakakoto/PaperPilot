import json

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.research_harness import ResearchHarness
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.experiment_import import register_experiment
from test_experiment_import import pack


class Reader:
    model = 'fixture-import-diagnosis'

    def __init__(self):
        self.messages = []

    def complete(self, messages, before_attempt=None, **kwargs):
        before_attempt()
        self.messages.append(messages)
        task = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        if 'lead' in messages[0]['content'].split('.')[0]:
            action = ('finish', {'summary': 'Stop: predictions alone do not establish a mechanism.'}) if task['independent_analysis'] else ('delegate_analysis', {})
        elif 'coverage' in task['available_views'] and 'coverage' not in task['observations']:
            action = ('inspect_experiment', {'view': 'coverage'})
        else:
            refs = task['required_report_fact_refs']
            extra = ['coverage/2/candidate/f1'] if 'coverage' in task['observations'] else refs
            action = ('report', {'fact_refs': refs, 'interpretations': [
                {'text': 'These synthetic counts do not prove activation behavior.', 'fact_refs': extra}],
                'limitations': ['User-declared training source; no activations measured.'],
                'next_experiment': {'hypothesis': 'Measure activations before explaining the mechanism.',
                    'support_refs': refs, 'counter_refs': extra,
                    'missing_observations': ['Activation distributions'], 'falsification': 'No predicted activation difference.'}})
        return LLMResponse(text=json.dumps({'action': action[0], 'reason': 'Read only observed evidence.',
            'arguments': action[1]}), model=self.model, finish_reason='stop', prompt_tokens=20, completion_tokens=10)


def setup(tmp_path, seeds=(43,)):
    shell, path, _ = pack(tmp_path, seeds)
    reader = Reader()
    runtime = ResearchAgentRuntime(workspace=shell.workspace, db_path=tmp_path/'runtime.sqlite3',
        llm=reader, text_reader_protocol='import-test-v1')
    binding = register_experiment(runtime, path)
    return runtime, ResearchHarness(runtime), reader, binding


@pytest.mark.parametrize('seeds', [(43,), (3, 45, 81, 100)])
@pytest.mark.parametrize('mode', ['summary', 'tools'])
def test_imported_harness_uses_all_declared_seeds_and_reuses_completion(tmp_path, seeds, mode):
    runtime, harness, reader, binding = setup(tmp_path, seeds)
    result = harness.start_imported_diagnosis(binding['bundle_id'], 'imported-task-id-001', comparison_mode=mode)
    rid = result['run']['run_id']
    result = harness.advance(rid)
    episode = result['research_episode']
    assert episode['phase'] == 'COMPLETED', episode
    assert episode['seeds'] == list(seeds)
    assert episode['source_kind'] == 'imported_predictions'
    assert episode['verification']['training_source'].startswith('USER_DECLARED')
    assert len(episode['analysis']['facts']) >= len(seeds)
    calls = len(reader.messages)
    assert calls == (4 if mode == 'tools' else 3)
    assert episode['new_gpu_starts'] == 0 and not episode['candidates']
    assert not runtime.db.list_approvals(rid)
    assert harness.start_imported_diagnosis(binding['bundle_id'], 'imported-task-id-001', comparison_mode=mode)['run']['run_id'] == rid
    assert harness.advance(rid)['research_episode']['phase'] == 'COMPLETED'
    assert len(reader.messages) == calls
    with pytest.raises(ValueError, match='different inputs'):
        harness.start_imported_diagnosis('0'*64, 'imported-task-id-001', comparison_mode=mode)


def test_imported_task_cancel_does_not_start_another_call(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'cancel-import-task')['run']['run_id']
    harness.advance(rid, max_turns=1)
    harness.cancel(rid)
    harness.advance(rid)
    assert len(reader.messages) == 1


def test_imported_task_rejects_training_action_and_preserves_unknown(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'unknown-import-task')['run']['run_id']
    from paperpilot_agentic.research_diagnosis import DiagnosisHarness
    diagnosis = DiagnosisHarness(harness.runtime)
    state = diagnosis._load(rid)
    with pytest.raises(ValueError, match='lacks this action'):
        diagnosis._apply(rid, state, {'action': 'train', 'reason': 'bad', 'arguments': {}}, 1)
    def unknown(*args, before_attempt=None, **kwargs):
        before_attempt()
        reader.messages.append('unknown')
        raise TimeoutError('fixture response unknown')
    reader.complete = unknown
    result = harness.advance(rid)
    assert result['research_episode']['phase'] == 'UNKNOWN'
    harness.advance(rid)
    assert len(reader.messages) == 1
    with pytest.raises(ValueError, match='prior imported diagnosis'):
        harness.start_imported_diagnosis(binding['bundle_id'], 'another-request-after-unknown')


def test_imported_workbench_is_idempotent_and_rejects_browser_paths(tmp_path):
    runtime, _, reader, binding = setup(tmp_path)
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background
    background = Background()
    wb = ResearchWorkbench(runtime, background)
    payload = {'request_id': 'workbench-import-task', 'action': 'imported_diagnosis', 'bundle_id': binding['bundle_id']}
    a = wb.submit(payload)
    repeated = wb.submit(payload)
    assert repeated['run_id'] == a['run_id'] and repeated['reused'] is True
    assert len(background.submitted) == 1 and not reader.messages
    with pytest.raises(ValueError, match='Unknown submission fields'):
        wb.submit({**payload, 'manifest': 'C:/untrusted/file'})


def test_registered_bundle_api_lists_only_host_assets(tmp_path):
    runtime, _, reader, binding = setup(tmp_path)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        response = client.get('/agentic/imported-experiments')
        assert response.status_code == 200
        item = response.json()['items'][0]
        assert item['bundle_id'] == binding['bundle_id'] and item['seeds'] == [43]
        assert 'assets' not in item and not reader.messages


def test_imported_process_death_after_return_reuses_response(tmp_path, monkeypatch):
    _, harness, reader, binding = setup(tmp_path)
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'interrupted-import-task')['run']['run_id']
    from paperpilot_agentic.research_diagnosis import DiagnosisHarness
    original = DiagnosisHarness._apply
    class ProcessDeath(BaseException):
        pass
    def die(*args, **kwargs):
        raise ProcessDeath()
    monkeypatch.setattr(DiagnosisHarness, '_apply', die)
    with pytest.raises(ProcessDeath):
        harness.advance(rid, max_turns=1)
    assert len(reader.messages) == 1
    monkeypatch.setattr(DiagnosisHarness, '_apply', original)
    harness.advance(rid, max_turns=1)
    assert len(reader.messages) == 1  # persisted RETURNED is reused


def test_new_request_id_cannot_compete_but_completed_independent_repeat_is_allowed(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'first-source-task')['run']['run_id']
    with pytest.raises(ValueError, match='prior imported diagnosis'):
        harness.start_imported_diagnosis(binding['bundle_id'], 'second-source-task')
    harness.advance(rid)
    assert len(reader.messages) == 4
    second = harness.start_imported_diagnosis(binding['bundle_id'], 'second-source-task')['run']['run_id']
    assert second != rid
    harness.advance(second)
    assert len(reader.messages) == 8


def test_repeated_missing_reason_stops_after_one_repair_and_survives_refresh(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    def malformed(messages, before_attempt=None, **kwargs):
        before_attempt()
        reader.messages.append(messages)
        return LLMResponse(text=json.dumps({'action': 'delegate_analysis', 'arguments': {}}),
            model=reader.model, finish_reason='stop', prompt_tokens=20, completion_tokens=10)
    reader.complete = malformed
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'repeated-invalid-envelope')['run']['run_id']
    first = harness.advance(rid, max_turns=1)
    assert first['research_episode']['phase'] == 'READY'
    # Refresh/reconstruct the harness; rejection allowance must not reset.
    result = ResearchHarness(harness.runtime).advance(rid)
    assert len(reader.messages) == 2
    assert result['research_episode']['phase'] == 'PARTIAL'
    assert 'Repeated rejected diagnostic action' in result['research_episode']['stop_reason']
    assert not result['research_episode']['analysis']
    harness.advance(rid)
    assert len(reader.messages) == 2


def test_successful_action_resets_consecutive_rejection_allowance(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    original = reader.complete
    calls = 0
    def intermittently_invalid(messages, before_attempt=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls in (1, 3):
            before_attempt()
            reader.messages.append(messages)
            return LLMResponse(text=json.dumps({'action': 'delegate_analysis', 'arguments': {}}),
                model=reader.model, finish_reason='stop', prompt_tokens=20, completion_tokens=10)
        return original(messages, before_attempt=before_attempt, **kwargs)
    reader.complete = intermittently_invalid
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'intermittent-invalid-envelope')['run']['run_id']
    result = harness.advance(rid)
    assert result['research_episode']['phase'] == 'COMPLETED'
    assert len(reader.messages) == 6


def test_scope_contract_reaches_both_roles_and_preserves_all_observed_evidence(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    rid = harness.start_imported_diagnosis(binding['bundle_id'], 'scope-contract-both-roles')['run']['run_id']
    result = harness.advance(rid)
    assert result['research_episode']['phase'] == 'COMPLETED'
    roles = set()
    for messages in reader.messages:
        roles.add(messages[0]['content'].split(' research agent')[0].split()[-1])
        task = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        contract = task['scientific_scope_contract']
        assert contract['version'] == 'prediction-diagnosis-scope-v2'
        assert contract['training_assurance'].startswith('USER_DECLARED')
        assert 'Object counts and object sizes' in contract['not_observed']
        assert 'paired' in task['observations']
        assert task['required_report_fact_refs'] == ['paired/0/delta/f1']
        assert 'not an independent scientific correctness verdict' in contract['assurance']
    assert roles == {'lead', 'analyst'}
