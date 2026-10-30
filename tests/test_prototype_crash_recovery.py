"""Crash windows use maintained providers; no generated code is executed."""
import copy
import json

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from test_prototype_workspace import Reader
from paperpilot_agentic.research_training import episode_folder


def setup(tmp_path, monkeypatch):
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'crash-fixture', 'parent', ['source'])['run']['run_id']
    return runtime, harness, rid


def test_saved_local_result_replays_without_duplicate_step_or_model_call(tmp_path, monkeypatch):
    runtime, harness, rid = setup(tmp_path, monkeypatch)
    original = harness._save

    def die(*args, **kwargs):
        if kwargs.get('applied_turn') == 1:
            raise KeyboardInterrupt('Process death before task state commit')
        return original(*args, **kwargs)

    monkeypatch.setattr(harness, '_save', die)
    with pytest.raises(KeyboardInterrupt):
        harness.advance(rid, max_turns=1)
    assert harness._load(rid)['turn'] == 0
    assert harness._load(rid)['work']['tasks'] == {}
    before = runtime.db.get_run(rid)
    steps = copy.deepcopy(runtime.db.list_steps(rid))
    calls = copy.deepcopy(runtime.db.list_tool_calls(rid))
    assert len(steps) == len(calls) == 1 and calls[0]['status'] == 'COMPLETED'
    assert before['used_llm_calls'] == 1
    PrototypeHarness(runtime).advance(rid, max_turns=1)
    after = runtime.db.get_run(rid)
    assert after['used_llm_calls'] == before['used_llm_calls']
    assert after['used_tokens'] == before['used_tokens']
    assert after['used_steps'] == before['used_steps']
    assert runtime.db.list_steps(rid) == steps
    assert len(runtime.db.list_tool_calls(rid)) == 1
    assert harness._load(rid)['turn'] == 1
    assert harness._load(rid)['work']['revision'] == 1


@pytest.mark.parametrize('position', ['before', 'after', 'corrupt'])
def test_saved_provider_response_recovers_snapshot_without_resending(tmp_path, monkeypatch, position):
    from paperpilot_agentic import episode_runner
    runtime, harness, rid = setup(tmp_path, monkeypatch)
    original = episode_runner.immutable

    def die(folder, name, data):
        if name.endswith('-response.json'):
            if position != 'before':
                original(folder, name, data)
            raise KeyboardInterrupt('Process death around response file write')
        return original(folder, name, data)

    monkeypatch.setattr(episode_runner, 'immutable', die)
    with pytest.raises(KeyboardInterrupt):
        harness.advance(rid, max_turns=1)
    before = runtime.db.get_run(rid)
    assert before['used_llm_calls'] == 1
    path = episode_folder(runtime, rid)/'turn-01-response.json'
    previous = path.read_bytes() if path.exists() else None
    if position == 'corrupt':
        path.write_text(json.dumps({'text': 'tampered response'}), encoding='utf-8')
    monkeypatch.setattr(episode_runner, 'immutable', original)
    PrototypeHarness(runtime).advance(rid, max_turns=1)
    assert runtime.db.get_run(rid)['used_llm_calls'] == before['used_llm_calls']
    assert runtime.db.get_run(rid)['used_tokens'] == before['used_tokens']
    if position == 'corrupt':
        assert harness._load(rid)['phase'] == 'FAILED'
        assert harness._load(rid)['turn'] == 0
        assert 'tampered response' in path.read_text(encoding='utf-8')
    else:
        assert harness._load(rid)['turn'] == 1
        assert path.exists()
        assert json.loads(path.read_text(encoding='utf-8'))['finish_reason'] == 'stop'
        if previous:
            assert path.read_bytes() == previous


@pytest.mark.parametrize('position', ['before_metadata', 'after_metadata'])
def test_prototype_terminal_publication_recovers_without_new_work(tmp_path, monkeypatch, position):
    runtime, harness, rid = setup(tmp_path, monkeypatch)
    state = harness._load(rid)
    harness._finish(rid, state, 'Maintained partial fixture; no candidate executed', complete=False)
    harness._save(rid, state)
    original = runtime.db.update_run_metadata

    def die(run_id, payload, *args, **kwargs):
        if 'final_answer' in payload:
            if position == 'after_metadata':
                original(run_id, payload, *args, **kwargs)
            raise KeyboardInterrupt('Process death during terminal publication')
        return original(run_id, payload, *args, **kwargs)

    monkeypatch.setattr(runtime.db, 'update_run_metadata', die)
    with pytest.raises(KeyboardInterrupt):
        harness.advance(rid)
    before = runtime.db.get_run(rid)
    monkeypatch.setattr(runtime.db, 'update_run_metadata', original)
    result = PrototypeHarness(runtime).advance(rid)
    assert result['run']['status'] == 'COMPLETED'
    assert harness._load(rid)['phase'] == 'PARTIAL'
    assert result['final_answer']['scientific_validity'] == 'NOT_VERIFIED'
    assert result['final_answer']['answer'] == state['stop_reason']
    assert result['run']['used_llm_calls'] == before['used_llm_calls'] == 0
    assert runtime.db.list_tool_calls(rid) == []
    PrototypeHarness(runtime).advance(rid)
    events = runtime.db.list_events(run_id=rid)
    assert len([e for e in events if e['event_type'] == 'research.publication_recovered']) == 1


def test_committed_verification_receipt_replays_without_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    launches = []

    def verify(*args):
        launches.append(True)
        return SimpleNamespace(status='COMPLETED', exit_code=0, timed_out=False,
            metrics={'passed': False, 'failure_kind': 'IMPLEMENTATION'}, backend='docker',
            to_dict=lambda: {'status': 'COMPLETED', 'stderr_path': 'retained-fixture.log'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', verify)
    original = harness._save
    monkeypatch.setattr(harness, '_save', lambda *a, **k: (_ for _ in ()).throw(
        KeyboardInterrupt('Process death after tool receipt commit')))
    with pytest.raises(KeyboardInterrupt):
        harness._training(rid, state)
    before = runtime.db.get_run(rid)
    assert runtime.db.get_tool_call(state['pending']['call'])['status'] == 'COMPLETED'
    assert harness._load(rid)['phase'] == 'WAITING_APPROVAL'
    monkeypatch.setattr(harness, '_save', original)
    recovered = harness._load(rid)
    assert harness._training(rid, recovered) is True
    assert len(launches) == 1
    assert recovered['last_observation']['failure_kind'] == 'IMPLEMENTATION'
    assert recovered['work']['tasks']['check']['status'] == 'DONE'
    after = runtime.db.get_run(rid)
    assert after['used_llm_calls'] == before['used_llm_calls']
    assert after['used_steps'] == before['used_steps']
