from types import SimpleNamespace

import pytest

from paperpilot_agentic.prototype_environment import preflight_environment


def image(letter='a'):
    return {'available': True, 'image_id': 'sha256:'+letter*64, 'platform': 'linux/amd64',
            'image_reference': 'paperpilot-prototype:cpu'}


def test_preflight_resolves_immutable_image_without_starting_container(monkeypatch):
    calls = []

    def inspect(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=image()['image_id']+'|linux|amd64\n')

    monkeypatch.setattr('paperpilot_agentic.prototype_environment.subprocess.run', inspect)
    assert preflight_environment() == image()
    assert calls[0][:3] == ['docker', 'image', 'inspect']
    assert 'run' not in calls[0] and 'pull' not in calls[0]


@pytest.mark.parametrize('stdout', ['mutable-tag|linux|amd64', 'sha256:'+'a'*64+'|windows|amd64'])
def test_preflight_rejects_unpinned_or_wrong_platform(monkeypatch, stdout):
    monkeypatch.setattr('paperpilot_agentic.prototype_environment.subprocess.run',
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout))
    assert preflight_environment()['available'] is False


def test_unavailable_docker_is_a_pending_environment_result(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('docker')

    monkeypatch.setattr('paperpilot_agentic.prototype_environment.subprocess.run', missing)
    assert preflight_environment()['available'] is False


def test_environment_wait_resumes_without_new_llm_calls_or_execution(tmp_path, monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from test_prototype_workspace import Reader

    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.preflight_environment',
                        lambda: {'available': False, 'reason': 'DOCKER_UNAVAILABLE'})
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run',
                        lambda *a: pytest.fail('Pending environment/approval cannot execute'))
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'environment-fixture', 'parent', ['source'])['run']['run_id']
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'WAITING_ENVIRONMENT'
    from paperpilot_agentic.workbench import ResearchWorkbench
    from test_workbench import Background
    view = ResearchWorkbench(runtime, Background()).turn(rid)
    assert view['can_resume'] and '执行环境尚未就绪' in view['state_text']
    calls = runtime.db.get_run(rid)['used_llm_calls']
    assert runtime.db.list_approvals(rid) == []
    harness.advance(rid)
    assert runtime.db.get_run(rid)['used_llm_calls'] == calls
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.preflight_environment', image)
    harness.advance(rid)
    state = harness._load(rid)
    assert state['phase'] == 'WAITING_APPROVAL'
    assert state['pending']['binding']['execution_environment']['image_id'] == image()['image_id']
    assert runtime.db.get_run(rid)['used_llm_calls'] == calls
    assert len(runtime.db.list_approvals(rid)) == 1


def test_approved_execution_uses_frozen_image_not_mutable_tag(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    frozen_image = state['pending']['binding']['execution_environment']['image_id']
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.preflight_environment', lambda: image('b'))
    launched = []

    def execute(self, protocol):
        launched.append(protocol.policy.docker_image)
        assert protocol.policy.to_dict() == state['pending']['binding']['execution_environment']['policy']
        return SimpleNamespace(status='COMPLETED', exit_code=0, timed_out=False, metrics={'passed': True},
                               to_dict=lambda: {'status': 'COMPLETED'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', execute)
    assert harness._training(rid, state)
    assert launched == [frozen_image]


def test_environment_binding_tamper_is_rejected_before_execution(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    runtime.approve(state['pending']['approval'])
    state['execution_environment'] = {**state['execution_environment'], 'image_id': image('b')['image_id']}
    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run',
                        lambda *a: pytest.fail('Changed execution environment must not launch'))
    with pytest.raises(ValueError, match='environment binding'):
        harness._training(rid, state)


def test_existing_legacy_approval_is_not_silently_upgraded(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    from paperpilot_agentic.prototype_environment import freeze_environment
    with monkeypatch.context() as legacy:
        legacy.setattr('paperpilot_agentic.prototype_harness.freeze_environment',
                       lambda report, **kwargs: freeze_environment(report, filesystem='legacy-writable-v1'))
        runtime, harness, rid = waiting(tmp_path, legacy)
    state = harness._load(rid)
    assert 'filesystem' not in state['execution_environment']
    approved_binding = state['pending']['binding'].copy()
    runtime.approve(state['pending']['approval'])
    executions = []

    def execute(self, protocol):
        executions.append(protocol)
        assert protocol.policy.read_only_files == ()
        assert 'read_only_files' not in protocol.policy.to_dict()
        return SimpleNamespace(status='COMPLETED', exit_code=0, timed_out=False, metrics={'passed': True},
                               to_dict=lambda: {'status': 'COMPLETED'})

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.SandboxExperimentRunner.run', execute)
    assert harness._training(rid, state)
    assert len(executions) == 1 and state['pending']['binding'] == approved_binding


def test_preflight_http_does_not_start_model_or_execution(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.web_api import create_fastapi_router
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db')
    monkeypatch.setattr('paperpilot_agentic.prototype_environment.preflight_environment', image)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        result = client.get('/agentic/prototype-environment')
        assert result.status_code == 200 and result.json() == image()
    assert runtime.list_runs() == []


def test_cancel_during_preflight_never_creates_execution_approval(tmp_path, monkeypatch):
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    from paperpilot_agentic.prototype_harness import PrototypeHarness
    from test_prototype_workspace import Reader
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda _: [{'evidence_id': 'source', 'content': 'source'}])
    harness = PrototypeHarness(runtime)
    rid = harness.start('build', 'cancel-preflight-fixture', 'parent', ['source'])['run']['run_id']

    def cancelling_probe():
        runtime.cancel(rid)
        return image()

    monkeypatch.setattr('paperpilot_agentic.prototype_harness.preflight_environment', cancelling_probe)
    harness.advance(rid)
    assert harness._load(rid)['phase'] == 'CANCELLED'
    assert runtime.db.list_approvals(rid) == []
    assert not any(c['tool_name'] == 'prototype.validate' for c in runtime.db.list_tool_calls(rid))
