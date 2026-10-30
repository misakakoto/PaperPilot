"""Runtime-contract fixtures only. Real upstream/GPU checks are separately recorded."""
import base64
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from paperpilot_agentic import reference_methods as methods
from paperpilot_agentic.orchestrator import ResearchAgentRuntime


@pytest.fixture
def profile(tmp_path, monkeypatch):
    dataset, workspace = tmp_path/'dataset', tmp_path/'workspace'
    dataset.mkdir()
    for split in ('train', 'val'):
        lines = []
        for index in range(3):
            name = f'{split}_{index}.png'
            lines.append(' '.join(f'/old/location/{split}/{role}/{name}' for role in ('time1', 'time2', 'label')))
            for role in ('time1', 'time2', 'label'):
                folder = dataset/split/role
                folder.mkdir(parents=True, exist_ok=True)
                if role == 'label':
                    image = Image.frombytes('L', (256, 256), bytes([0, 255]) * (256*256//2))
                else:
                    image = Image.new('RGB', (256, 256), (index, 12, 25))
                image.save(folder/name)
        (dataset/f'{split}.txt').write_text('\n'.join(lines))
    # A poisoned test split must never be accessed.
    (dataset/'test.txt').write_text('INVALID DO NOT READ')
    source, license_text = 'raise RuntimeError("fixture must not execute")\n', 'synthetic-license'
    monkeypatch.setattr(methods, 'SOURCE_SHA', methods.sha(source.encode()))
    monkeypatch.setattr(methods, 'LICENSE_SHA', methods.sha(license_text.encode()))
    snapshot = tmp_path/'snapshot.json'
    snapshot.write_text(json.dumps({'code_url': methods.PIN+'models/Models.py', 'code': source,
        'license_url': methods.PIN+'LICENSE', 'license_text': license_text}))
    binding = methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable)
    return workspace, binding, snapshot, dataset


def fake_report(root, manifest):
    request = manifest['request']
    return {'scope': 'UPSTREAM_ADAPTER_AND_SYSU_SMOKE_ONLY', 'source_sha256': methods.SOURCE_SHA,
        'seed': request['seed'], 'steps_completed': request['steps'], 'device': request['device'],
        'parity_max_abs_errors': [0.0, 0.0], 'changed_input_max_abs_difference': 0.1,
        'weight_updated': True, 'losses': [0.7] * request['steps'], 'gradient_norms': [1.0] * request['steps'],
        'test_opened': False, 'paper_accuracy_reproduced': False,
        'validation': {'tn': 65536, 'fp': 0, 'fn': 65536, 'tp': 0, 'f1': 0., 'iou': 0.},
        'predictions': [{'sample_id': row['sample_id'], 'pixels': 65536,
            'packed_bits': base64.b64encode(bytes(8192)).decode()} for row in request['samples']['val']]}


class FakeRunner:
    def __init__(self):
        self.calls = 0
        self.change = lambda report: None
    def run(self, protocol):
        self.calls += 1
        root = Path(protocol.source_dir)
        manifest = json.loads((root/'manifest.json').read_text())
        report = fake_report(root, manifest)
        self.change(report)
        return SimpleNamespace(status='COMPLETED', source_unchanged=True, metrics=report,
            error=None, network_isolated=False, isolation_notes=['test fixture; no execution'],
            result_path=str(Path(protocol.output_dir)/'fixture-result.json'), artifacts=[])


def runtime_with_runner(workspace, runner):
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=workspace/'runtime.sqlite3')
    spec = runtime.registry.get('method.validate')
    runtime.registry.register(replace(spec, handler=lambda ctx, args: methods.run_profile(ctx, args, runner=runner)), replace=True)
    return runtime


def test_profile_is_pinned_immutable_and_train_val_only(profile):
    workspace, binding, snapshot, dataset = profile
    root, manifest = methods.load_profile(workspace, binding)
    assert set(manifest['splits']) == {'train', 'val'}
    assert all(len(rows) == 2 for rows in manifest['request']['samples'].values())
    again = methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable)
    assert binding == again
    other = methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable, steps=3)
    assert other != binding
    (root/'upstream.py').write_text('changed')
    with pytest.raises(ValueError, match='assets changed'):
        methods.load_profile(workspace, binding)
    with pytest.raises(ValueError, match='overwrite'):
        methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable)


@pytest.mark.parametrize('field', ['code', 'license_text', 'code_url'])
def test_uninspected_snapshot_cannot_become_executable(profile, field):
    workspace, _, snapshot, dataset = profile
    content = json.loads(snapshot.read_text())
    content[field] += 'tampered'
    snapshot.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='Only the inspected'):
        methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable)


@pytest.mark.parametrize('steps', [0, 5, True, '2'])
def test_step_limit_cannot_be_bypassed(profile, steps):
    workspace, _, snapshot, dataset = profile
    with pytest.raises(ValueError, match='seed or steps'):
        methods.prepare_snunet(workspace, snapshot=snapshot, dataset=dataset, python=sys.executable, steps=steps)


def test_independent_metrics_detect_misreported_counts_and_wrong_predictions(profile):
    workspace, binding, _, _ = profile
    root, manifest = methods.load_profile(workspace, binding)
    report = fake_report(root, manifest)
    assert methods.verify_predictions(report, root, manifest['request'])['fn'] == 65536
    report['validation']['f1'] = .99
    with pytest.raises(ValueError, match='metric verification'):
        methods.verify_predictions(report, root, manifest['request'])
    report = fake_report(root, manifest)
    report['predictions'][0]['sample_id'] = 'wrong-label-pair'
    with pytest.raises(ValueError, match='identity'):
        methods.verify_predictions(report, root, manifest['request'])


def test_validation_requires_approval_replays_and_never_claims_scientific_success(profile):
    from paperpilot_agentic.models import RunStatus
    from paperpilot_agentic.research_notebook import ResearchNotebook
    workspace, binding, _, _ = profile
    runner = FakeRunner()
    runtime = runtime_with_runner(workspace, runner)
    parent = runtime.enqueue(question='Research notebook root')['run']['run_id']
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    run_id = methods.enqueue_validation(runtime, binding, parent_run_id=parent)['run']['run_id']
    waiting = runtime.resume(run_id)
    assert waiting['run']['status'] == 'WAITING_APPROVAL' and runner.calls == 0
    approval = waiting['pending_approvals'][0]
    runtime.approve(approval['approval_id'], actor='test')
    observed_states = []
    runner.change = lambda report: observed_states.append(runtime.db.get_run(run_id)['status'])
    completed = runtime.resume(run_id)
    assert completed['run']['status'] == 'COMPLETED', completed['run'].get('error')
    assert completed['run']['research_outcome'] == 'PARTIAL'
    receipt = completed['run']['metadata']['method_validation']
    assert receipt['status'] == 'ADAPTER_SMOKE_VERIFIED'
    assert receipt['scientific_hypothesis_supported'] is False and receipt['paper_accuracy_reproduced'] is False
    assert 'predictions' not in receipt
    assert len(completed['artifacts']) == 1 and runner.calls == 1
    assert observed_states == ['RUNNING']
    assert runtime.resume(run_id)['run']['status'] == 'COMPLETED' and runner.calls == 1
    follow = runtime.enqueue_follow_up(parent_run_id=run_id, question='Explain the diagnostic')['run']
    assert follow['metadata']['engine'] == 'answer_first_text'
    notebook = ResearchNotebook(runtime.db, parent)
    notebook.import_experiment_run(run_id)
    notes = notebook.notes(kind='experiment')
    assert len(notes) == 1 and notes[0]['is_scientific_evidence'] is False
    assert notes[0]['record']['result']['output']['paper_accuracy_reproduced'] is False


def test_rejected_or_changed_approval_does_not_run(profile):
    workspace, binding, _, _ = profile
    runner = FakeRunner()
    runtime = runtime_with_runner(workspace, runner)
    run_id = methods.enqueue_validation(runtime, binding)['run']['run_id']
    waiting = runtime.resume(run_id)
    runtime.reject(waiting['pending_approvals'][0]['approval_id'], actor='test')
    assert runtime.resume(run_id)['run']['status'] == 'FAILED'
    assert runner.calls == 0
    rid = methods.enqueue_validation(runtime, binding)['run']['run_id']
    waiting = runtime.resume(rid)
    runtime.approve(waiting['pending_approvals'][0]['approval_id'], actor='test')
    (workspace/'method-profiles'/binding/'worker.py').write_text('changed after approval')
    assert runtime.resume(rid)['run']['status'] == 'FAILED'
    assert runner.calls == 0


@pytest.mark.parametrize('problem', ['metrics', 'steps', 'gradient', 'paper_claim', 'test_access'])
def test_bad_worker_receipt_cannot_be_published_as_verified(profile, problem):
    workspace, binding, _, _ = profile
    runner = FakeRunner()
    def change(report):
        if problem == 'metrics':
            report['validation']['f1'] = .9
        if problem == 'steps':
            report['steps_completed'] = 200
        if problem == 'gradient':
            report['gradient_norms'][0] = float('nan')
        if problem == 'paper_claim':
            report['paper_accuracy_reproduced'] = True
        if problem == 'test_access':
            report['test_opened'] = True
    runner.change = change
    ctx = SimpleNamespace(workspace=workspace, tool_call_id='fixture-call')
    result = methods.run_profile(ctx, {'binding_id': binding}, runner=runner)
    assert result.ok is False
    assert result.output['status'] == 'VALIDATION_FAILED'


def test_no_unknown_retry_or_model_selected_command(profile):
    workspace, binding, _, _ = profile
    runner = FakeRunner()
    ctx = SimpleNamespace(workspace=workspace, tool_call_id='fixture-call')
    attempt = workspace/'method-runs'/'fixture-call'
    attempt.mkdir(parents=True)
    (attempt/'started.json').write_text('{}')
    with pytest.raises(FileExistsError):
        methods.run_profile(ctx, {'binding_id': binding}, runner=runner)
    assert runner.calls == 0
    runtime = runtime_with_runner(workspace, runner)
    run_id = methods.enqueue_validation(runtime, binding)['run']['run_id']
    step = runtime.db.list_steps(run_id)[0]['step_id']
    with pytest.raises(ValueError, match='Unexpected tool arguments'):
        runtime.tools.invoke(run_id=run_id, step_id=step, tool_name='method.validate',
            arguments={'binding_id': binding, 'command': ['anything']})
    for invalid in ('../escape', 'fake', ''):
        with pytest.raises(ValueError, match='binding ID'):
            methods.load_profile(workspace, invalid)


def test_method_validation_endpoint_only_accepts_host_binding(profile):
    import time
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.models import RunStatus
    from paperpilot_agentic.web_api import create_fastapi_router
    workspace, binding, _, _ = profile
    runner = FakeRunner()
    runtime = runtime_with_runner(workspace, runner)
    parent = runtime.enqueue(question='Read and inspect the baseline')['run']['run_id']
    runtime.db.update_run_status(parent, RunStatus.COMPLETED)
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        response = client.post(f'/agentic/runs/{parent}/method-validation',
            json={'binding_id': binding, 'command': ['python', 'generated.py']})
        assert response.status_code == 422
        response = client.post(f'/agentic/runs/{parent}/method-validation', json={'binding_id': binding})
        assert response.status_code == 202, response.text
        run_id = response.json()['run']['run_id']
        for _ in range(100):
            state = runtime.status(run_id)
            if state['run']['status'] in {'WAITING_APPROVAL', 'FAILED'}:
                break
            time.sleep(.02)
        assert state['run']['status'] == 'WAITING_APPROVAL', state['run'].get('error')
        assert runner.calls == 0
        assert state['run']['metadata']['parent_run_id'] == parent
