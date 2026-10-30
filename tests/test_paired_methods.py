"""Paired workflow contract fixtures; no GPU or scientific accuracy measured here."""
import base64
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperpilot_agentic import paired_methods as paired
from paperpilot_agentic.method_drafts import record_draft_review
from paperpilot_agentic.research_design import enqueue_design
from test_method_drafts import bound as bound
import test_method_drafts as draft_fixtures
from test_reference_methods import profile as profile


@pytest.fixture(autouse=True)
def synthetic_ablation(monkeypatch):
    prefix = 'class SNUNet_ECAM:\n    def forward(self, xA, xB):\n        out = xA + xB\n'
    suffix = '        out = self.conv_final(out)\n        return (out,)\n'
    monkeypatch.setattr(draft_fixtures, 'SOURCE', prefix + paired.ATTENTION_BLOCK + suffix)
    monkeypatch.setattr(draft_fixtures, 'REPLACEMENT',
        'def forward(self, xA, xB):\n    out = xA + xB\n    out = self.conv_final(out)\n    return (out,)\n')


def fake_paired_report(manifest):
    request = manifest['request']
    trials = []
    for seed in request['seeds']:
        row = {'seed': seed, 'pretraining_max_abs_logit_difference': .5, 'pool_adapter_max_abs_error': 0.}
        for variant in ('baseline', 'candidate'):
            candidate = variant == 'candidate'
            validation = ({'tn': 0, 'fp': 65536, 'fn': 0, 'tp': 65536, 'f1': 2/3, 'iou': .5}
                if candidate else {'tn': 65536, 'fp': 0, 'fn': 65536, 'tp': 0, 'f1': 0., 'iou': 0.})
            row[variant] = {'initial_state_sha256': f'{seed:064x}',
                'steps_completed': request['steps'], 'weight_updated': True,
                'train_order': manifest['protocol']['training_order'],
                'losses': [.8]*request['steps'], 'gradient_norms': [1.]*request['steps'],
                'missing_gradients': [paired.EXPECTED_UNUSED if candidate else []]*request['steps'],
                'validation': validation, 'predictions': [
                    {'sample_id': s['sample_id'], 'pixels': 65536,
                     'packed_bits': base64.b64encode(bytes([255 if candidate else 0])*8192).decode()}
                    for s in request['samples']['val']]}
        trials.append(row)
    return {'scope': 'PAIRED_SNUNET_DIAGNOSTIC_ONLY', 'source_hashes': request['source_hashes'],
        'protocol_sha256': request['protocol_sha256'], 'device': request['device'], 'trials': trials,
        'environment': {'deterministic_algorithms': True},
        'test_opened': False, 'scientific_hypothesis_supported': False}


class PairedRunner:
    def __init__(self):
        self.calls = 0
        self.change = lambda x: None
    def run(self, protocol):
        self.calls += 1
        assert protocol.command[1:] == ['-I', '-B', '-u', 'worker.py']
        assert protocol.policy.timeout_seconds == 300
        manifest = json.loads((Path(protocol.source_dir)/'manifest.json').read_bytes())
        report = fake_paired_report(manifest)
        self.change(report)
        return SimpleNamespace(status='COMPLETED', source_unchanged=True, metrics=report, error=None,
            network_isolated=False, isolation_notes=['synthetic fixture; no process started'],
            result_path=str(Path(protocol.output_dir)/'fake-result.json'), artifacts=[])


@pytest.fixture
def ready(bound):
    runtime, provider, parent, args = bound
    design = enqueue_design(runtime, parent, **args)['run']['run_id']
    result = runtime.resume(design)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    record_draft_review(runtime, design, reviewer='fixture-review', decision='READY_FOR_EXECUTION_REVIEW',
        findings=['Synthetic reviewed ablation; original performance claim not approved; diagnostics only'])
    binding = paired.prepare_comparison(runtime, design)
    runner = PairedRunner()
    spec = runtime.registry.get('method.compare')
    runtime.registry.register(replace(spec,
        handler=lambda ctx, args: paired.run_comparison(ctx, args, runner=runner)), replace=True)
    return runtime, design, binding, runner


def test_paired_runtime_requires_approval_and_preserves_all_seeds(ready):
    from paperpilot_agentic.research_notebook import ResearchNotebook
    runtime, design, binding, runner = ready
    rid = paired.enqueue_comparison(runtime, binding)['run']['run_id']
    waiting = runtime.resume(rid)
    assert waiting['run']['status'] == 'WAITING_APPROVAL' and runner.calls == 0
    runtime.approve(waiting['pending_approvals'][0]['approval_id'], actor='test')
    result = runtime.resume(rid)
    assert result['run']['status'] == 'COMPLETED', result['run'].get('error')
    assert result['run']['research_outcome'] == 'PARTIAL'
    report = result['run']['metadata']['method_comparison']
    assert report['status'] == 'PAIRED_DIAGNOSTIC_VERIFIED'
    assert report['scientific_acceptance'] == 'NOT_ASSESSED'
    assert [t['seed'] for t in report['trials']] == [7, 11, 19]
    assert all(t['delta'] == {'f1': 2/3, 'iou': .5} for t in report['trials'])
    assert 'packed_bits' not in json.dumps(report)
    assert runtime.resume(rid)['run']['status'] == 'COMPLETED' and runner.calls == 1
    assert runtime.db.get_run(rid)['used_llm_calls'] == 0
    assert runtime.enqueue_follow_up(parent_run_id=rid, question='Explain limits')['run']['metadata']['engine'] == 'answer_first_text'
    notebook = ResearchNotebook(runtime.db, design)
    notebook.import_experiment_run(rid)
    notes = notebook.notes(kind='experiment')
    assert len(notes) == 1 and notes[0]['is_scientific_evidence'] is False


@pytest.mark.parametrize('problem', ['initialization', 'order', 'gradients', 'nan', 'metrics', 'seeds', 'science', 'test', 'adapter'])
def test_paired_violations_cannot_be_reported_as_verified(ready, problem):
    runtime, _, binding, _ = ready
    root, manifest = paired.load_comparison(runtime, binding)
    report = fake_paired_report(manifest)
    candidate = report['trials'][0]['candidate']
    if problem == 'initialization':
        candidate['initial_state_sha256'] = 'f'*64
    elif problem == 'order':
        candidate['train_order'] = ['different-sample']*2
    elif problem == 'gradients':
        candidate['missing_gradients'][0] = []
    elif problem == 'nan':
        candidate['gradient_norms'][0] = float('nan')
    elif problem == 'metrics':
        candidate['validation']['f1'] = 1.
    elif problem == 'seeds':
        report['trials'].pop()
    elif problem == 'science':
        report['scientific_hypothesis_supported'] = True
    elif problem == 'test':
        report['test_opened'] = True
    elif problem == 'adapter':
        report['trials'][0]['pool_adapter_max_abs_error'] = .1
    with pytest.raises(ValueError):
        paired.verify_comparison(report, root, manifest)


def test_revised_review_and_modified_code_invalidate_pending_approval(ready):
    runtime, design, binding, runner = ready
    rid = paired.enqueue_comparison(runtime, binding)['run']['run_id']
    waiting = runtime.resume(rid)
    runtime.approve(waiting['pending_approvals'][0]['approval_id'], actor='test')
    record_draft_review(runtime, design, reviewer='fixture-review', decision='REVISION_REQUIRED',
                       findings=['New safety concern; revoke readiness'])
    assert runtime.resume(rid)['run']['status'] == 'FAILED' and runner.calls == 0
    with pytest.raises(ValueError, match='host review'):
        paired.prepare_comparison(runtime, design)


def test_unknown_attempt_is_not_repeated(ready):
    runtime, _, binding, runner = ready
    rid = paired.enqueue_comparison(runtime, binding)['run']['run_id']
    ctx = SimpleNamespace(workspace=runtime.workspace, db=runtime.db, run_id=rid, tool_call_id='interrupted')
    folder = runtime.workspace/'paired-method-runs'/'interrupted'
    folder.mkdir(parents=True)
    (folder/'started.json').write_text('{}')
    with pytest.raises(FileExistsError):
        paired.run_comparison(ctx, {'binding_id': binding}, runner=runner)
    assert runner.calls == 0


@pytest.mark.parametrize('seeds,steps', [([7, 7], 2), ([True], 2), ([7], 5), ([7], True), ([7, 11, 19, 23], 2)])
def test_training_budget_is_bounded_before_execution(ready, seeds, steps):
    runtime, design, _, runner = ready
    with pytest.raises(ValueError, match='1-3 distinct seeds'):
        paired.prepare_comparison(runtime, design, seeds=seeds, steps=steps)
    assert runner.calls == 0


def test_only_exact_reviewed_ast_is_admitted(ready):
    runtime, _, binding, _ = ready
    root, _ = paired.load_comparison(runtime, binding)
    source = (root/'baseline.py').read_text()
    candidate = (root/'candidate.py').read_text()
    paired.check_ablation(source, candidate)
    with pytest.raises(ValueError, match='Only the reviewed'):
        paired.check_ablation(source, candidate+'\nprint("UNREVIEWED_SIDE_EFFECT")\n')
    (root/'candidate.py').write_text(candidate+'\n# changed after approval\n')
    with pytest.raises(ValueError, match='source or samples changed'):
        paired.load_comparison(runtime, binding)


def test_web_cannot_change_frozen_protocol(ready):
    import time
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from paperpilot_agentic.web_api import create_fastapi_router
    runtime, design, binding, runner = ready
    app = FastAPI()
    app.include_router(create_fastapi_router(runtime_factory=lambda: runtime))
    with TestClient(app) as client:
        url = f'/agentic/runs/{design}/method-comparison'
        assert client.post(url, json={'binding_id': binding, 'steps': 10000}).status_code == 422
        result = client.post(url, json={'binding_id': binding})
        assert result.status_code == 202, result.text
        rid = result.json()['run']['run_id']
        for _ in range(100):
            state = runtime.status(rid)
            if state['run']['status'] in {'WAITING_APPROVAL', 'FAILED'}:
                break
            time.sleep(.02)
        assert state['run']['status'] == 'WAITING_APPROVAL', state['run'].get('error')
        assert runner.calls == 0
