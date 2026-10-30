"""Synthetic predictions exercise the same verifier as historical experiments."""
import base64
import json

import numpy as np
import pytest
from PIL import Image

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.providers import LLMResponse
from paperpilot_agentic.reference_methods import encoded, sha
from paperpilot_agentic.research_harness import ResearchHarness
from paperpilot_agentic.research_diagnosis import DiagnosisHarness
from paperpilot_agentic.research_diagnostics import freeze_experiment, observation, metrics
from paperpilot_agentic.research_training import episode_folder


def action(name, **arguments):
    return {'action': name, 'reason': 'Synthetic decision based on supplied observations', 'arguments': arguments}


def report(*, diagnostic=False):
    refs = [f'paired/{i}/delta/f1' for i in range(3)]
    if diagnostic:
        refs.append('coverage/0/candidate/fp')
    return action('report', fact_refs=refs,
        interpretations=[{'text': 'Synthetic observations do not prove activation saturation.', 'fact_refs': refs}],
        limitations=['Synthetic development data, no activation observations'],
        next_experiment={'hypothesis': 'Measure activation distributions before a new candidate',
            'support_refs': refs[:1], 'counter_refs': refs[1:],
            'missing_observations': ['Pre/post sigmoid distributions'],
            'falsification': 'No predicted saturation differences would reject the mechanism'})


class Provider:
    model = 'synthetic-model'
    def __init__(self, responses=None):
        self.responses, self.messages = list(responses or []), []
    def complete(self, messages, before_attempt=None, **kwargs):
        if before_attempt:
            before_attempt()
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(text=json.dumps(response), model=self.model, finish_reason='stop',
                           prompt_tokens=100, completion_tokens=50)


@pytest.fixture
def experiment(tmp_path):
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.sqlite3',
        llm=Provider(), text_reader_protocol='synthetic-protocol')
    ResearchHarness(runtime)
    parent = runtime.enqueue(question='Synthetic parent', metadata={'engine': 'research_loop_v1'})['run']['run_id']
    code = b'# Synthetic frozen source; never executed\n'
    source = tmp_path/'research-harness'/'candidates'
    manifest = {'version': 3, 'episode_id': parent, 'environment': {},
        'protocol': {'epochs': 1, 'train_images': 4},
        'request': {'protocol_sha256': 'synthetic-protocol', 'expected_unused': [], 'samples': {'val': []}},
        'files': {'baseline.py': sha(code), 'candidate.py': sha(code)}}
    truths = []
    for i, positive in enumerate((0, 1, 1000, 20000)):
        truth = np.zeros(65536, dtype=np.uint8)
        truth[:positive] = 255
        truths.append(truth)
        manifest['request']['samples']['val'].append({'sample_id': f's{i}', 'label': f'label{i}.png',
                                                       'hashes': {}})
    # Compute label hashes before calculating the content-addressed manifest.
    for i, sample in enumerate(manifest['request']['samples']['val']):
        from io import BytesIO
        image = BytesIO()
        Image.fromarray(truths[i].reshape(256, 256)).save(image, format='PNG')
        sample['hashes']['label'] = sha(image.getvalue())
    binding = sha(encoded(manifest))
    root = source/binding
    root.mkdir(parents=True)
    (root/'manifest.json').write_bytes(encoded(manifest))
    for name in manifest['files']:
        (root/name).write_bytes(code)
    for i in range(4):
        Image.fromarray(truths[i].reshape(256, 256)).save(root/f'label{i}.png')
    trials = []
    for seed in (7, 11, 19):
        pair = {'seed': seed}
        for variant in ('baseline', 'candidate'):
            predictions, total = [], np.zeros(4, dtype=np.int64)
            for i, truth in enumerate(truths):
                pred = (truth == 255).astype(np.uint8)
                if variant == 'candidate':
                    pred[:min(20, int((truth == 255).sum()))] = 0
                    pred[-5:] = 1
                total += np.bincount((truth == 255).astype(np.uint8)*2+pred, minlength=4)
                predictions.append({'sample_id': f's{i}', 'pixels': 65536,
                    'packed_bits': base64.b64encode(np.packbits(pred, bitorder='big')).decode()})
            checked = metrics([int(v) for v in total])
            raw = {'binding_id': binding, 'identity': {'binding_id': binding, 'protocol_sha256': 'synthetic-protocol',
                'stage': 'formal', 'seed': seed, 'variant': variant}, 'status': 'COMPLETED',
                'step': 1, 'target': 1, 'environment': {}, 'test_opened': False,
                'pool_adapter_max_abs_error': 0, 'expected_unused': [], 'scientific_hypothesis_supported': False,
                'losses': [1.], 'gradient_norms': [1.], 'initial_state_sha256': str(seed),
                'final_state_sha256': 'updated', 'predictions': predictions, 'validation': checked}
            path = episode_folder(runtime, parent)/binding/'trials'/f'{seed}-{variant}'/'receipt-synthetic.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded(raw))
            pair[variant] = {'status': 'VERIFIED', 'metrics': checked,
                'initial_state_sha256': str(seed), 'receipt': {'path': str(path), 'sha256': sha(path.read_bytes())}}
        pair['delta'] = {k: pair['candidate']['metrics'][k]-pair['baseline']['metrics'][k]
                         for k in ('precision', 'recall', 'f1', 'iou')}
        trials.append(pair)
    result = {'status': 'TRAINING_VERIFIED', 'trials': trials}
    (episode_folder(runtime, parent)/('result-'+binding+'.json')).write_bytes(encoded(result))
    state = {'candidates': [{'card': {'binding_id': binding}, 'result': result}]}
    with runtime.db.transaction() as conn:
        conn.execute('INSERT INTO research_episodes VALUES(?,?,?,?)',
                     (parent, 'synthetic-parent', 'parent', encoded(state).decode()))
    return runtime, parent, binding


def test_recomputed_groups_reconcile_and_top_errors_are_not_cherry_picked(experiment):
    runtime, parent, candidate = experiment
    binding = freeze_experiment(runtime, parent, candidate)
    paired = observation(runtime, binding, 'paired')
    grouped = observation(runtime, binding, 'coverage')
    samples = observation(runtime, binding, 'samples')
    for seed in (7, 11, 19):
        parts = [r for r in grouped['rows'] if r['seed'] == seed]
        assert len(parts) == 4 and sum(r['sample_count'] for r in parts) == 4
        whole = next(r for r in paired['rows'] if r['seed'] == seed)
        for variant in ('baseline', 'candidate'):
            for key in ('tn', 'fp', 'fn', 'tp'):
                assert sum(r[variant][key] for r in parts) == whole[variant][key]
        errors = next(r for r in samples['rows'] if r['seed'] == seed and r['error'] == 'fp')
        assert [r['sample_id'] for r in errors['rows']] == ['s0', 's1', 's2', 's3']
    assert paired['new_gpu_starts'] == 0


def test_modified_prediction_label_or_bundle_rejected(experiment):
    runtime, parent, candidate = experiment
    binding = freeze_experiment(runtime, parent, candidate)
    bundle = runtime.workspace/'research-harness'/'diagnostic-inputs'/binding['identity']/'bundle.json'
    original = bundle.read_bytes()
    bundle.write_bytes(original+b' ')
    with pytest.raises(ValueError, match='bundle identity'):
        observation(runtime, binding, 'coverage')
    bundle.write_bytes(original)
    data = json.loads(original)
    receipt = next(Path for Path in data['assets'] if 'receipt-' in Path)
    from pathlib import Path
    path = Path(receipt)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='input changed'):
        observation(runtime, binding, 'samples')


def test_public_harness_is_read_only_idempotent_and_role_isolated(experiment):
    runtime, parent, candidate = experiment
    provider = runtime.llm_runtime.model
    provider.responses = [action('request_experiment'), action('delegate_analysis'),
        action('inspect_experiment', view='coverage'), report(diagnostic=True), action('finish', summary='Synthetic diagnostic complete')]
    h = ResearchHarness(runtime)
    before = runtime.db.get_run(parent)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-diagnosis')['run']['run_id']
    output = runtime.resume(rid)
    assert output['run']['status'] == 'COMPLETED'
    assert h.view(rid)['completed'] is True
    assert h.view(rid)['observed_views'] == ['paired', 'coverage']
    assert h.view(rid)['new_gpu_starts'] == 0
    assert runtime.db.get_run(parent) == before
    assert h.start_diagnosis(parent, candidate, 'synthetic-diagnosis')['run']['run_id'] == rid
    h.advance(rid)
    assert len(provider.messages) == 5
    assert not any(m['role'] == 'assistant' for m in provider.messages[2])
    assert output['pending_approvals'] == []
    assert not list((runtime.workspace/'research-harness'/'episodes').glob('*/gpu-budget.sqlite3'))


def test_unknown_call_stops_and_never_replays(experiment):
    runtime, parent, candidate = experiment
    provider = runtime.llm_runtime.model
    provider.responses = [TimeoutError('Synthetic unknown response')]
    h = ResearchHarness(runtime)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-unknown')['run']['run_id']
    result = h.advance(rid)
    assert h.view(rid)['phase'] == 'UNKNOWN'
    assert result['llm_budget']['unknown_results']
    h.advance(rid)
    assert len(provider.messages) == 1


def test_summary_mode_cannot_read_diagnostics_or_change_inputs(experiment):
    runtime, parent, candidate = experiment
    h = DiagnosisHarness(runtime)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-summary', comparison_mode='summary')['run']['run_id']
    with pytest.raises(ValueError, match='different inputs'):
        h.start_diagnosis(parent, candidate, 'synthetic-summary', comparison_mode='tools')
    state = h._load(rid)
    with pytest.raises(ValueError, match='only the paired'):
        h._apply(rid, state, action('inspect_experiment', view='samples'), 1)
    with pytest.raises(ValueError, match='lacks this action'):
        h._apply(rid, state, action('request_experiment'), 1)
    forged = report()
    forged['arguments']['fact_refs'].append('coverage/not-observed')
    state['role'] = 'analyst'
    with pytest.raises(ValueError, match='unknown observed fact IDs'):
        h._apply(rid, state, forged, 1)


def test_workbench_diagnosis_is_owned_and_duplicate_does_not_submit_twice(experiment):
    from types import SimpleNamespace
    from paperpilot_agentic.workbench import ResearchWorkbench
    runtime, parent, candidate = experiment
    submitted = []
    background = SimpleNamespace(active=lambda _: False, submit=lambda *args: submitted.append(args))
    workbench = ResearchWorkbench(runtime, background)
    cid = runtime.db.get_run(parent)['metadata']['conversation_id']
    request = {'action': 'research_diagnosis', 'request_id': 'synthetic-web-diagnosis',
               'parent_run_id': parent, 'conversation_id': cid, 'candidate_id': candidate}
    result = workbench.submit(request)
    repeated = workbench.submit(request)
    assert result['run_id'] == repeated['run_id'] and repeated['reused']
    assert len(submitted) == 1 and not runtime.llm_runtime.model.messages
    assert workbench.turn(result['run_id'])['research_episode']['new_gpu_starts'] == 0
    with pytest.raises(ValueError, match='another conversation'):
        workbench.submit(dict(request, request_id='synthetic-foreign-diagnosis', conversation_id='foreign'))
    runtime.cancel(result['run_id'])
    assert ResearchHarness(runtime).view(result['run_id'])['phase'] == 'CANCELLED'


def test_process_death_after_model_return_recovers_the_saved_response(experiment):
    runtime, parent, candidate = experiment
    provider = runtime.llm_runtime.model
    provider.responses = [action('delegate_analysis'), action('inspect_experiment', view='coverage'),
                          report(diagnostic=True), action('finish', summary='Recovered diagnostic')]
    h = DiagnosisHarness(runtime)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-return-recovery')['run']['run_id']
    def crash(*args):
        raise SystemExit('Synthetic host process death after persisted model response')
    h._apply = crash
    with pytest.raises(SystemExit):
        h.advance(rid)
    assert len(provider.messages) == 1
    result = runtime.resume(rid)
    assert ResearchHarness(runtime).view(rid)['completed']
    assert result['run']['used_llm_calls'] == 4 and len(provider.messages) == 4
    assert result['llm_budget']['unknown_results'] == 0


def test_diagnostic_prompt_keeps_all_observations_once_and_preserves_raw_receipts(experiment):
    runtime, parent, candidate = experiment
    h = DiagnosisHarness(runtime)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-prompt-packet')['run']['run_id']
    state = h._load(rid)
    result = h._apply(rid, state, action('inspect_experiment', view='samples'), 1)
    state['last_observation'] = result
    state['contexts']['lead'] = [{'role': 'user', 'content': json.dumps({'tool_result': result})}]
    before = json.dumps(state, sort_keys=True)
    _, messages = h._context(rid, state)
    packet = json.loads(messages[1]['content'])
    assert packet['schema']['delegate_analysis']['arguments'] == {}
    assert packet['available_view_contracts']['coverage']['grouping_basis'].startswith('Fraction of ground-truth')
    assert 'prediction confidence' in packet['available_view_contracts']['coverage']['not_measured']
    for view, observed in state['observations'].items():
        assert packet['observations'][view]['rows'] == observed['rows']
        for reference, expected in observed['facts'].items():
            value = packet['observations'][view]['rows']
            for key in reference.split('/')[1:]:
                value = value[int(key)] if isinstance(value, list) else value[key]
            assert value == expected
    historical = json.loads(messages[2]['content'])['tool_result']['observation']
    assert set(historical) == {'observation_id', 'view', 'source_identity'}
    assert packet['last_observation']['observation'] == historical
    assert json.dumps(state, sort_keys=True) == before
    with pytest.raises(ValueError, match='Delegate independent analysis'):
        h._apply(rid, state, action('finish', summary='Premature finish'), 2)


def test_lead_can_request_more_evidence_without_deleting_the_prior_analysis(experiment):
    runtime, parent, candidate = experiment
    h = DiagnosisHarness(runtime)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-analysis-feedback')['run']['run_id']
    state = h._load(rid)
    h._apply(rid, state, action('delegate_analysis'), 1)
    h._apply(rid, state, action('inspect_experiment', view='coverage'), 2)
    h._apply(rid, state, report(diagnostic=True), 3)
    first = state['analysis']
    h._apply(rid, state, action('delegate_analysis', feedback='Distinguish label coverage from confidence'), 4)
    assert state['role'] == 'analyst'
    _, messages = h._context(rid, state)
    packet = json.loads(messages[1]['content'])
    assert packet['analysis_feedback'] == 'Distinguish label coverage from confidence'
    assert packet['independent_analysis'] == first
    h._apply(rid, state, report(diagnostic=True), 5)
    assert state['analysis_history'] == [first]
    assert state['role'] == 'lead' and not runtime.llm_runtime.model.messages


def test_missing_rationale_reports_the_missing_field_instead_of_replaying_a_generic_error(experiment):
    runtime, parent, candidate = experiment
    h = DiagnosisHarness(runtime)
    rid = h.start_diagnosis(parent, candidate, 'synthetic-missing-reason')['run']['run_id']
    state = h._load(rid)
    before = json.dumps(state, sort_keys=True)
    with pytest.raises(ValueError, match="missing fields.*reason"):
        h._apply(rid, state, {'action': 'finish', 'arguments': {'summary': 'Unsupported early finish'}}, 1)
    assert json.dumps(state, sort_keys=True) == before
    assert not runtime.llm_runtime.model.messages
