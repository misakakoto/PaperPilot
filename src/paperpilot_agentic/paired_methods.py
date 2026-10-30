"""One host-reviewed SNUNet ablation, paired diagnostics and independent metrics.

Not a general generated-Python executor: only the exact AST transformation removing
both attention branches is supported. Model-proposed other changes stay drafts.
"""
import ast
import json
import math
from pathlib import Path
import re

from . import reference_methods as reference
from .method_drafts import build_patch, load_draft
from .models import Budget, ExperimentProtocol, RiskLevel, SandboxPolicy, ToolResult
from .sandbox import SandboxExperimentRunner
from .tooling import ToolSpec

ATTENTION_BLOCK = ('        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)\n'
                   '        ca1 = self.ca1(intra)\n'
                   '        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))\n')
EXPECTED_UNUSED = ['ca.fc1.weight', 'ca.fc2.weight', 'ca1.fc1.weight', 'ca1.fc2.weight']
sha, encoded = reference.sha, reference.encoded


def check_ablation(source, candidate):
    if sha(source.encode()) != reference.SOURCE_SHA or source.count(ATTENTION_BLOCK) != 1:
        raise ValueError('Expected the reviewed pinned SNUNet attention implementation')
    expected = source.replace(ATTENTION_BLOCK, '')
    if ast.dump(ast.parse(candidate)) != ast.dump(ast.parse(expected)):
        raise ValueError('Only the reviewed attention-removal AST is executable; keep other ideas as drafts')


def _review(runtime, design_run_id, material):
    review = runtime.db.get_run(design_run_id)['metadata'].get('research_design_review', {})
    artifact = review.get('artifact', {})
    if review.get('decision') != 'READY_FOR_EXECUTION_REVIEW' or not artifact.get('path'):
        raise ValueError('A host review must explicitly allow a separate execution review')
    body = {k: v for k, v in review.items() if k != 'artifact'}
    digest = sha(encoded(body))
    path = Path(artifact['path'])
    expected = runtime.workspace/'research-drafts'/design_run_id/'reviews'/(digest+'.json')
    if (path.is_symlink() or path.resolve() != expected.resolve()
            or not path.resolve().is_relative_to(runtime.workspace.resolve())
            or sha(path.read_bytes()) != artifact.get('sha256') or json.loads(path.read_bytes()) != body
            or body.get('draft_artifacts') != material['draft_artifacts']
            or body.get('candidate_sha256') != material['contract']['candidate_sha256']
            or body.get('binding_id') != material['baseline']['binding_id']
            or body.get('execution_authorized') is not False):
        raise ValueError('Host review changed or is not bound to this draft')
    return digest


def comparison_protocol(profile, seeds, steps):
    """The one supported diagnostic protocol; model text cannot redefine its controls."""
    if (not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 3
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds) or type(steps) is not int or not 1 <= steps <= 4):
        raise ValueError('Choose 1-3 distinct seeds and 1-4 steps per variant')
    return {'version': 1, 'recipe': 'snunet_remove_both_attention_paths_v1',
        'objective': 'Validate paired forward/backward execution and report descriptive metrics only',
        'supersedes_design_acceptance_rule': True,
        'not_tested': 'The original no-significant-decrease/non-inferiority hypothesis is NOT evaluated',
        'seeds': list(seeds), 'steps_per_variant': steps, 'variants': ['baseline', 'candidate'],
        'initialization': 'Copy all baseline state_dict parameters and buffers into candidate for each seed',
        'sample_policy': 'Same previously frozen two train and two val images for every seed; never test',
        'training_order': [profile['request']['samples']['train'][i % 2]['sample_id'] for i in range(steps)],
        'optimizer': {'name': 'SGD', 'lr': 0.001, 'momentum': 0, 'weight_decay': 0},
        'loss': 'two-class unweighted cross_entropy', 'batch_size': 1,
        'loss_implementation': 'Flatten NHW into classification samples; same pixel mean, deterministic 2-D CE kernel',
        'global_pool_implementation': 'ca/ca1 1x1 max pool uses flattened first-max reduction; forward parity required',
        'preprocessing': 'RGB / 127.5 - 1, label 255 is change; no resizing/augmentation',
        'checkpoint_selection': 'none; evaluate at fixed final step',
        'metrics': ['changed-class pixelwise F1', 'changed-class pixelwise IoU'],
        'aggregation': 'per-seed confusion pooled over frozen val images; candidate-minus-baseline paired deltas',
        'scientific_acceptance': 'NOT_ASSESSED; no p-values, non-inferiority or performance claims',
        'engineering_acceptance': ['equal initial states', 'finite logits/losses/gradients',
            'only ca/ca1 gradients absent in candidate', 'classifier weights update', 'zero upstream pool-adapter logit difference',
            'independent prediction-to-confusion verification for both variants in every seed'],
        'timeout_seconds': 300, 'max_total_optimizer_steps': len(seeds)*2*steps,
        'execution_mode': 'reviewed host-local code; NOT hard filesystem/network isolation'}


def prepare_comparison(runtime, design_run_id, *, seeds=(7, 11, 19), steps=2):
    """Host-only: freeze reviewed sources, paired protocol and copied original sample set."""
    material = load_draft(runtime, design_run_id)
    review_sha = _review(runtime, design_run_id, material)
    baseline, contract = material['baseline'], material['contract']
    built = build_patch({'target': contract['target'], 'replacement': contract['replacement'],
                         'review_notes': contract['review_notes']}, baseline)
    source, candidate = baseline['source'], built['candidate_source']
    if (sha(candidate.encode()) != contract['candidate_sha256']
            or built['contract']['patch_sha256'] != contract['patch_sha256']):
        raise ValueError('Candidate reconstruction changed')
    check_ablation(source, candidate)
    original, profile = reference.load_profile(runtime.workspace, baseline['binding_id'])
    protocol = comparison_protocol(profile, seeds, steps)
    files = {name: (original/name).read_bytes() for name in profile['files'] if name.startswith('inputs/')}
    files.update({'baseline.py': source.encode(), 'candidate.py': candidate.encode(),
        'LICENSE': (original/'LICENSE').read_bytes(), 'helper.py': (original/'worker.py').read_bytes(),
        'worker.py': Path(__file__).with_name('paired_snunet_worker.py').read_bytes()})
    request = {'seeds': list(seeds), 'steps': steps, 'device': profile['request']['device'],
        'samples': profile['request']['samples'], 'expected_unused': EXPECTED_UNUSED,
        'source_hashes': {n: sha(files[n+'.py']) for n in ('baseline', 'candidate', 'helper')},
        'protocol_sha256': sha(encoded(protocol))}
    files['request.json'] = encoded(request)
    manifest = {'version': 1, 'design_run_id': design_run_id, 'review_sha256': review_sha,
        'baseline_binding_id': baseline['binding_id'], 'draft_artifacts': material['draft_artifacts'],
        'python': profile['python'], 'python_sha256': profile['python_sha256'],
        'protocol': protocol, 'request': request, 'files': {n: sha(d) for n, d in files.items()}}
    binding = sha(encoded(manifest))
    root = runtime.workspace/'paired-method-profiles'/binding
    if not root.resolve().is_relative_to(runtime.workspace.resolve()):
        raise ValueError('Paired profile escapes workspace')
    root.mkdir(parents=True, exist_ok=True)
    for name, data in {**files, 'manifest.json': encoded(manifest)}.items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe paired asset path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('Refusing to overwrite modified paired profile')
        else:
            with path.open('xb') as handle:
                handle.write(data)
    return binding


def load_comparison(runtime, binding):
    if not isinstance(binding, str) or not re.fullmatch('[a-f0-9]{64}', binding):
        raise ValueError('Invalid comparison binding ID')
    root = runtime.workspace/'paired-method-profiles'/binding
    if not root.resolve().is_relative_to(runtime.workspace.resolve()):
        raise ValueError('Paired profile escapes workspace')
    manifest = json.loads((root/'manifest.json').read_bytes())
    if manifest.get('version') == 2:
        from .training_profiles import load_training
        return load_training(runtime, binding)
    if sha(encoded(manifest)) != binding:
        raise ValueError('Comparison protocol changed')
    for name, digest in manifest['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path.read_bytes()) != digest:
            raise ValueError('Paired source or samples changed')
    material = load_draft(runtime, manifest['design_run_id'])
    _, original = reference.load_profile(runtime.workspace, material['baseline']['binding_id'])
    request = manifest['request']
    protocol = comparison_protocol(original, request['seeds'], request['steps'])
    input_files = {n: d for n, d in original['files'].items() if n.startswith('inputs/')}
    expected_hashes = {'baseline': reference.SOURCE_SHA,
        'candidate': material['contract']['candidate_sha256'], 'helper': original['files']['worker.py']}
    expected_files = {**input_files, 'baseline.py': expected_hashes['baseline'],
        'candidate.py': expected_hashes['candidate'], 'helper.py': expected_hashes['helper'],
        'worker.py': sha(Path(__file__).with_name('paired_snunet_worker.py').read_bytes()),
        'LICENSE': reference.LICENSE_SHA, 'request.json': sha(encoded(request))}
    expected_request = {'seeds': list(request['seeds']), 'steps': request['steps'],
        'device': original['request']['device'], 'samples': original['request']['samples'],
        'expected_unused': EXPECTED_UNUSED, 'source_hashes': expected_hashes, 'protocol_sha256': sha(encoded(protocol))}
    if (manifest['protocol'] != protocol or request != expected_request or manifest['files'] != expected_files
            or manifest['python'] != original['python'] or manifest['python_sha256'] != original['python_sha256']):
        raise ValueError('Comparison differs from the bounded protocol and original frozen assets')
    if (_review(runtime, manifest['design_run_id'], material) != manifest['review_sha256']
            or material['draft_artifacts'] != manifest['draft_artifacts']
            or material['baseline']['binding_id'] != manifest['baseline_binding_id']
            or manifest['files']['worker.py'] != sha(Path(__file__).with_name('paired_snunet_worker.py').read_bytes())
            or manifest['files']['helper.py'] != sha(Path(__file__).with_name('snunet_worker.py').read_bytes())
            or manifest['files']['LICENSE'] != reference.LICENSE_SHA
            or sha(Path(manifest['python']).read_bytes()) != manifest['python_sha256']):
        raise ValueError('Comparison authority, worker or environment changed')
    check_ablation((root/'baseline.py').read_text(encoding='utf-8'), (root/'candidate.py').read_text(encoding='utf-8'))
    return root, manifest


def describe_comparison(runtime, binding):
    _, m = load_comparison(runtime, binding)
    if m['version'] == 2:
        from .training_profiles import describe_training
        return describe_training(runtime, binding)
    return {'binding_id': binding, 'design_run_id': m['design_run_id'], 'review_sha256': m['review_sha256'],
            'protocol': m['protocol'], 'source_hashes': m['request']['source_hashes'],
            'data_scope': {'dataset':'SYSU-CD', 'train_images':2, 'val_images':2, 'test_images':0},
            'limits': {'timeout_seconds':300, 'max_output_mb':32,
                       'memory_mb_requested':6144, 'hard_host_memory_limit':False},
            'output_location':str(runtime.workspace/'paired-method-runs'/'<owned-tool-call-id>'),
            'code_change':'Only bypass both ca/ca1 attention paths in SNUNet_ECAM.forward',
            'requires_approval': True, 'scientific_hypothesis_supported': False,
            'network_isolated': False, 'filesystem_isolated': False}


def verify_comparison(report, root, manifest):
    request = manifest['request']
    if (report.get('scope') != 'PAIRED_SNUNET_DIAGNOSTIC_ONLY'
            or report.get('source_hashes') != request['source_hashes']
            or report.get('protocol_sha256') != request['protocol_sha256']
            or report.get('device') != request['device'] or report.get('test_opened') is not False
            or report.get('environment', {}).get('deterministic_algorithms') is not True
            or report.get('scientific_hypothesis_supported') is not False):
        raise ValueError('Paired worker identity or scope mismatch')
    trials = report.get('trials')
    if (not isinstance(trials, list) or any(type(t.get('seed')) is not int for t in trials)
            or [t.get('seed') for t in trials] != request['seeds']):
        raise ValueError('Missing, reordered or duplicated paired seeds')
    def finite(values, positive=False):
        return (isinstance(values, list) and len(values) == request['steps']
            and all(type(x) in (int, float) and math.isfinite(x) and (not positive or x > 0) for x in values))
    verified = []
    for trial in trials:
        b, c = trial['baseline'], trial['candidate']
        effect = trial.get('pretraining_max_abs_logit_difference')
        if (not re.fullmatch('[a-f0-9]{64}', str(b.get('initial_state_sha256', '')))
                or b['initial_state_sha256'] != c.get('initial_state_sha256')
                or trial.get('pool_adapter_max_abs_error') != 0
                or type(effect) not in (int, float) or not math.isfinite(effect) or effect <= 0):
            raise ValueError('Unpaired initialization or ineffective candidate')
        row = {'seed': trial['seed'], 'initial_state_sha256': b['initial_state_sha256'],
               'pretraining_max_abs_logit_difference': effect, 'pool_adapter_max_abs_error': trial['pool_adapter_max_abs_error']}
        for variant in ('baseline', 'candidate'):
            record = trial[variant]
            expected_unused = EXPECTED_UNUSED if variant == 'candidate' else []
            if (type(record.get('steps_completed')) is not int or record['steps_completed'] != request['steps']
                    or record.get('train_order') != manifest['protocol']['training_order']
                    or record.get('weight_updated') is not True
                    or not finite(record.get('losses')) or not finite(record.get('gradient_norms'), positive=True)
                    or record.get('missing_gradients') != [expected_unused]*request['steps']):
                raise ValueError('Paired training or gradient contract failed: '+variant)
            row[variant] = {'validation': reference.verify_predictions(record, root, request),
                'losses': record['losses'], 'gradient_norms': record['gradient_norms'],
                'expected_unused_parameters': expected_unused}
        row['delta'] = {metric: (row['candidate']['validation'][metric] - row['baseline']['validation'][metric]
            if row['candidate']['validation'][metric] is not None and row['baseline']['validation'][metric] is not None
            else None) for metric in ('f1', 'iou')}
        verified.append(row)
    return verified


def run_comparison(context, args, *, runner=None):
    root, manifest = load_comparison(context, args['binding_id'])
    if manifest['version'] != 1:
        raise ValueError('Long-training bindings require the method_training runtime and its dedicated approval')
    parent = context.db.get_run(manifest['design_run_id'])
    active = context.db.get_run(context.run_id)
    if parent['metadata']['conversation_id'] != active['metadata']['conversation_id']:
        raise ValueError('Comparison is outside this conversation')
    attempt = context.workspace/'paired-method-runs'/context.tool_call_id
    if not attempt.resolve().is_relative_to(context.workspace.resolve()):
        raise ValueError('Comparison attempt escapes workspace')
    attempt.mkdir(parents=True, exist_ok=True)
    with (attempt/'started.json').open('xb') as handle:
        handle.write(encoded({'binding_id': args['binding_id']}))
    protocol = ExperimentProtocol(name='snunet-reviewed-paired-diagnostic',
        command=[manifest['python'], '-I', '-B', '-u', 'worker.py'], source_dir=str(root),
        output_dir=str(attempt), metric_files=['metrics.json'],
        hypothesis=manifest['protocol']['objective'], study_design=manifest['protocol'],
        policy=SandboxPolicy(backend='local', allow_local_fallback=True, network='host',
            timeout_seconds=300, cpu_seconds=300, memory_mb=6144, max_output_mb=32,
            env_allowlist=('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'CUDA_VISIBLE_DEVICES')))
    result = (runner or SandboxExperimentRunner(cancel_requested=context.cancelled)).run(protocol)
    verified, error = [], result.error
    try:
        if result.status != 'COMPLETED' or not result.source_unchanged:
            raise ValueError(result.error or
                f"Paired worker status={result.status}, exit_code={getattr(result, 'exit_code', None)}, "
                f"source_unchanged={result.source_unchanged}; inspect the persisted stderr artifact")
        load_comparison(context, args['binding_id'])
        verified = verify_comparison(result.metrics, root, manifest)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        error = str(exc)
    valid = bool(verified)
    summary = {'status': 'PAIRED_DIAGNOSTIC_VERIFIED' if valid else 'COMPARISON_FAILED',
        'binding_id': args['binding_id'], 'design_run_id': manifest['design_run_id'],
        'protocol': manifest['protocol'], 'trials': verified, 'error': error if not valid else None,
        'environment': result.metrics.get('environment'), 'result_path': result.result_path,
        'network_isolated': result.network_isolated, 'isolation_notes': result.isolation_notes,
        'scientific_hypothesis_supported': False, 'scientific_acceptance': 'NOT_ASSESSED',
        'limitations': ['Two previously seen diagnostic train/val pairs; not an independent benchmark',
                        'Step losses may use different images; not evidence of convergence',
                        'Host-local execution, not a hard security sandbox']}
    path = attempt/'verification.json'
    with path.open('xb') as handle:
        handle.write(encoded(summary))
    return ToolResult(ok=valid, output=summary, artifacts=result.artifacts + [{
        'kind': 'paired-method-verification', 'path': str(path), 'sha256': sha(path.read_bytes())}],
        error=None if valid else error, retryable=False)


def register_comparison_tool(registry):
    registry.register(ToolSpec(name='method.compare', description='Approved host-bound paired SNUNet attention ablation diagnostic. No arbitrary code or scientific success claim.',
        handler=run_comparison, risk_level=RiskLevel.HIGH, side_effects=True, requires_approval=True,
        timeout_seconds=330, max_attempts=1, tags=('method', 'paired-diagnostic', 'approval'),
        input_schema={'type': 'object', 'required': ['binding_id'], 'additionalProperties': False,
                      'properties': {'binding_id': {'type': 'string'}}}))


def enqueue_comparison(runtime, binding):
    card = describe_comparison(runtime, binding)
    if card['protocol']['version'] == 2:
        from .paired_training import enqueue_training
        return enqueue_training(runtime, binding)
    parent = runtime.db.get_run(card['design_run_id'])
    queued = runtime.enqueue(question='比较已审阅注意力消融的成对短训练；仅报告观察，不判定科学假设。',
        metadata={'engine': 'method_comparison', 'binding_id': binding, 'method_card': card,
                  'parent_run_id': parent['run_id'], 'conversation_id': parent['metadata']['conversation_id'],
                  'resolved_paper_sha256s': parent['metadata'].get('resolved_paper_sha256s', [])},
        budget=Budget(max_steps=1, max_tool_calls=1, max_llm_calls=0, max_seconds=600,
                      synthesis_reserve_calls=0, repair_reserve_calls=0))
    runtime.db.create_step(queued['run']['run_id'], 'method.compare', {'binding_id': binding})
    return runtime.status(queued['run']['run_id'])
