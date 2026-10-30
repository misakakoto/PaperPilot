"""Host-prepared, immutable SNUNet smoke profiles; not arbitrary generated-code execution."""
import base64
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import re

from .models import ExperimentProtocol, RiskLevel, SandboxPolicy, ToolResult
from .sandbox import SandboxExperimentRunner
from .tooling import ToolSpec

PIN = 'https://raw.githubusercontent.com/likyoo/Siam-NestedUNet/aed3f1833f3d9d3eccba8c894fd097f358b5329e/'
SOURCE_SHA = 'c14f3da927c21ea5503ac333a42afd3e5a9ed05efece962b637eb10eb15186c5'
LICENSE_SHA = '1483d321bc5234386be7eceda64de7087f1048cfa355d3c0a66b8200ec8d6244'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _samples(dataset, seed):
    """Freeze selection before reading labels; train/val only, preserve supplied split."""
    dataset = Path(dataset).resolve(strict=True)
    files, samples, splits, seen = {}, {}, {}, set()
    for split in ('train', 'val'):
        listing = dataset / (split + '.txt')
        if not listing.resolve().is_relative_to(dataset):
            raise ValueError('Split list escapes dataset')
        raw = listing.read_bytes()
        names = []
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            row = [PurePosixPath(x.replace('\\', '/')) for x in line.split()]
            if (len(row) != 3 or len({p.name for p in row}) != 1
                    or any('..' in p.parts or p.parts[-3:-1] != (split, role)
                           for p, role in zip(row, ('time1', 'time2', 'label')))):
                raise ValueError('Invalid split row')
            if row[0].name in seen:
                raise ValueError('Duplicate or overlapping train/val IDs')
            seen.add(row[0].name)
            names.append(row[0].name)
        if len(names) < 2:
            raise ValueError('Each split needs two diagnostic samples')
        splits[split] = {'count': len(names), 'list_sha256': sha(raw)}
        samples[split] = []
        for index, name in enumerate(random.Random(seed).sample(sorted(names), 2)):
            item = {'sample_id': name, 'hashes': {}}
            for role in ('time1', 'time2', 'label'):
                original = dataset / split / role / name
                if original.is_symlink() or not original.resolve().is_relative_to(dataset):
                    raise ValueError('Sample escapes dataset')
                if original.stat().st_size > 2_000_000:
                    raise ValueError('Diagnostic sample exceeds size bound')
                data = original.read_bytes()
                path = f'inputs/{split}-{index}-{role}.png'
                files[path], item[role], item['hashes'][role] = data, path, sha(data)
            samples[split].append(item)
    return files, samples, splits


def prepare_snunet(workspace, *, snapshot, dataset, python, seed=7, steps=2, device='cuda'):
    """Host-only preparation; never called with model-selected paths or commands."""
    if type(seed) is not int or not 0 <= seed < 2**32 or type(steps) is not int or not 1 <= steps <= 4:
        raise ValueError('Invalid seed or steps (1-4)')
    if device not in {'cpu', 'cuda'}:
        raise ValueError('Device must be cpu or cuda')
    workspace, dataset = Path(workspace).resolve(), Path(dataset).resolve(strict=True)
    if workspace.is_relative_to(dataset) or dataset.is_relative_to(workspace):
        raise ValueError('Keep the dataset outside the diagnostic workspace')
    snapshot = json.loads(Path(snapshot).read_text(encoding='utf-8'))
    source, license_text = snapshot['code'].encode(), snapshot['license_text'].encode()
    if (snapshot['code_url'] != PIN + 'models/Models.py' or sha(source) != SOURCE_SHA
            or snapshot['license_url'] != PIN + 'LICENSE' or sha(license_text) != LICENSE_SHA):
        raise ValueError('Only the inspected pinned upstream source and license are allowed')
    python = Path(python).resolve(strict=True)
    if not python.is_file() or python.name.lower() not in {'python', 'python.exe', 'python3'}:
        raise ValueError('Expected an explicitly selected Python interpreter')
    files, samples, splits = _samples(dataset, seed)
    files.update({'upstream.py': source, 'LICENSE': license_text,
                  'worker.py': Path(__file__).with_name('snunet_worker.py').read_bytes()})
    request = {'seed': seed, 'steps': steps, 'device': device, 'samples': samples}
    files['request.json'] = encoded(request)
    manifest = {'version': 1, 'method': 'SNUNet_ECAM', 'source_url': PIN + 'models/Models.py',
        'source_sha256': SOURCE_SHA, 'license_url': PIN + 'LICENSE',
        'paper_doi': '10.1109/LGRS.2021.3056416', 'python': str(python),
        'python_sha256': sha(python.read_bytes()), 'splits': splits, 'request': request,
        'files': {name: sha(data) for name, data in files.items()},
        'scope': 'random-weight adapter equivalence and tiny SYSU training diagnostic; NOT paper reproduction',
        'host_reviewed_source_only': True, 'requires_approval': True,
        'network_isolated': False, 'filesystem_isolated': False}
    binding_id = sha(encoded(manifest))
    root = workspace / 'method-profiles' / binding_id
    if not root.resolve().is_relative_to(workspace):
        raise ValueError('Method profile escapes workspace')
    root.mkdir(parents=True, exist_ok=True)
    for name, data in {**files, 'manifest.json': encoded(manifest)}.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Unsafe profile path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('Refusing to overwrite a modified profile')
        else:
            with path.open('xb') as handle:
                handle.write(data)
    return binding_id


def load_profile(workspace, binding_id):
    if not isinstance(binding_id, str) or not re.fullmatch('[a-f0-9]{64}', binding_id):
        raise ValueError('Invalid method binding ID')
    workspace = Path(workspace).resolve()
    root = workspace / 'method-profiles' / binding_id
    if not root.resolve().is_relative_to(workspace):
        raise ValueError('Profile escapes workspace')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if sha(encoded(manifest)) != binding_id:
        raise ValueError('Profile changed; prepare and approve a new binding')
    for name, digest in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path.read_bytes()) != digest:
            raise ValueError('Bound method assets changed')
    if (manifest['files'].get('upstream.py') != SOURCE_SHA
            or manifest['files'].get('LICENSE') != LICENSE_SHA
            or manifest['files'].get('worker.py') != sha(Path(__file__).with_name('snunet_worker.py').read_bytes())
            or sha(Path(manifest['python']).read_bytes()) != manifest['python_sha256']):
        raise ValueError('Unreviewed source, worker or changed interpreter')
    return root, manifest


def describe_profile(workspace, binding_id):
    _, manifest = load_profile(workspace, binding_id)
    return {key: manifest[key] for key in ('method', 'source_url', 'source_sha256', 'paper_doi', 'scope',
            'requires_approval', 'network_isolated', 'filesystem_isolated')} | {
        'binding_id': binding_id, 'seed': manifest['request']['seed'], 'steps': manifest['request']['steps'],
        'device': manifest['request']['device'],
        'data_scope': {'dataset':'SYSU-CD', 'train_images':2, 'val_images':2, 'test_images':0},
        'limits': {'timeout_seconds':180, 'max_output_mb':32,
                   'memory_mb_requested':6144, 'hard_host_memory_limit':False},
        'output_location':str(Path(workspace)/'method-runs'/'<owned-tool-call-id>'),
        'code_change':'Pinned upstream SNUNet with reviewed adapter; no arbitrary generated code',
        'validation_state': 'PROFILE_ONLY; consult the separate validation run receipt'}


def verify_predictions(report, root, request):
    """Independent stdlib pixel counts, against frozen labels; no worker metric reuse."""
    from PIL import Image
    rows = report.get('predictions')
    expected = request['samples']['val']
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Missing validation predictions')
    counts = [0, 0, 0, 0]
    for row, sample in zip(rows, expected):
        if row['sample_id'] != sample['sample_id'] or row['pixels'] != 256 * 256:
            raise ValueError('Prediction identity or size mismatch')
        bits = base64.b64decode(row['packed_bits'], validate=True)
        if len(bits) != 256 * 256 // 8:
            raise ValueError('Wrong packed prediction length')
        label = (root / sample['label']).read_bytes()
        if sha(label) != sample['hashes']['label']:
            raise ValueError('Frozen label changed')
        with Image.open(io.BytesIO(label)) as image:
            if image.mode != 'L' or image.size != (256, 256):
                raise ValueError('Wrong frozen label shape')
            truth = image.tobytes()
        for index, value in enumerate(truth):
            if value not in {0, 255}:
                raise ValueError('Unexpected label value')
            prediction = (bits[index // 8] >> (7 - index % 8)) & 1
            counts[2 * int(value == 255) + prediction] += 1
    tn, fp, fn, tp = counts
    expected_metrics = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        'iou': tp / (tp + fp + fn) if tp + fp + fn else None}
    for key, value in expected_metrics.items():
        actual = report.get('validation', {}).get(key)
        if value is None:
            matches = actual is None
        elif key in {'tn', 'fp', 'fn', 'tp'}:
            matches = type(actual) is int and actual == value
        else:
            matches = type(actual) in {int, float} and math.isclose(actual, value, rel_tol=0, abs_tol=1e-12)
        if not matches:
            raise ValueError('Independent metric verification failed: ' + key)
    return expected_metrics


def run_profile(context, args, *, runner=None):
    root, manifest = load_profile(context.workspace, args['binding_id'])
    # An interrupted side effect is never silently repeated after lease expiry.
    attempt = context.workspace / 'method-runs' / context.tool_call_id
    if not attempt.resolve().is_relative_to(context.workspace.resolve()):
        raise ValueError('Run directory escapes workspace')
    attempt.mkdir(parents=True, exist_ok=True)
    with (attempt / 'started.json').open('x', encoding='utf-8') as handle:
        json.dump({'binding_id': args['binding_id'], 'tool_call_id': context.tool_call_id}, handle)
    protocol = ExperimentProtocol(name='snunet-upstream-adapter-smoke',
        command=[manifest['python'], '-I', '-u', 'worker.py'], source_dir=str(root),
        output_dir=str(attempt), metric_files=['metrics.json'],
        hypothesis='Validate the adapter and training plumbing only, not a scientific hypothesis',
        study_design={'binding_id': args['binding_id'], 'source_sha256': SOURCE_SHA},
        policy=SandboxPolicy(backend='local', allow_local_fallback=True, network='host',
            timeout_seconds=180, cpu_seconds=180, memory_mb=6144, max_output_mb=32,
            env_allowlist=('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'CUDA_VISIBLE_DEVICES')))
    result = (runner or SandboxExperimentRunner(cancel_requested=context.cancelled)).run(protocol)
    report = result.metrics
    checks = report.get('parity_max_abs_errors')
    losses, gradients = report.get('losses'), report.get('gradient_norms')
    def finite_steps(values, positive=False):
        return (isinstance(values, list) and len(values) == manifest['request']['steps']
                and all(type(x) in {int, float} and math.isfinite(x) and (not positive or x > 0)
                        for x in values))
    valid = (result.status == 'COMPLETED' and result.source_unchanged
             and report.get('source_sha256') == SOURCE_SHA
             and report.get('scope') == 'UPSTREAM_ADAPTER_AND_SYSU_SMOKE_ONLY'
             and report.get('seed') == manifest['request']['seed']
             and report.get('steps_completed') == manifest['request']['steps']
             and report.get('device') == manifest['request']['device']
             and report.get('test_opened') is False and report.get('paper_accuracy_reproduced') is False
             and report.get('weight_updated') is True
             and finite_steps(losses) and finite_steps(gradients, positive=True)
             and isinstance(checks, list) and len(checks) == 2 and all(x == 0 for x in checks)
             and 0 < report.get('changed_input_max_abs_difference', 0) < float('inf'))
    error = result.error
    verified = None
    if valid:
        try:
            load_profile(context.workspace, args['binding_id'])
            verified = verify_predictions(report, root, manifest['request'])
        except (ValueError, KeyError, OSError) as exc:
            valid, error = False, str(exc)
    summary = {'binding_id': args['binding_id'], 'method': manifest['method'],
        'status': 'ADAPTER_SMOKE_VERIFIED' if valid else 'VALIDATION_FAILED',
        'validation': verified, 'steps_completed': report.get('steps_completed'),
        'parity_max_abs_errors': checks, 'environment': report.get('environment'),
        'losses': losses if finite_steps(losses) else None,
        'gradient_norms': gradients if finite_steps(gradients, positive=True) else None,
        'test_opened': report.get('test_opened'), 'paper_accuracy_reproduced': False,
        'scientific_hypothesis_supported': False, 'network_isolated': result.network_isolated,
        'isolation_notes': result.isolation_notes, 'result_path': result.result_path,
        'limitations': ['Random initialization, two train/val pairs, no accuracy claim',
                        'Direct/adapter equivalence is NOT independent scientific reproduction',
                        'Host-local execution; no hard filesystem/network isolation']}
    with (attempt / 'verification.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return ToolResult(ok=bool(valid), output=summary, artifacts=result.artifacts + [{
        'kind': 'method-verification', 'path': str(attempt / 'verification.json'),
        'sha256': sha((attempt / 'verification.json').read_bytes())}],
        error=None if valid else error or 'Worker contract did not pass', retryable=False)


def register_reference_tools(registry):
    from .paired_methods import register_comparison_tool
    register_comparison_tool(registry)
    schema = {'type': 'object', 'required': ['binding_id'], 'additionalProperties': False,
              'properties': {'binding_id': {'type': 'string'}}}
    registry.register(ToolSpec(name='method.describe', description='Inspect a host-prepared immutable method profile; never execute.',
        handler=lambda ctx, args: ToolResult(ok=True, output=describe_profile(ctx.workspace, args['binding_id'])),
        input_schema=schema, tags=('method', 'read-only')))
    registry.register(ToolSpec(name='method.validate',
        description='Run the approved pinned SNUNet adapter smoke on frozen SYSU train/val samples. NOT paper reproduction. Host-local, not a security sandbox.',
        handler=run_profile, input_schema=schema, risk_level=RiskLevel.HIGH,
        side_effects=True, requires_approval=True, timeout_seconds=210, max_attempts=1,
        tags=('method', 'diagnostic', 'approval')))


def enqueue_validation(runtime, binding_id, *, parent_run_id=None):
    from .models import Budget
    card = describe_profile(runtime.workspace, binding_id)
    metadata = {'engine': 'method_validation', 'binding_id': binding_id, 'method_card': card,
                'resolved_paper_sha256s': []}
    if parent_run_id:
        parent = runtime.db.get_run(parent_run_id)
        if parent['status'] != 'COMPLETED':
            raise ValueError('Parent reading/design run must be completed')
        metadata.update(parent_run_id=parent_run_id, conversation_id=parent['metadata']['conversation_id'],
                        resolved_paper_sha256s=parent['metadata'].get('resolved_paper_sha256s', []))
    queued = runtime.enqueue(question='验证固定方法的封装、短训练与指标；不是研究假设验证。',
        metadata=metadata, budget=Budget(max_steps=1, max_tool_calls=1, max_llm_calls=0,
            max_seconds=600, synthesis_reserve_calls=0, repair_reserve_calls=0))
    run_id = queued['run']['run_id']
    runtime.db.create_step(run_id, 'method.validate', {'binding_id': binding_id})
    return runtime.status(run_id)


def resume_validation(runtime, run_id):
    from datetime import datetime, timezone
    from .models import Budget, ResearchOutcome, RunStatus, StepStatus
    db, run = runtime.db, runtime.db.get_run(run_id)
    comparison = run['metadata'].get('engine') == 'method_comparison'
    tool_name = 'method.compare' if comparison else 'method.validate'
    if run['status'] in {'COMPLETED', 'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'}:
        return runtime.status(run_id)
    try:
        budget = Budget.from_dict(run['budget'])
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))).total_seconds()
        if elapsed >= budget.max_seconds:
            db.update_run_status(run_id, RunStatus.BUDGET_EXHAUSTED, error='Method validation time budget exhausted')
            return runtime.status(run_id)
        binding = run['metadata']['binding_id']
        if comparison:
            from .paired_methods import describe_comparison
            card = describe_comparison(runtime, binding)
        else:
            card = describe_profile(runtime.workspace, binding)
        steps = db.list_steps(run_id)
        if len(steps) != 1 or steps[0]['kind'] != tool_name or steps[0]['input'] != {'binding_id': binding}:
            raise ValueError('Validation requires its single preregistered step')
        step = steps[0]['step_id']
        calls = db.list_tool_calls(run_id)
        if calls and calls[0]['status'] in {'APPROVED', 'RUNNING'}:
            db.update_run_status(run_id, RunStatus.RUNNING, current_step_id=step)
            db.update_step(step, StepStatus.RUNNING)
        outcome = runtime.tools.invoke(run_id=run_id, step_id=step,
            tool_name=tool_name, arguments={'binding_id': binding}, approval_context=card)
        if outcome.status.value in {'WAITING_APPROVAL', 'RUNNING', 'APPROVED'}:
            return runtime.status(run_id)
        if db.get_run(run_id)['status'] == 'CANCELLED':
            return runtime.status(run_id)
        result = outcome.result
        passed = outcome.status.value == 'COMPLETED' and result is not None and result.ok
        if result:
            for artifact in result.artifacts:
                artifact = {**artifact, 'artifact_id': 'artifact_method_' + sha(encoded([run_id, artifact['path']]))[:24]}
                db.add_artifact(run_id, step, artifact)
            if comparison and passed:
                answer = '成对短训练与独立指标核验完成；以下仅为固定小样本观察，不检验性能非劣效或方法有效性。'
                for row in result.output['trials']:
                    answer += ('\nseed=' + str(row['seed']) + ': baseline=' +
                               str(row['baseline']['validation']) + '; candidate=' + str(row['candidate']['validation']))
            else:
                answer = ('固定方法的封装和短训练检查通过；未复现论文精度，也未验证新方法。'
                          if passed else '方法验证/对照未通过，不能作为成功实验。')
            db.update_run_metadata(run_id, {('method_comparison' if comparison else 'method_validation'): result.output,
                'final_answer': {'answer': answer,
                    'answer_scope': 'PARTIAL', 'review_required': True}})
        db.update_step(step, StepStatus.COMPLETED if passed else StepStatus.FAILED,
            output=result.output if result else {}, error=result.error if result else 'Method execution rejected')
        db.update_run_outcome(run_id, ResearchOutcome.PARTIAL if passed else ResearchOutcome.UNRESOLVED)
        db.update_run_status(run_id, RunStatus.COMPLETED if passed else RunStatus.FAILED,
            error=None if passed else (result.error if result else 'Method execution rejected'))
    except Exception as exc:
        if db.get_run(run_id)['status'] != 'CANCELLED':
            db.update_run_status(run_id, RunStatus.FAILED, error=f'{type(exc).__name__}: {exc}')
    return runtime.status(run_id)
