"""Candidate-bound adapter over the existing worker, GPU ledger and metric verifier.

V2 files/checkpoints remain untouched. The new worker is the frozen trusted kernel
with ONLY its ledger cap changed. Model code is never allowed to supply a runner.
"""
import json
from pathlib import Path

from .paired_training import run_job, _reference, _verify_reference, confirm_exits
from .reference_methods import encoded, sha
from .research_assets import (GPU_SECONDS, immutable, load_recipe, owned_folder,
                              build_fusion_patch)
from .training_metrics import paired_summary, verify_trial
from .training_state import GpuBudget, atomic_json, file_sha, TrainingBusy
from .research_task import task_lock


def episode_folder(runtime, run_id):
    return owned_folder(runtime, 'episodes', sha(run_id.encode()))


def gpu_ledger(runtime, run_id):
    return GpuBudget(episode_folder(runtime, run_id)/'gpu-budget.sqlite3', GPU_SECONDS)


def candidate_worker(kernel):
    old = "ledger = state.GpuBudget(job['ledger'])"
    if kernel.count(old) != 1:
        raise ValueError('Trusted worker ledger seam changed; freeze another implementation')
    return kernel.replace(old, f"ledger = state.GpuBudget(job['ledger'], cap_seconds={GPU_SECONDS})")


def freeze_candidate(runtime, run_id, recipe_id, number, proposal, fusion_body):
    recipe, base_root, base = load_recipe(runtime, recipe_id)
    patch = build_fusion_patch((base_root/'baseline.py').read_text(encoding='utf-8'), fusion_body)
    protocol = dict(base['protocol'], version=3, recipe='snunet_reviewed_fusion_v3',
                    gpu_cap_seconds=GPU_SECONDS, candidate_number=number,
                    validation_scope=recipe['validation_scope'])
    request = dict(base['request'], expected_unused=patch['expected_unused'],
                   protocol_sha256=sha(encoded(protocol)))
    worker = candidate_worker((base_root/'worker.py').read_text(encoding='utf-8')).encode()
    files = {n: digest for n, digest in base['files'].items()
             if n.startswith('inputs/') or n in {'baseline.py', 'LICENSE', 'support.py', 'training_state.py'}}
    generated = {'candidate.py': patch['candidate_source'].encode(), 'worker.py': worker,
                 'request.json': encoded(request), 'candidate.patch': patch['diff'].encode()}
    files.update({n: sha(raw) for n, raw in generated.items()})
    manifest = {'version': 3, 'episode_id': run_id, 'recipe_id': recipe_id,
        'candidate_number': number, 'proposal': proposal, 'fusion_body': patch['fusion_body'],
        'python': base['python'], 'python_sha256': base['python_sha256'],
        'environment': recipe.get('execution_environment', base['environment']), 'protocol': protocol, 'request': request,
        'files': files, 'source_authority': recipe['training_binding']}
    binding = sha(encoded(manifest))
    folder = owned_folder(runtime, 'candidates', binding)
    for name in files:
        immutable(folder, name, generated[name] if name in generated else (base_root/name).read_bytes())
    immutable(folder, 'manifest.json', encoded(manifest))
    card = {'binding_id': binding, 'recipe_id': recipe_id, 'candidate_number': number,
        'hypothesis': proposal, 'diff': patch['diff'], 'static_checks': patch['static_checks'],
        'not_checked': patch['not_checked'], 'protocol': protocol,
        'data_scope': {'dataset': 'SYSU-CD', 'train': 512, 'val': 256, 'test': 0,
                       'scope': recipe['validation_scope']},
        'cost': {'episode_gpu_cap_seconds': GPU_SECONDS,
                 'parameter_updates': 15360, 'estimate': 'Prior v2 six-trial campaign took about 5815 seconds; candidate cost must pass preflight'},
        'source_hashes': {n: files[n] for n in ('baseline.py', 'candidate.py', 'worker.py')},
        'environment': manifest['environment'], 'environment_delta': recipe.get('environment_delta', {}),
        'network_isolated': False, 'filesystem_isolated': False,
        'requires_approval': True, 'scientific_hypothesis_supported': False}
    immutable(folder, 'card.json', encoded(card))
    return card


def load_candidate(runtime, binding):
    root = owned_folder(runtime, 'candidates', binding)
    manifest = json.loads((root/'manifest.json').read_bytes())
    if sha(encoded(manifest)) != binding or manifest.get('version') != 3:
        raise ValueError('Candidate protocol identity changed')
    recipe, base_root, base = load_recipe(runtime, manifest['recipe_id'])
    patch = build_fusion_patch((base_root/'baseline.py').read_text(encoding='utf-8'), manifest['fusion_body'])
    expected_protocol = dict(base['protocol'], version=3, recipe='snunet_reviewed_fusion_v3',
        gpu_cap_seconds=GPU_SECONDS, candidate_number=manifest['candidate_number'],
        validation_scope='Previously used development subset, NOT an independent holdout')
    if (manifest['protocol'] != expected_protocol
            or manifest['files']['candidate.py'] != patch['candidate_sha256']
            or manifest['request'] != dict(base['request'], expected_unused=patch['expected_unused'],
                protocol_sha256=sha(encoded(expected_protocol)))
            or manifest['python'] != base['python'] or manifest['python_sha256'] != file_sha(base['python'])
            or manifest['environment'] != recipe.get('execution_environment', base['environment'])
            or manifest['files']['worker.py'] != sha(candidate_worker((base_root/'worker.py').read_text(encoding='utf-8')).encode())):
        raise ValueError('Candidate differs from reviewed fusion/host training recipe')
    for name, digest in manifest['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or file_sha(path) != digest:
            raise ValueError('Frozen candidate/input changed: '+name)
    card = json.loads((root/'card.json').read_bytes())
    if (card['binding_id'] != binding or card['source_hashes']['candidate.py'] != patch['candidate_sha256']
            or card['diff'] != patch['diff'] or card['protocol'] != manifest['protocol']
            or card['hypothesis'] != manifest['proposal']):
        raise ValueError('Candidate review card changed')
    return root, manifest, card


def execute_candidate(runtime, run_id, binding, worker_run_id, *, runner=run_job):
    # A file lock alone disappears when the host dies while its worker survives.
    # Check all owned campaign ledgers as well before allocating this GPU.
    with task_lock(runtime.workspace, 'research-gpu') as acquired:
        if not acquired:
            raise TrainingBusy('Another research episode currently owns the GPU')
        for path in (runtime.workspace/'research-harness'/'episodes').glob('*/gpu-budget.sqlite3'):
            other = GpuBudget(path, GPU_SECONDS)
            other.recover()
            confirm_exits(other)
        return _execute_candidate(runtime, run_id, binding, worker_run_id, runner=runner)


def _execute_candidate(runtime, run_id, binding, worker_run_id, *, runner=run_job):
    """Resume committed stages; every receipt is rechecked before model analysis."""
    root, manifest, _ = load_candidate(runtime, binding)
    if manifest['episode_id'] != run_id:
        raise ValueError('Candidate belongs to another episode')
    output = episode_folder(runtime, run_id)/binding
    output.mkdir(parents=True, exist_ok=True)
    state_path = output/'state.json'
    state = json.loads(state_path.read_bytes()) if state_path.exists() else {'trials': {}, 'checks': {}}
    ledger = gpu_ledger(runtime, run_id)
    ledger.recover()
    for check in state['checks'].values():
        for key in ('receipt', 'continuous', 'resumed'):
            if key in check:
                _verify_reference(check[key], output)
    for trial in state['trials'].values():
        if trial['status'] == 'VERIFIED':
            raw = _verify_reference(trial['receipt'], output)
            trial['metrics'] = verify_trial(raw, root, manifest,
                seed=raw['identity']['seed'], variant=raw['identity']['variant'])
    if not state['checks'].get('preflight'):
        raw, path = runner(runtime, worker_run_id, root, manifest, output/'preflight', ledger,
                           stage='preflight', variant='candidate')
        if raw['status'] != 'COMPLETED':
            return result(state, ledger, 'PAUSED', 'Candidate preflight checkpoint retained')
        estimate = raw['launch_elapsed_seconds']/manifest['protocol']['preflight']['steps']*15360*1.25
        if (not raw.get('learning', {}).get('passed') or raw['peak_memory_bytes'] > 15*1024**3
                or estimate > ledger.snapshot()['remaining_seconds']):
            return result(state, ledger, 'PREFLIGHT_REJECTED', 'Learning/memory/time gate failed; protocol not changed')
        state['checks']['preflight'] = {'receipt': _reference(path), 'estimated_formal_seconds': estimate}
        atomic_json(state_path, state)
    if not state['checks'].get('recovery'):
        full, p1 = runner(runtime, worker_run_id, root, manifest, output/'recovery-continuous', ledger,
                          stage='recovery', variant='candidate')
        split, p2 = runner(runtime, worker_run_id, root, manifest, output/'recovery-resumed', ledger,
                           stage='recovery', variant='candidate', pause_at=3)
        if split['status'] == 'PAUSED':
            split, p2 = runner(runtime, worker_run_id, root, manifest, output/'recovery-resumed', ledger,
                               stage='recovery', variant='candidate')
        keys = ('step', 'final_state_sha256', 'optimizer_sha256', 'rng_sha256', 'losses', 'gradient_norms')
        if full['status'] != 'COMPLETED' or split['status'] != 'COMPLETED' or any(full[k] != split[k] for k in keys):
            raise ValueError('Candidate continuous/resumed training differs')
        state['checks']['recovery'] = {'continuous': _reference(p1), 'resumed': _reference(p2), 'equal': list(keys)}
        atomic_json(state_path, state)
    for seed in (7, 11, 19):
        for variant in ('baseline', 'candidate'):
            name = f'{seed}-{variant}'
            if state['trials'].get(name, {}).get('status') == 'VERIFIED':
                continue
            if runtime.db.get_run(run_id)['metadata'].get('cancel_requested'):
                return result(state, ledger, 'PAUSED', 'Cancellation requested')
            if ledger.snapshot()['remaining_seconds'] <= 30:
                return result(state, ledger, 'PARTIAL', 'Cumulative GPU budget exhausted')
            raw, path = runner(runtime, worker_run_id, root, manifest, output/'trials'/name, ledger,
                               stage='formal', seed=seed, variant=variant)
            if raw['status'] == 'PAUSED':
                state['trials'][name] = {'status': 'PAUSED', 'step': raw['step'], 'receipt': _reference(path)}
                atomic_json(state_path, state)
                return result(state, ledger, 'PAUSED', 'Checkpoint retained')
            metrics = verify_trial(raw, root, manifest, seed=seed, variant=variant)
            state['trials'][name] = {'status': 'VERIFIED', 'metrics': metrics,
                'initial_state_sha256': raw['initial_state_sha256'], 'receipt': _reference(path),
                'elapsed_seconds': raw['elapsed_seconds'], 'peak_memory_bytes': raw['peak_memory_bytes']}
            paired_summary(state['trials'])
            atomic_json(state_path, state)
            runtime.db.emit_event(run_id, 'research.trial_verified', {'binding_id': binding, 'seed': seed, 'variant': variant, 'metrics': metrics})
    return result(state, ledger, 'TRAINING_VERIFIED')


def result(state, ledger, status, error=None):
    return {'status': status, 'trials': paired_summary(state['trials']), 'checks': state['checks'],
            'gpu_budget': ledger.snapshot(), 'error': error,
            'scientific_acceptance': 'DESCRIPTIVE_ONLY',
            'limitations': ['Previously used 512 train/256 development val; no test',
                            'Three seeds do not establish significance or novelty',
                            'Reviewed host-local execution; not a hard security sandbox']}


def failure_result(runtime, run_id, binding, error):
    path = episode_folder(runtime, run_id)/binding/'state.json'
    state = json.loads(path.read_bytes()) if path.exists() else {'trials': {}, 'checks': {}}
    # A source/receipt revalidation may itself have failed. Cached scores must not
    # be promoted into the analyst's verified catalog after such a failure.
    for trial in state['trials'].values():
        if trial.get('status') == 'VERIFIED':
            trial['status'] = 'UNVERIFIED_AFTER_FAILURE'
    return result(state, gpu_ledger(runtime, run_id), 'EXECUTION_FAILED', error)
