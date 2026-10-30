"""Read-only experiment observations from frozen predictions, never model scores.

The historic worker and recipe are evidence identities, not executable authority.
No recipe loader, interpreter, checkpoint or GPU is started by this module.
"""
import base64
import json
from pathlib import Path

from .reference_methods import encoded, sha
from .research_assets import immutable, owned_folder
from .research_training import episode_folder
from .training_metrics import verify_trial, paired_summary
from .training_state import file_sha
from .diagnostic_core import GROUPS, metrics, coverage_group, views as diagnostic_views

VERSION = 'research-diagnostics-v2-label-coverage'
SEEDS = (7, 11, 19)
VIEW_CONTRACTS = {
    'paired': {'meaning': 'All three seeds: baseline/candidate global pixel confusion counts and metrics',
        'same_frozen_validation_ids_and_labels_across_seeds': True,
        'threshold_observation': 'No threshold interventions, logits or probabilities are observed in these views'},
    'coverage': {'grouping_basis': 'Fraction of ground-truth label pixels equal to 255 in each image',
        'groups': list(GROUPS), 'not_measured': ['prediction confidence', 'activation', 'object size'],
        'aggregation': 'Sum confusion counts within each label-coverage group, then compute metrics'},
    'samples': {'ranking': 'Top five strictly positive NET increases: candidate minus baseline FN/FP pixel counts',
        'ties': 'sample_id ascending', 'not_measured': ['confidence scores', 'activation distributions'],
        'prediction_kind': 'Frozen binary class predictions; no logits or probability scores'},
}


def safe_file(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Diagnostic source escapes frozen asset root')
    return path


def freeze_experiment(runtime, parent_run_id, candidate_id):
    """Register a verified historical experiment; do not reopen its execution."""
    import numpy as np
    from PIL import Image
    from .paired_training import _verify_reference

    parent = runtime.db.get_run(parent_run_id)
    if parent['metadata'].get('engine') != 'research_loop_v1':
        raise ValueError('Select an owned research episode')
    with runtime.db.connect() as conn:
        row = conn.execute('SELECT state_json FROM research_episodes WHERE run_id=?',
                           (parent_run_id,)).fetchone()
    if row is None:
        raise ValueError('Missing historical episode')
    state = json.loads(row['state_json'])
    candidates = [c for c in state['candidates']
                  if c.get('card', {}).get('binding_id') == candidate_id]
    if len(candidates) != 1:
        raise ValueError('Candidate not owned by selected parent')
    candidate = candidates[0]
    result = candidate.get('result', {})
    if result.get('status') != 'TRAINING_VERIFIED':
        raise ValueError('Only fully verified paired experiments can be diagnosed')
    original = episode_folder(runtime, parent_run_id) / ('result-' + candidate_id + '.json')
    if original.read_bytes() != encoded(result):
        raise ValueError('Historical experiment result changed')
    root = owned_folder(runtime, 'candidates', candidate_id)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if (sha(encoded(manifest)) != candidate_id or manifest.get('episode_id') != parent_run_id
            or manifest.get('version') != 3):
        raise ValueError('Historical manifest identity mismatch')
    assets = {str(original): file_sha(original), str(manifest_path): file_sha(manifest_path)}
    for name, digest in manifest['files'].items():
        if name.endswith('.py'):
            path = safe_file(root, name)
            if file_sha(path) != digest:
                raise ValueError('Historical source code changed')
            assets[str(path)] = digest
    expected_samples = manifest['request']['samples']['val']
    truths, groups = {}, {}
    for sample in expected_samples:
        sid = sample['sample_id']
        if sid in truths:
            raise ValueError('Duplicate validation sample identity')
        path = safe_file(root, sample['label'])
        digest = file_sha(path)
        if digest != sample['hashes']['label']:
            raise ValueError('Frozen validation label changed')
        assets[str(path)] = digest
        with Image.open(path) as image:
            truth = np.asarray(image).reshape(-1) == 255
        truths[sid] = truth
        groups[sid] = coverage_group(int(truth.sum()), len(truth))
    if [pair['seed'] for pair in result.get('trials', [])] != list(SEEDS):
        raise ValueError('Every seed is required exactly once in frozen order')
    samples, trials = {}, {}
    source_receipts = {}
    for pair in result['trials']:
        seed = pair['seed']
        samples[str(seed)] = {}
        for variant in ('baseline', 'candidate'):
            trial = pair[variant]
            if trial.get('status') != 'VERIFIED':
                raise ValueError('Missing verified trial')
            ref = trial['receipt']
            raw = _verify_reference(ref, original.parent / candidate_id)
            checked = verify_trial(raw, root, manifest, seed=seed, variant=variant)
            if checked != trial['metrics'] or raw['initial_state_sha256'] != trial['initial_state_sha256']:
                raise ValueError('Historical metric projection changed')
            assets[str(Path(ref['path']).resolve())] = ref['sha256']
            source_receipts[f'{seed}/{variant}'] = ref
            trials[f'{seed}-{variant}'] = trial
            for prediction in raw['predictions']:
                sid = prediction['sample_id']
                pred = np.unpackbits(np.frombuffer(base64.b64decode(prediction['packed_bits'],
                    validate=True), dtype=np.uint8), bitorder='big')
                truth = truths[sid]
                counts = [int(v) for v in np.bincount(truth.astype(np.uint8) * 2 + pred,
                                                     minlength=4)]
                entry = samples[str(seed)].setdefault(sid, {'sample_id': sid,
                    'group': groups[sid], 'positive_pixels': int(truth.sum()), 'pixels': len(truth)})
                entry[variant] = metrics(counts)
    paired = paired_summary(trials)  # Also verifies matching initial states.
    if any(p['delta'] != old['delta'] for p, old in zip(paired, result['trials'])):
        raise ValueError('Historical paired delta disagrees with independent recomputation')
    paired = [{'seed': p['seed'], 'baseline': p['baseline']['metrics'],
               'candidate': p['candidate']['metrics'], 'delta': p['delta']} for p in paired]
    views = diagnostic_views(samples, paired)
    payload = {'version': VERSION, 'parent_run_id': parent_run_id, 'candidate_id': candidate_id,
        'assets': assets, 'source_receipts': source_receipts,
        'historic_implementation': manifest['files'], 'views': views,
        'all_samples': samples, 'view_contracts': VIEW_CONTRACTS,
        'units': 'metric fractions on [0,1]; counts are pixels',
        'validation_scope': 'Previously used development validation; not independent holdout',
        'new_gpu_starts': 0}
    identity = sha(encoded(payload))
    folder = owned_folder(runtime, 'diagnostic-inputs', identity)
    path = immutable(folder, 'bundle.json', encoded(payload))
    return {'identity': identity, 'bundle_sha256': file_sha(path)}


def load_bundle(runtime, binding, *, verify_assets=True):
    path = owned_folder(runtime, 'diagnostic-inputs', binding['identity']) / 'bundle.json'
    raw = path.read_bytes()
    if sha(raw) != binding['bundle_sha256'] or sha(raw) != binding['identity']:
        raise ValueError('Diagnostic bundle identity changed')
    result = json.loads(raw)
    from .experiment_import import VERSION as IMPORT_VERSION
    from .diagnostic_core import VERSION as CORE_VERSION
    if result['version'] not in {VERSION, IMPORT_VERSION}:
        raise ValueError('Diagnostic protocol changed')
    if result['version'] == IMPORT_VERSION and result.get('core_version') != CORE_VERSION:
        raise ValueError('Imported diagnostic calculation protocol changed')
    if result['version'] == IMPORT_VERSION:
        positive = result.get('encoding', {}).get('positive')
        expected = f'Fraction of ground-truth label pixels equal to {positive} in each image'
        if (positive not in (1, 255) or
                result.get('view_contracts', {}).get('coverage', {}).get('grouping_basis') != expected):
            raise ValueError('Imported grouping contract conflicts with encoding; re-register source bundle')
    if verify_assets:
        for name, digest in result['assets'].items():
            path = Path(name)
            if path.is_symlink() or not path.resolve().is_relative_to(runtime.workspace.resolve()) or file_sha(path) != digest:
                raise ValueError('Frozen diagnostic input changed')
    return result


def observation(runtime, binding, view):
    if view not in {'paired', 'coverage', 'samples'}:
        raise ValueError('Unknown diagnostic view')
    bundle = load_bundle(runtime, binding)
    rows = bundle['views'][view]
    contract = bundle['view_contracts'][view]
    identity = sha(encoded({'bundle': binding['identity'], 'view': view, 'rows': rows, 'contract': contract}))
    facts = {}
    def collect(value, prefix):
        if isinstance(value, dict):
            for key, item in value.items():
                # Transport paths/hashes and worker histories are not scientific facts.
                if key not in {'receipt', 'losses', 'gradient_norms'}:
                    collect(item, prefix + '/' + str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                collect(item, prefix + '/' + str(i))
        elif value is None or type(value) in (int, float):
            facts[prefix] = value
    collect(rows, view)
    return {'observation_id': identity, 'view': view, 'rows': rows, 'facts': facts,
        'view_contract': contract,
        'units': bundle['units'], 'validation_scope': bundle['validation_scope'],
        'assurance': 'HOST_RECOMPUTED; interpretations and mechanisms not certified',
        'source_identity': binding['identity'], 'new_gpu_starts': 0}
