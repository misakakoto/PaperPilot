"""Host-only registration of discrete binary PNG experiments; never executes code."""
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from .diagnostic_core import VERSION as CORE_VERSION, metrics, coverage_group, views
from .reference_methods import encoded, sha
from .research_assets import immutable, owned_folder

VERSION = 'experiment-bundle-v1'


def read_mask(root, name, encoding):
    import numpy as np
    from PIL import Image
    if (not isinstance(name, str) or not name or '\\' in name
            or PureWindowsPath(name).drive or PurePosixPath(name).is_absolute()
            or '..' in PurePosixPath(name).parts):
        raise ValueError('Mask path must be relative to the result bundle')
    path = root / name
    if (not path.resolve(strict=True).is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)):
        raise ValueError('Mask path escapes bundle or uses a symlink')
    raw = path.read_bytes()
    with Image.open(BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode != 'L':
            raise ValueError('Only grayscale L-mode PNG masks are supported')
        values = np.asarray(image)
        if values.ndim != 2 or not values.size or values.size > 67108864:
            raise ValueError('Invalid mask dimensions')
        if not set(np.unique(values)).issubset(set(encoding.values())):
            raise ValueError('Mask has undeclared labels; probabilities and ignore labels unsupported')
        return values == encoding['positive'], raw


def _indexed(items, label, fields):
    if not isinstance(items, list) or not items:
        raise ValueError(label + ' must be a nonempty list')
    result = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(label + ' has unknown/missing fields')
        sid = item.get('sample_id')
        if not isinstance(sid, str) or not sid.strip() or len(sid) > 200 or sid in result:
            raise ValueError(label + ' has invalid or duplicate sample ID')
        result[sid] = item
    return result


def register_experiment(runtime, manifest_path):
    """Freeze an input pack by content. No model/key needed; repeat imports reuse IDs."""
    import numpy as np
    from .research_diagnostics import VIEW_CONTRACTS
    path = Path(manifest_path).resolve(strict=True)
    spec = json.loads(path.read_bytes())
    required = {'version', 'task', 'dataset', 'encoding', 'samples', 'trials'}
    if not isinstance(spec, dict) or set(spec) != required or spec['version'] != VERSION:
        raise ValueError('Expected experiment-bundle-v1 manifest')
    if spec['task'] != 'binary_change_detection':
        raise ValueError('Only binary_change_detection is supported')
    dataset = spec['dataset']
    if (not isinstance(dataset, dict) or set(dataset) != {'name', 'version', 'split', 'evaluation_scope'}
            or any(not isinstance(v, str) or not v.strip() for v in dataset.values())):
        raise ValueError('Dataset identity and user-declared evaluation scope required')
    encoding = spec['encoding']
    if (not isinstance(encoding, dict) or set(encoding) != {'negative', 'positive'}
            or any(type(v) is not int for v in encoding.values())
            or encoding['negative'] != 0 or encoding['positive'] not in (1, 255)):
        raise ValueError('Explicit binary encoding must be 0/1 or 0/255; ignore labels unsupported')
    labels = _indexed(spec['samples'], 'samples', {'sample_id', 'label'})
    if not isinstance(spec['trials'], list) or not spec['trials']:
        raise ValueError('Supply baseline and candidate trials')
    trials = {}
    for trial in spec['trials']:
        if (not isinstance(trial, dict) or set(trial) != {'seed', 'variant', 'predictions', 'training_provenance'}
                or type(trial['seed']) is not int or trial['seed'] < 0
                or trial['variant'] not in ('baseline', 'candidate')
                or not isinstance(trial['training_provenance'], dict)):
            raise ValueError('Invalid trial declaration')
        key = (trial['seed'], trial['variant'])
        if key in trials:
            raise ValueError('Duplicate seed/variant')
        predictions = _indexed(trial['predictions'], 'predictions', {'sample_id', 'path'})
        if set(predictions) != set(labels):
            raise ValueError('Paired predictions must match the complete label sample IDs')
        trials[key] = (trial, predictions)
    seeds = sorted({s for s, _ in trials})
    if set(trials) != {(s, v) for s in seeds for v in ('baseline', 'candidate')}:
        raise ValueError('Every declared seed requires baseline and candidate')
    truths, groups, assets, source_files = {}, {}, {}, {}
    def freeze(name):
        values, raw = read_mask(path.parent, name, encoding)
        digest = sha(raw)
        target = immutable(owned_folder(runtime, 'imported-masks', digest), 'mask.png', raw)
        assets[str(target)] = digest
        return values, digest
    for sid in sorted(labels):
        truth, digest = freeze(labels[sid]['label'])
        truths[sid] = truth
        groups[sid] = coverage_group(int(truth.sum()), int(truth.size))
        source_files['label/' + sid] = digest
    samples, paired, provenance = {}, [], []
    for seed in seeds:
        samples[str(seed)] = {}
        pair = {'seed': seed}
        for variant in ('baseline', 'candidate'):
            trial, predictions = trials[(seed, variant)]
            total = [0, 0, 0, 0]
            provenance.append({'seed': seed, 'variant': variant,
                'status': 'USER_DECLARED_NOT_INDEPENDENTLY_VERIFIED', 'claims': trial['training_provenance']})
            for sid in sorted(labels):
                pred, digest = freeze(predictions[sid]['path'])
                truth = truths[sid]
                if pred.shape != truth.shape:
                    raise ValueError('Prediction/label shape mismatch for sample ' + sid)
                source_files[f'{seed}/{variant}/{sid}'] = digest
                counts = [int(x) for x in np.bincount(truth.reshape(-1).astype(np.uint8) * 2
                    + pred.reshape(-1).astype(np.uint8), minlength=4)]
                total = [a + b for a, b in zip(total, counts)]
                row = samples[str(seed)].setdefault(sid, {'sample_id': sid, 'group': groups[sid],
                    'positive_pixels': int(truth.sum()), 'pixels': int(truth.size)})
                row[variant] = metrics(counts)
            pair[variant] = metrics(total)
        pair['delta'] = {k: pair['candidate'][k] - pair['baseline'][k]
            if pair['candidate'][k] is not None and pair['baseline'][k] is not None else None
            for k in ('precision', 'recall', 'f1', 'iou')}
        paired.append(pair)
    contracts = {**VIEW_CONTRACTS, 'paired': {**VIEW_CONTRACTS['paired'],
        'meaning': 'All declared paired seeds; host recomputed binary pixel metrics'},
        'coverage': {**VIEW_CONTRACTS['coverage'], 'grouping_basis':
            f"Fraction of ground-truth label pixels equal to {encoding['positive']} in each image"}}
    payload = {'version': VERSION, 'core_version': CORE_VERSION, 'source_kind': 'imported_predictions',
        'dataset': dataset, 'encoding': encoding, 'seeds': seeds, 'assets': assets,
        'source_files': source_files, 'training_provenance': provenance,
        'verification': {'structure': 'VERIFIED', 'prediction_label_metrics': 'HOST_RECOMPUTED',
            'training_source': 'USER_DECLARED_NOT_INDEPENDENTLY_VERIFIED',
            'scientific_mechanisms': 'NOT_VERIFIED'},
        'views': views(samples, paired), 'all_samples': samples, 'view_contracts': contracts,
        'units': 'metric fractions on [0,1]; counts are pixels',
        'validation_scope': 'USER_DECLARED: ' + dataset['evaluation_scope'], 'new_gpu_starts': 0}
    identity = sha(encoded(payload))
    immutable(owned_folder(runtime, 'diagnostic-inputs', identity), 'bundle.json', encoded(payload))
    return {'bundle_id': identity, 'identity': identity, 'bundle_sha256': identity,
            'source_kind': 'imported_predictions', 'seeds': seeds, 'verification': payload['verification']}
