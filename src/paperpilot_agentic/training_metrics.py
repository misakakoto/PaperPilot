"""Independent host metrics: predictions, copied labels and fixed protocol only."""
import base64
import math

from .reference_methods import encoded, sha
from .training_state import file_sha


def verify_trial(report, root, manifest, *, seed, variant):
    import numpy as np
    from PIL import Image
    expected_identity = {'binding_id': sha(encoded(manifest)),
        'protocol_sha256': manifest['request']['protocol_sha256'],
        'stage': 'formal', 'seed': seed, 'variant': variant}
    target = manifest['protocol']['epochs']*manifest['protocol']['train_images']//4
    unused = manifest['request']['expected_unused'] if variant == 'candidate' else []
    if (report.get('binding_id') != expected_identity['binding_id'] or report.get('identity') != expected_identity or report.get('status') != 'COMPLETED'
            or report.get('step') != target or report.get('target') != target
            or report.get('environment') != manifest['environment'] or report.get('test_opened') is not False
            or report.get('pool_adapter_max_abs_error') != 0 or report.get('expected_unused') != unused
            or report.get('scientific_hypothesis_supported') is not False):
        raise ValueError('Training report identity/protocol mismatch')
    for key in ('losses', 'gradient_norms'):
        values = report.get(key)
        if (not isinstance(values, list) or len(values) != target
                or any(type(v) not in (int, float) or not math.isfinite(v) or (key == 'gradient_norms' and v <= 0) for v in values)):
            raise ValueError('Incomplete/non-finite training history')
    if report['initial_state_sha256'] == report['final_state_sha256']:
        raise ValueError('Training did not update parameters/buffers')
    expected = manifest['request']['samples']['val']
    rows = report.get('predictions')
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Missing validation predictions')
    counts = np.zeros(4, dtype=np.int64)
    for row, sample in zip(rows, expected):
        if row['sample_id'] != sample['sample_id'] or row['pixels'] != 65536:
            raise ValueError('Prediction identity or dimensions mismatch')
        packed = base64.b64decode(row['packed_bits'], validate=True)
        if len(packed) != 8192:
            raise ValueError('Wrong prediction bit count')
        path = root/sample['label']
        if file_sha(path) != sample['hashes']['label']:
            raise ValueError('Frozen validation label changed')
        with Image.open(path) as image:
            if image.mode != 'L' or image.size != (256, 256):
                raise ValueError('Validation label shape changed')
            truth = np.array(image).reshape(-1)
        if not set(np.unique(truth)).issubset({0, 255}):
            raise ValueError('Invalid validation label values')
        pred = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder='big')
        counts += np.bincount((truth == 255).astype(np.uint8)*2+pred, minlength=4)
    tn, fp, fn, tp = map(int, counts)
    metrics = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
        'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'iou': tp/(tp+fp+fn) if tp+fp+fn else None}
    for key, expected_value in metrics.items():
        value = report.get('validation', {}).get(key)
        valid = (value is None if expected_value is None else type(value) is int and value == expected_value
                 if key in ('tn', 'fp', 'fn', 'tp') else
                 type(value) in (int, float) and math.isclose(value, expected_value, rel_tol=0, abs_tol=1e-12))
        if not valid:
            raise ValueError('Worker metric disagrees with independent counts: '+key)
    return metrics


def paired_summary(trials):
    """Always include every seed/variant, including failures and unfinished jobs."""
    rows = []
    for seed in (7, 11, 19):
        row = {'seed': seed}
        for variant in ('baseline', 'candidate'):
            row[variant] = trials.get(f'{seed}-{variant}', {'status': 'NOT_RUN'})
        b, c = row['baseline'], row['candidate']
        if b['status'] == c['status'] == 'VERIFIED':
            if b['initial_state_sha256'] != c['initial_state_sha256']:
                raise ValueError('Paired baseline/candidate initial states differ')
            row['delta'] = {k: (c['metrics'][k]-b['metrics'][k]
                if c['metrics'][k] is not None and b['metrics'][k] is not None else None)
                for k in ('precision', 'recall', 'f1', 'iou')}
        else:
            row['delta'] = None
        rows.append(row)
    return rows
