import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from paperpilot_agentic.experiment_import import register_experiment, VERSION
from paperpilot_agentic.research_diagnostics import load_bundle


def pack(tmp_path, seeds=(43,)):
    root = tmp_path / 'input'
    root.mkdir()
    truth = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    Image.fromarray(truth).save(root / 'label.png')
    Image.fromarray(np.zeros_like(truth)).save(root / 'baseline.png')
    Image.fromarray(truth).save(root / 'candidate.png')
    value = {'version': VERSION, 'task': 'binary_change_detection',
        'dataset': {'name': 'synthetic', 'version': '1', 'split': 'validation', 'evaluation_scope': 'synthetic'},
        'encoding': {'negative': 0, 'positive': 255},
        'samples': [{'sample_id': 'b', 'label': 'label.png'}, {'sample_id': 'a', 'label': 'label.png'}],
        'trials': [{'seed': s, 'variant': v, 'training_provenance': {'code': 'unverified-user-claim'},
            'predictions': [{'sample_id': sid, 'path': v + '.png'} for sid in ('a', 'b')]}
            for s in seeds for v in ('baseline', 'candidate')]}
    path = root / 'manifest.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return SimpleNamespace(workspace=tmp_path / 'host'), path, value


@pytest.mark.parametrize('seeds', [(43,), (2, 43, 103, 104)])
def test_import_arbitrary_seeds_alignment_metrics_and_idempotency(tmp_path, seeds):
    runtime, path, value = pack(tmp_path, seeds)
    binding = register_experiment(runtime, path)
    bundle = load_bundle(runtime, binding)
    assert binding['seeds'] == list(seeds)
    assert bundle['verification']['training_source'].startswith('USER_DECLARED')
    for pair in bundle['views']['paired']:
        assert pair['baseline']['tp'] == 0 and pair['baseline']['fn'] == 4
        assert pair['candidate']['tp'] == 4 and pair['candidate']['tn'] == 4
        assert pair['candidate']['f1'] == 1 and pair['baseline']['f1'] == 0
        assert pair['delta']['f1'] == 1
    assert bundle['new_gpu_starts'] == 0
    value['samples'].reverse()
    for t in value['trials']:
        t['predictions'].reverse()
    value['trials'].reverse()
    path.write_text(json.dumps(value), encoding='utf-8')
    assert register_experiment(runtime, path) == binding
    (path.parent / 'label.png').unlink()
    assert load_bundle(runtime, binding) == bundle  # source original no longer authoritative


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'shape', 'encoding', 'escape', 'seed', 'mode'])
def test_import_rejects_invalid_inputs(tmp_path, bad):
    runtime, path, value = pack(tmp_path)
    if bad == 'duplicate':
        value['samples'].append(value['samples'][0])
    elif bad == 'missing':
        value['trials'][0]['predictions'].pop()
    elif bad == 'shape':
        Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(path.parent / 'baseline.png')
    elif bad == 'encoding':
        Image.fromarray(np.array([[127]], dtype=np.uint8)).save(path.parent / 'baseline.png')
    elif bad == 'escape':
        value['samples'][0]['label'] = '../escape.png'
    elif bad == 'seed':
        value['trials'].pop()
    else:
        Image.new('RGB', (2, 2)).save(path.parent / 'baseline.png')
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError)):
        register_experiment(runtime, path)


def test_copied_asset_tampering_is_rejected(tmp_path):
    runtime, path, _ = pack(tmp_path)
    binding = register_experiment(runtime, path)
    bundle = load_bundle(runtime, binding)
    asset = next(iter(bundle['assets']))
    from pathlib import Path
    Path(asset).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_bundle(runtime, binding)


def test_empty_positive_groups_are_undefined_and_observed_zero_is_not_missing(tmp_path):
    runtime, path, _ = pack(tmp_path)
    for name in ('label', 'baseline', 'candidate'):
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path.parent/(name+'.png'))
    bundle = load_bundle(runtime, register_experiment(runtime, path))
    assert bundle['views']['paired'][0]['baseline']['f1'] is None
    assert bundle['views']['coverage'][0]['sample_count'] == 2
    assert all(row['sample_count'] == 0 for row in bundle['views']['coverage'][1:])
    assert not bundle['views']['samples'][0]['rows']


def test_binary_01_encoding_is_explicit_and_not_thresholded(tmp_path):
    runtime, path, value = pack(tmp_path)
    value['encoding']['positive'] = 1
    for name in ('label', 'baseline', 'candidate'):
        values = np.asarray(Image.open(path.parent/(name+'.png'))) // 255
        Image.fromarray(values.astype(np.uint8)).save(path.parent/(name+'.png'))
    path.write_text(json.dumps(value), encoding='utf-8')
    bundle = load_bundle(runtime, register_experiment(runtime, path))
    assert bundle['views']['paired'][0]['candidate']['f1'] == 1
