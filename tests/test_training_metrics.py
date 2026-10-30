import base64
import copy

from PIL import Image
import pytest

from paperpilot_agentic.reference_methods import encoded, sha
from paperpilot_agentic.training_metrics import paired_summary, verify_trial
from paperpilot_agentic.training_profiles import protocol, select_samples


def fixture(tmp_path):
    path = tmp_path/'label.png'
    Image.new('L', (256, 256), 255).save(path)
    p = protocol()
    # A short synthetic host fixture tests the verifier; not a training score.
    p.update(train_images=4, epochs=1)
    manifest = {'protocol': p, 'environment': {'test': True},
        'request': {'protocol_sha256': 'protocol', 'expected_unused': [],
                    'samples': {'val': [{'sample_id': 'one', 'label': 'label.png',
                                         'hashes': {'label': sha(path.read_bytes())}}]}}}
    binding = sha(encoded(manifest))
    report = {'binding_id': binding, 'identity': {'binding_id': binding, 'protocol_sha256': 'protocol',
        'stage': 'formal', 'seed': 7, 'variant': 'baseline'}, 'status': 'COMPLETED', 'step': 1, 'target': 1,
        'environment': manifest['environment'], 'test_opened': False, 'pool_adapter_max_abs_error': 0,
        'expected_unused': [], 'scientific_hypothesis_supported': False, 'losses': [.2], 'gradient_norms': [1.],
        'initial_state_sha256': 'first', 'final_state_sha256': 'last',
        'predictions': [{'sample_id': 'one', 'pixels': 65536,
                         'packed_bits': base64.b64encode(bytes([255])*8192).decode()}],
        'validation': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 65536, 'precision': 1., 'recall': 1., 'f1': 1., 'iou': 1.}}
    return manifest, report


def test_host_independently_recomputes_and_rejects_forgery(tmp_path):
    manifest, report = fixture(tmp_path)
    assert verify_trial(report, tmp_path, manifest, seed=7, variant='baseline')['tp'] == 65536
    fake = copy.deepcopy(report)
    fake['validation']['f1'] = .999
    with pytest.raises(ValueError, match='independent'):
        verify_trial(fake, tmp_path, manifest, seed=7, variant='baseline')
    fake = copy.deepcopy(report)
    fake['predictions'][0]['sample_id'] = 'another'
    with pytest.raises(ValueError, match='identity'):
        verify_trial(fake, tmp_path, manifest, seed=7, variant='baseline')
    Image.new('L', (256, 256), 0).save(tmp_path/'label.png')
    with pytest.raises(ValueError, match='label changed'):
        verify_trial(report, tmp_path, manifest, seed=7, variant='baseline')


def test_summary_preserves_unfinished_seeds_and_negative_results():
    baseline = {'status': 'VERIFIED', 'initial_state_sha256': 'same', 'metrics': dict.fromkeys(['precision', 'recall', 'f1', 'iou'], .8)}
    candidate = {**baseline, 'metrics': dict.fromkeys(['precision', 'recall', 'f1', 'iou'], .7)}
    rows = paired_summary({'7-baseline': baseline, '7-candidate': candidate})
    assert [r['seed'] for r in rows] == [7, 11, 19]
    assert rows[0]['delta']['f1'] == pytest.approx(-.1)
    assert rows[1]['candidate']['status'] == 'NOT_RUN'
    with pytest.raises(ValueError, match='initial'):
        paired_summary({'7-baseline': baseline, '7-candidate': {**candidate, 'initial_state_sha256': 'other'}})


def test_hash_selection_does_not_read_labels_or_test_and_excludes_old(tmp_path):
    for split in ('train', 'val'):
        (tmp_path/(split+'.txt')).write_text('\n'.join(
            f'{split}/time1/{split}-{i}.png {split}/time2/{split}-{i}.png {split}/label/{split}-{i}.png'
            for i in range(8)))
    selected, _ = select_samples(tmp_path, {'train-0.png', 'val-0.png'}, {'train': 3, 'val': 2})
    again, _ = select_samples(tmp_path, {'train-0.png', 'val-0.png'}, {'train': 3, 'val': 2})
    assert selected == again
    assert 'train-0.png' not in selected['train'] and 'val-0.png' not in selected['val']
    # No image/label/test directories exist: selection used only official lists.
