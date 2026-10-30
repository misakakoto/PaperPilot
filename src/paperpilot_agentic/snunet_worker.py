"""Standalone trusted diagnostic worker, copied beside a hash-pinned upstream model.

Not a paper reproduction: random initialization, two fixed train/val pairs, no
augmentation or test access. No model-generated Python is imported here.
"""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path

SOURCE_SHA = 'c14f3da927c21ea5503ac333a42afd3e5a9ed05efece962b637eb10eb15186c5'


def forward_logits(model, first, second):
    """The only adapter: preserve upstream weights/math; unwrap its one-item tuple."""
    import torch
    if (first.ndim != 4 or first.shape != second.shape or first.shape[1] != 3
            or any(x % 16 for x in first.shape[2:])
            or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
        raise ValueError('Expected aligned finite NCHW RGB tensors, spatial sizes divisible by 16')
    output = model(first, second)
    if (not isinstance(output, tuple) or len(output) != 1
            or tuple(output[0].shape) != (first.shape[0], 2, *first.shape[2:])
            or not torch.isfinite(output[0]).all()):
        raise ValueError('Invalid upstream two-class logits')
    return output[0]


def scores(tn, fp, fn, tp):
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else None}


def main():
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    request = json.loads(Path('request.json').read_text(encoding='utf-8'))
    seed, steps = request['seed'], request['steps']
    if type(seed) is not int or type(steps) is not int or not 1 <= steps <= 4:
        raise ValueError('Invalid bounded smoke request')
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CPU is explicit in request; no silent device fallback.
    device = torch.device(request['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    source = Path('upstream.py')
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Upstream source fingerprint changed')
    spec = importlib.util.spec_from_file_location('approved_snunet', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model = upstream.SNUNet_ECAM(in_ch=3, out_ch=2).to(device)

    def load(row):
        arrays = []
        for role in ('time1', 'time2', 'label'):
            path = Path(row[role])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Input must be inside the copied workspace')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['hashes'][role]:
                raise ValueError('Frozen sample changed')
            with Image.open(io.BytesIO(raw)) as image:
                expected_mode = 'L' if role == 'label' else 'RGB'
                if image.mode != expected_mode or image.size != (256, 256):
                    raise ValueError('Unexpected SYSU image shape or mode')
                arrays.append(np.array(image, copy=True))
        if not set(np.unique(arrays[2])).issubset({0, 255}):
            raise ValueError('Expected binary 0/255 labels; no implicit remapping')
        a, b = [torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 127.5 - 1
                for x in arrays[:2]]
        target = torch.from_numpy((arrays[2] == 255).astype(np.int64)).unsqueeze(0).to(device)
        return a, b, target

    train = [load(row) for row in request['samples']['train']]
    val = [load(row) for row in request['samples']['val']]
    model.eval()
    checks = []
    with torch.no_grad():
        # Direct upstream call is the reference, not the adapter calling itself.
        for a, b, _ in val:
            reference = model(a, b)[0]
            adapted = forward_logits(model, a, b)
            checks.append(float((reference - adapted).abs().max().item()))
        a, b, _ = val[0]
        sensitivity = float((model(a, b)[0] - model(a, -b)[0]).abs().max().item())
    if any(error != 0 for error in checks) or sensitivity <= 0:
        raise ValueError('Adapter parity or changed-input sensitivity failed')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = model.conv_final.weight.detach().clone()
    losses, gradient_norms = [], []
    model.train()
    for step in range(steps):
        a, b, target = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(forward_logits(model, a, b), target)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        norm = torch.stack(norms).norm()
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Missing or non-finite gradients')
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradient_norms.append(float(norm.item()))
    updated = not torch.equal(before, model.conv_final.weight.detach())
    if not updated:
        raise ValueError('Optimizer did not change model weights')
    predictions, confusion = [], np.zeros(4, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for row, (a, b, target) in zip(request['samples']['val'], val):
            pred = forward_logits(model, a, b).argmax(1).cpu().numpy().astype(np.uint8).reshape(-1)
            truth = target.cpu().numpy().reshape(-1)
            confusion += np.bincount(truth * 2 + pred, minlength=4)
            predictions.append({'sample_id': row['sample_id'], 'pixels': int(pred.size),
                'packed_bits': base64.b64encode(np.packbits(pred, bitorder='big').tobytes()).decode()})
    tn, fp, fn, tp = [int(x) for x in confusion]
    report = {'scope': 'UPSTREAM_ADAPTER_AND_SYSU_SMOKE_ONLY', 'source_sha256': SOURCE_SHA,
        'seed': seed, 'steps_completed': steps, 'device': str(device),
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'numpy': np.__version__, 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None},
        'parity_max_abs_errors': checks, 'changed_input_max_abs_difference': sensitivity,
        'weight_updated': updated, 'losses': losses, 'gradient_norms': gradient_norms,
        'validation': scores(tn, fp, fn, tp), 'predictions': predictions,
        'preprocessing': 'RGB / 127.5 - 1; label 255 is change; no resizing/augmentation',
        'test_opened': False, 'paper_accuracy_reproduced': False}
    with Path('metrics.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != 'predictions'}), flush=True)


if __name__ == '__main__':
    main()
