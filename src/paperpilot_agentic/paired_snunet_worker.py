"""Trusted paired diagnostic worker. Imports only host-reviewed, hash-bound sources.

Two frozen train/val pairs and 1-4 steps per variant/seed: no benchmark, checkpoint
selection, significance test or autonomous arbitrary-Python execution.
"""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def classification_loss(logits, target):
    """Same pixelwise CE, using the deterministic 2-D classification kernel.

The selected CUDA/PyTorch build rejects NLL's spatial kernel in strict mode.
Flattening NHW into samples preserves the unweighted pixel mean and gradients.
"""
    import torch
    if logits.ndim != 4 or tuple(target.shape) != (logits.shape[0], *logits.shape[2:]):
        raise ValueError('Expected NCHW logits and matching NHW labels')
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    return torch.nn.functional.cross_entropy(flat_logits, target.reshape(-1))


def global_max_pool(value):
    """Global (1x1) pool only; select first flattened maximum, including ties."""
    if value.ndim != 4:
        raise ValueError('Global pool expects NCHW')
    return value.flatten(2).max(dim=2).values.unsqueeze(-1).unsqueeze(-1)


def install_global_pool_adapter(model):
    import torch
    class GlobalMax(torch.nn.Module):
        def forward(self, value):
            return global_max_pool(value)
    for name in ('ca', 'ca1'):
        attention = getattr(model, name)
        if type(attention.max_pool) is not torch.nn.AdaptiveMaxPool2d or attention.max_pool.output_size != 1:
            raise ValueError('Only the pinned global 1x1 attention pool may be adapted')
        attention.max_pool = GlobalMax()


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    request = json.loads(Path('request.json').read_text(encoding='utf-8'))
    seeds, steps = request['seeds'], request['steps']
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or type(steps) is not int or not 1 <= steps <= 4):
        raise ValueError('Invalid bounded paired request')
    device = torch.device(request['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')

    def module(name):
        path = Path(name+'.py')
        if digest(path.read_bytes()) != request['source_hashes'][name]:
            raise ValueError('Reviewed source changed: '+name)
        spec = importlib.util.spec_from_file_location('approved_'+name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    helper, baseline_module, candidate_module = module('helper'), module('baseline'), module('candidate')

    def load(row):
        arrays = []
        for role in ('time1', 'time2', 'label'):
            path = Path(row[role])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Input path escapes copied workspace')
            raw = path.read_bytes()
            if digest(raw) != row['hashes'][role]:
                raise ValueError('Frozen input changed')
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode != ('L' if role == 'label' else 'RGB') or image.size != (256, 256):
                    raise ValueError('Wrong SYSU input shape or mode')
                arrays.append(np.array(image, copy=True))
        if not set(np.unique(arrays[2])).issubset({0, 255}):
            raise ValueError('Expected binary 0/255 labels')
        a, b = [torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)/127.5-1
                for x in arrays[:2]]
        label = torch.from_numpy((arrays[2] == 255).astype(np.int64)).unsqueeze(0).to(device)
        return a, b, label
    train = [load(r) for r in request['samples']['train']]
    val = [load(r) for r in request['samples']['val']]
    if len(train) != 2 or len(val) != 2:
        raise ValueError('Exactly two frozen samples per split required')

    def state_digest(model):
        result = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()

    def evaluate(model):
        model.eval()
        confusion, predictions = np.zeros(4, dtype=np.int64), []
        with torch.no_grad():
            for row, (a, b, target) in zip(request['samples']['val'], val):
                output = helper.forward_logits(model, a, b)
                pred = output.argmax(1).cpu().numpy().astype(np.uint8).reshape(-1)
                truth = target.cpu().numpy().reshape(-1)
                confusion += np.bincount(truth*2+pred, minlength=4)
                predictions.append({'sample_id': row['sample_id'], 'pixels': int(pred.size),
                    'packed_bits': base64.b64encode(np.packbits(pred, bitorder='big').tobytes()).decode()})
        return {'validation': helper.scores(*[int(x) for x in confusion]), 'predictions': predictions}

    def fit(model, variant, seed, initial_hash):
        torch.manual_seed(seed)  # Same RNG reset, data order and optimizer state per pair.
        expected_unused = request['expected_unused'] if variant == 'candidate' else []
        before = model.conv_final.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0, weight_decay=0)
        losses, norms, missing = [], [], []
        started = time.monotonic()
        model.train()
        for step in range(steps):
            a, b, label = train[step % len(train)]
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(helper.forward_logits(model, a, b), label)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward()
            absent = sorted(name for name, p in model.named_parameters() if p.grad is None)
            if absent != expected_unused:
                raise ValueError('Unexpected missing gradients: '+str(absent))
            grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in grads):
                raise ValueError('Non-finite gradients')
            norm = torch.stack([g.norm() for g in grads]).norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Inactive training gradients')
            optimizer.step()
            losses.append(float(loss.detach().item()))
            norms.append(float(norm.item()))
            missing.append(absent)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            print(json.dumps({'event': 'optimizer_step_completed', 'seed': seed,
                              'variant': variant, 'step': step+1}), flush=True)
        updated = not torch.equal(before, model.conv_final.weight.detach())
        if not updated:
            raise ValueError('Classifier weights did not update')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        return {'initial_state_sha256': initial_hash, 'steps_completed': steps,
            'train_order': [request['samples']['train'][i % 2]['sample_id'] for i in range(steps)],
            'losses': losses, 'gradient_norms': norms, 'missing_gradients': missing,
            'weight_updated': updated, 'training_seconds': elapsed, **evaluate(model)}

    trials = []
    for seed in seeds:
        torch.manual_seed(seed)
        baseline = baseline_module.SNUNet_ECAM(in_ch=3, out_ch=2)
        candidate = candidate_module.SNUNet_ECAM(in_ch=3, out_ch=2)
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        baseline_hash, candidate_hash = state_digest(baseline), state_digest(candidate)
        if baseline_hash != candidate_hash:
            raise ValueError('Initial weights/buffers are not paired')
        baseline, candidate = baseline.to(device), candidate.to(device)
        baseline.eval()
        candidate.eval()
        with torch.no_grad():
            a, b, _ = val[0]
            before_pool_adapter = helper.forward_logits(baseline, a, b)
            install_global_pool_adapter(baseline)
            install_global_pool_adapter(candidate)
            base_logits = helper.forward_logits(baseline, a, b)
            pool_parity = float((before_pool_adapter-base_logits).abs().max().item())
            if pool_parity != 0:
                raise ValueError('Global pool adapter changed upstream logits')
            candidate_logits = helper.forward_logits(candidate, a, b)
            effect = float((base_logits-candidate_logits).abs().max().item())
        if effect <= 0:
            raise ValueError('Candidate has no observable pretraining effect')
        trial = {'seed': seed, 'pretraining_max_abs_logit_difference': effect,
            'pool_adapter_max_abs_error': pool_parity,
            'baseline': fit(baseline, 'baseline', seed, baseline_hash),
            'candidate': fit(candidate, 'candidate', seed, candidate_hash)}
        trials.append(trial)
        print(json.dumps({'seed': seed, 'status': 'PAIR_COMPLETED',
                          'baseline': trial['baseline']['validation'],
                          'candidate': trial['candidate']['validation']}), flush=True)
        del baseline, candidate
    report = {'scope': 'PAIRED_SNUNET_DIAGNOSTIC_ONLY', 'device': str(device),
        'source_hashes': request['source_hashes'], 'protocol_sha256': request['protocol_sha256'],
        'trials': trials, 'test_opened': False, 'scientific_hypothesis_supported': False,
        'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
        'limitations': ['Fixed order, two train/val samples; no significance or performance claim',
                        'Timing is diagnostic, not a speed benchmark']}
    with Path('metrics.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
