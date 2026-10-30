"""Standalone trusted worker for the frozen SNUNet v2 campaign."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main(job_path):
    started = time.monotonic()
    job = json.loads(Path(job_path).read_bytes())
    root, output = Path(job['profile']), Path(job['output'])
    state = module(root/'training_state.py', 'owned_training_state')
    ledger = state.GpuBudget(job['ledger'])
    reservation = ledger.attach(job['token'])
    status, report = 'FAILED', {}
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import numpy as np
        import torch
        import PIL
        from PIL import Image
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        manifest = json.loads((root/'manifest.json').read_bytes())
        def encoded(value):
            return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(encoded(manifest)).hexdigest() != job['binding_id']:
            raise ValueError('Worker profile binding changed')
        env = dict(torch=torch.__version__, numpy=np.__version__, pillow=PIL.__version__,
                   cuda_build=torch.version.cuda, python=sys.version)
        if env != manifest['environment']:
            raise ValueError('Worker package environment changed')
        for name in ('baseline.py', 'candidate.py', 'support.py', 'training_state.py', 'worker.py', 'request.json'):
            if state.file_sha(root/name) != manifest['files'][name]:
                raise ValueError('Worker source changed: '+name)
        request, protocol = manifest['request'], manifest['protocol']
        stage, seed, variant = job['stage'], job['seed'], job['variant']
        if stage not in ('preflight', 'recovery', 'formal') or seed not in (7, 11, 19) or variant not in ('baseline', 'candidate'):
            raise ValueError('Unknown host-bound training stage')
        if stage == 'formal' and job.get('pause_at') is not None:
            raise ValueError('Formal protocol cannot accept a custom step limit')
        device = torch.device('cuda')
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen CUDA device unavailable')
        support = module(root/'support.py', 'owned_support')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        baseline = module(root/'baseline.py', 'owned_baseline').SNUNet_ECAM(in_ch=3, out_ch=2)
        candidate = module(root/'candidate.py', 'owned_candidate').SNUNet_ECAM(in_ch=3, out_ch=2)
        candidate.load_state_dict(baseline.state_dict(), strict=True)

        def fingerprint(value):
            digest = hashlib.sha256()
            def add(x):
                if torch.is_tensor(x):
                    digest.update(encoded([str(x.dtype), list(x.shape)]))
                    digest.update(x.detach().cpu().contiguous().numpy().tobytes())
                elif isinstance(x, dict):
                    for key in sorted(x, key=str):
                        digest.update(encoded(key))
                        add(x[key])
                elif isinstance(x, (tuple, list)):
                    digest.update(encoded(len(x)))
                    for part in x:
                        add(part)
                else:
                    digest.update(encoded(x))
            add(value)
            return digest.hexdigest()

        initial_sha = fingerprint(baseline.state_dict())
        if fingerprint(candidate.state_dict()) != initial_sha:
            raise ValueError('Paired initial parameter/buffer mismatch')
        train_rows = request['samples']['train'] if stage == 'formal' else request['samples']['train'][:4]

        def batch(rows):
            a, b, y = [], [], []
            for row in rows:
                values = []
                for role in ('time1', 'time2', 'label'):
                    path = root/row[role]
                    if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                            or state.file_sha(path) != row['hashes'][role]):
                        raise ValueError('Frozen input changed')
                    with Image.open(path) as image:
                        if image.size != (256, 256) or image.mode != ('L' if role == 'label' else 'RGB'):
                            raise ValueError('Frozen input format changed')
                        values.append(np.array(image, copy=True))
                if not set(np.unique(values[2])).issubset({0, 255}):
                    raise ValueError('Invalid binary label')
                a.append(values[0].transpose(2, 0, 1))
                b.append(values[1].transpose(2, 0, 1))
                y.append((values[2] == 255).astype(np.int64))
            return (torch.from_numpy(np.stack(a)).float().to(device)/127.5-1,
                    torch.from_numpy(np.stack(b)).float().to(device)/127.5-1,
                    torch.from_numpy(np.stack(y)).to(device))

        a, b, _ = batch(train_rows[:4])
        # Compare the identical pool replacement before any optimizer update.
        baseline = baseline.to(device).eval()
        candidate = candidate.to(device).eval()
        with torch.no_grad():
            before = baseline(a, b)[0]
            support.install_global_pool_adapter(baseline)
            support.install_global_pool_adapter(candidate)
            after = baseline(a, b)[0]
            parity = float((before-after).abs().max().item())
            effect = float((after-candidate(a, b)[0]).abs().max().item())
        if parity != 0 or not np.isfinite(effect) or effect <= 0:
            raise ValueError('Pool adapter parity/candidate effectiveness failed')
        model = baseline if variant == 'baseline' else candidate
        del baseline, candidate, before, after, a, b
        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(model.parameters(), **{k: v for k, v in protocol['optimizer'].items() if k != 'name'})
        identity = {'binding_id': job['binding_id'], 'protocol_sha256': request['protocol_sha256'],
                    'seed': seed, 'variant': variant, 'stage': stage}
        checkpoint = state.load_checkpoint(output/'checkpoints', identity)
        step, losses, norms, prior_elapsed = 0, [], [], 0.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            step, losses, norms = checkpoint['step'], checkpoint['losses'], checkpoint['gradient_norms']
            prior_elapsed = checkpoint['elapsed_seconds']
            random.setstate(checkpoint['rng_python'])
            n = checkpoint['rng_numpy']
            np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
            torch.set_rng_state(checkpoint['rng_torch'])
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        steps_per_epoch = len(train_rows)//4
        target = (protocol['epochs']*steps_per_epoch if stage == 'formal'
                  else protocol['preflight']['steps'] if stage == 'preflight'
                  else protocol['recovery_check']['steps'])
        if (type(step) is not int or not 0 <= step <= target or len(losses) != step or len(norms) != step
                or checkpoint and (checkpoint['epoch'] != step//steps_per_epoch
                                   or checkpoint['batch_cursor'] != step % steps_per_epoch)):
            raise ValueError('Checkpoint cursor/loss history inconsistent')
        torch.cuda.reset_peak_memory_stats()
        model.train()
        expected_unused = request['expected_unused'] if variant == 'candidate' else []

        def commit():
            n = np.random.get_state()
            return state.save_checkpoint(output/'checkpoints', {'step': step,
                'epoch': step//steps_per_epoch, 'batch_cursor': step % steps_per_epoch,
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'rng_python': random.getstate(), 'rng_numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
                'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
                'losses': losses, 'gradient_norms': norms,
                'elapsed_seconds': prior_elapsed + time.monotonic()-started}, identity)

        pause = False
        for _ in range(step, target):
            if ((output/'stop').exists() or time.monotonic()-started >= reservation['reserved']-20
                    or job.get('pause_at') is not None and step >= job['pause_at']):
                pause = True
                break
            cursor = (step % steps_per_epoch)*4
            a, b, y = batch(train_rows[cursor:cursor+4])
            optimizer.zero_grad(set_to_none=True)
            loss = support.classification_loss(model(a, b)[0], y)
            if not torch.isfinite(loss).item():
                raise ValueError('Non-finite training loss')
            loss.backward()
            absent = sorted(n for n, p in model.named_parameters() if p.grad is None)
            if absent != expected_unused:
                raise ValueError('Unexpected unused gradient set')
            grad_norm = torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm()
            if not torch.isfinite(grad_norm).item() or grad_norm.item() <= 0:
                raise ValueError('Non-finite/zero gradient norm')
            optimizer.step()
            losses.append(float(loss.item()))
            norms.append(float(grad_norm.item()))
            step += 1
            if step % 25 == 0:
                commit()
                print(json.dumps({'step': step, 'target': target, 'loss': losses[-1]}), flush=True)
        receipt = commit()
        report = {'binding_id': job['binding_id'], 'identity': identity, 'step': step, 'target': target,
            'initial_state_sha256': initial_sha, 'final_state_sha256': fingerprint(model.state_dict()),
            'optimizer_sha256': fingerprint(optimizer.state_dict()),
            'rng_sha256': fingerprint({'python': random.getstate(), 'numpy': list(np.random.get_state()[1].tolist()),
                                      'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}),
            'pool_adapter_max_abs_error': parity, 'pretraining_max_abs_logit_difference': effect,
            'losses': losses, 'gradient_norms': norms, 'expected_unused': expected_unused,
            'checkpoint': receipt, 'environment': env, 'gpu': torch.cuda.get_device_name(),
            'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'test_opened': False,
            'scientific_hypothesis_supported': False, 'elapsed_seconds': prior_elapsed + time.monotonic()-started}
        if not pause and stage == 'formal':
            model.eval()
            confusion = np.zeros(4, dtype=np.int64)
            predictions = []
            with torch.no_grad():
                for start in range(0, len(request['samples']['val']), 4):
                    if time.monotonic()-started >= reservation['reserved']-15 or (output/'stop').exists():
                        pause = True
                        break
                    rows = request['samples']['val'][start:start+4]
                    a, b, y = batch(rows)
                    logits = model(a, b)[0]
                    if not torch.isfinite(logits).all().item():
                        raise ValueError('Non-finite validation logits')
                    pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    truth = y.cpu().numpy()
                    for row, p, t in zip(rows, pred, truth):
                        confusion += np.bincount((2*t+p).reshape(-1), minlength=4)
                        predictions.append({'sample_id': row['sample_id'], 'pixels': 65536,
                            'packed_bits': base64.b64encode(np.packbits(p.reshape(-1), bitorder='big').tobytes()).decode()})
            tn, fp, fn, tp = map(int, confusion)
            report['predictions'] = predictions
            report['validation'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
                'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                'iou': tp/(tp+fp+fn) if tp+fp+fn else None}
        if stage == 'preflight' and not pause:
            first, last = sum(losses[:10])/10, sum(losses[-10:])/10
            report['learning'] = {'first_10_mean': first, 'last_10_mean': last, 'passed': last < .9*first}
        status = 'PAUSED' if pause else 'COMPLETED'
    except Exception as exc:
        report = {'error': type(exc).__name__+': '+str(exc)}
        raise
    finally:
        elapsed = time.monotonic()-started
        report.update(status=status, launch_elapsed_seconds=elapsed, token=job['token'])
        state.atomic_json(output/('receipt-'+job['token']+'.json'), report)
        ledger.finish(job['token'], status, elapsed)
    return report


if __name__ == '__main__':
    main(sys.argv[1])
