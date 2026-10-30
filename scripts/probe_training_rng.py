"""Standalone RNG/cursor continuation probe; no network or optimizer training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time


def main(job_path):
    started = time.monotonic()
    job = json.loads(Path(job_path).read_bytes())
    root, output = Path(job['profile']), Path(job['output'])
    spec = importlib.util.spec_from_file_location('owned_state', root/'training_state.py')
    state = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state)
    ledger = state.GpuBudget(job['ledger'])
    ledger.attach(job['token'])
    status = 'FAILED'
    try:
        import numpy as np
        import torch
        manifest = json.loads((root/'manifest.json').read_bytes())
        canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if (hashlib.sha256(canonical).hexdigest() != job['binding_id']
                or state.file_sha(__file__) != job['source_sha256']):
            raise ValueError('RNG probe binding/source changed')
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        def draw():
            return {'python': [random.random() for _ in range(17)], 'numpy': np.random.random(17).tolist(),
                    'torch_cpu': torch.rand(17).tolist(), 'torch_cuda': torch.rand(17, device='cuda').cpu().tolist()}
        for _ in range(3):
            draw()  # Every RNG actually advances before the checkpoint.
        n = np.random.get_state()
        identity = {'binding_id': job['binding_id'], 'probe': 'RNG_CURSOR_CONTINUATION_V1',
                    'source_sha256': job['source_sha256']}
        state.save_checkpoint(output/'checkpoints', {'step': 3, 'epoch': 0, 'batch_cursor': 3,
            'python': random.getstate(), 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}, identity)
        expected = [draw() for _ in range(3)]
        for _ in range(11):
            draw()  # Perturb all streams, so omitting restoration would fail.
        payload = state.load_checkpoint(output/'checkpoints', identity)
        random.setstate(payload['python'])
        n = payload['numpy']
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(payload['torch_cpu'])
        torch.cuda.set_rng_state_all(payload['torch_cuda'])
        actual = [draw() for _ in range(3)]
        if actual != expected:
            raise ValueError('Random stream continuation differs from uninterrupted draws')
        samples = manifest['request']['samples']['train']
        cursor = (payload['epoch']*(len(samples)//4)+payload['batch_cursor'])*4
        recovered_ids = [r['sample_id'] for r in samples[cursor:cursor+12]]
        expected_ids = [r['sample_id'] for r in samples[12:24]]
        if recovered_ids != expected_ids or cursor != 12:
            raise ValueError('Nonzero batch cursor continuation failed')
        report = {'binding_id': job['binding_id'], 'source_sha256': job['source_sha256'],
            'scope': 'RNG_STATE_AND_NONZERO_CURSOR_ONLY; no model/optimizer training',
            'streams_exact': ['Python', 'NumPy', 'Torch CPU', 'Torch CUDA'],
            'draws_before_checkpoint': 3, 'perturbation_draws': 11, 'continuation_draws': 3,
            'batch_cursor': 3, 'next_12_train_ids_match': True, 'test_opened': False,
            'new_llm_calls': 0, 'gpu': torch.cuda.get_device_name(), 'token': job['token']}
        state.atomic_json(output/'verification.json', report)
        status = 'COMPLETED'
    finally:
        ledger.finish(job['token'], status, time.monotonic()-started)


if __name__ == '__main__':
    main(sys.argv[1])
