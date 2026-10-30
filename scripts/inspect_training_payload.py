"""Standalone CPU check of committed v2 payloads, using the frozen training interpreter."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main(job_path):
    import torch
    job = json.loads(Path(job_path).read_bytes())
    root = Path(job['profile'])
    def encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    manifest = json.loads((root/'manifest.json').read_bytes())
    if hashlib.sha256(encoded(manifest)).hexdigest() != job['binding_id']:
        raise ValueError('CPU inspection binding changed')
    def module(name):
        path = root/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('CPU inspection source changed')
        spec = importlib.util.spec_from_file_location('owned_'+name.replace('.', '_'), path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    state = module('training_state.py')
    baseline = module('baseline.py')
    def fingerprint(value):
        result = hashlib.sha256()
        def add(item):
            if torch.is_tensor(item):
                if item.device.type != 'cpu':
                    raise ValueError('Inspection must remain on CPU')
                result.update(encoded([str(item.dtype), list(item.shape)]))
                result.update(item.contiguous().numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    result.update(encoded(key))
                    add(item[key])
            elif isinstance(item, (tuple, list)):
                result.update(encoded(len(item)))
                for part in item:
                    add(part)
            else:
                result.update(encoded(item))
        add(value)
        return result.hexdigest()
    verified = []
    for entry in job['trials']:
        report_path = Path(entry['path'])
        if state.file_sha(report_path) != entry['sha256']:
            raise ValueError('Worker receipt changed before CPU inspection')
        report = json.loads(report_path.read_bytes())
        payload = state.load_checkpoint(report_path.parent/'checkpoints', report['identity'])
        if payload is None or payload['step'] != report['step'] or payload['losses'] != report['losses']:
            raise ValueError('Final payload cursor/history differs from worker receipt')
        rejected_identities = []
        for field in ('binding_id', 'protocol_sha256'):
            changed_identity = dict(report['identity'])
            changed_identity[field] = 'changed-for-cpu-verification'
            try:
                state.load_checkpoint(report_path.parent/'checkpoints', changed_identity)
            except ValueError as error:
                if 'identity' not in str(error):
                    raise
                rejected_identities.append(field)
            else:
                raise ValueError('Checkpoint accepted a different source or protocol identity')
        if fingerprint(payload['model']) != report['final_state_sha256'] or fingerprint(payload['optimizer']) != report['optimizer_sha256']:
            raise ValueError('Actual saved tensor state differs from worker report')
        torch.manual_seed(report['identity']['seed'])
        initial = baseline.SNUNet_ECAM(in_ch=3, out_ch=2).state_dict()
        if fingerprint(initial) != report['initial_state_sha256']:
            raise ValueError('Saved initialization differs from independent seeded CPU reconstruction')
        if torch.equal(initial['conv_final.weight'], payload['model']['conv_final.weight']):
            raise ValueError('Classifier weights did not update')
        rng = {'python': payload['rng_python'], 'numpy': payload['rng_numpy'][1],
               'torch': payload['rng_torch'], 'cuda': payload['rng_cuda']}
        if fingerprint(rng) != report['rng_sha256']:
            raise ValueError('Actual checkpoint RNG differs from the reported digest')
        if torch.cuda.is_initialized():
            raise ValueError('CPU inspection unexpectedly initialized CUDA')
        verified.append({'seed': report['identity']['seed'], 'variant': report['identity']['variant'],
            'saved_state_verified': True, 'optimizer_verified': True, 'initial_state_reconstructed': True,
            'classifier_updated': True, 'rng_verified': True, 'changed_identities_rejected': rejected_identities,
            'checkpoint_sha256': report['checkpoint']['sha256']})
        del payload, initial
    print(json.dumps({'binding_id': job['binding_id'], 'trials': verified, 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
