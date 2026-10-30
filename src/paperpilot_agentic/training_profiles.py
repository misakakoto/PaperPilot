"""Host-only v2 protocol preparation; no model-selected commands or training paths."""
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from . import paired_methods as paired
from . import reference_methods as reference
from .method_drafts import build_patch, load_draft
from .training_state import file_sha

encoded, sha = reference.encoded, reference.sha


def protocol():
    return {'version': 2, 'recipe': 'snunet_attention_bypass_small_scale_v2',
        'seeds': [7, 11, 19], 'variants': ['baseline', 'candidate'],
        'train_images': 512, 'val_images': 256, 'epochs': 20, 'batch_size': 4,
        'image_size': [256, 256], 'dtype': 'FP32', 'augmentation': False,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'foreach': False},
        'loss': 'unweighted pixel mean CE via deterministic 2-D kernel',
        'preprocessing': 'RGB / 127.5 - 1; label 255 is change; no resize',
        'selection': 'sha256(paperpilot-sysu-v2/split/sample_id), ascending; exclude old smoke inputs',
        'training_order': 'fixed selected order every epoch, identical across paired variants',
        'initialization': 'copy every baseline parameter and buffer before training; reset RNG per variant',
        'checkpoint_selection': 'fixed final epoch; no best-val selection',
        'checkpoint_interval': 25, 'gpu_cap_seconds': 28800,
        'preflight': {'steps': 100, 'images': 4, 'max_seconds': 600,
                      'learning': 'mean last 10 losses < 0.9 * mean first 10 losses'},
        'recovery_check': {'steps': 6, 'pause_step': 3},
        'execution_mode': 'reviewed host-local code; no hard filesystem/network isolation',
        'scientific_acceptance': 'descriptive paired results only; no significance or innovation claim'}


def select_samples(dataset, excluded, counts=None):
    """Choose IDs using split/name alone, before reading any labels."""
    dataset = Path(dataset).resolve(strict=True)
    counts = counts or {'train': 512, 'val': 256}
    selected, splits, seen = {}, {}, set()
    for split in ('train', 'val'):
        listing = dataset/(split+'.txt')
        if listing.is_symlink() or not listing.resolve().is_relative_to(dataset):
            raise ValueError('Unsafe official split list')
        raw, names = listing.read_bytes(), []
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            paths = [PurePosixPath(x.replace('\\', '/')) for x in line.split()]
            if (len(paths) != 3 or len({p.name for p in paths}) != 1
                    or any('..' in p.parts or p.parts[-3:-1] != (split, role)
                           for p, role in zip(paths, ('time1', 'time2', 'label')))):
                raise ValueError('Invalid official split row')
            name = paths[0].name
            # Official lists may carry the author's machine prefix. Never open
            # those paths: use only the validated split/role/basename under dataset.
            if name in seen:
                raise ValueError('Duplicate/overlapping train and val IDs')
            seen.add(name)
            if name not in excluded:
                names.append(name)
        if len(names) < counts[split]:
            raise ValueError('Not enough independent samples in '+split)
        names.sort(key=lambda n: (sha(f'paperpilot-sysu-v2/{split}/{n}'.encode()), n))
        selected[split] = names[:counts[split]]
        splits[split] = {'list_sha256': sha(raw), 'eligible': len(names), 'selected': counts[split]}
    return selected, splits


def environment(python):
    """Read package identities without allocating CUDA tensors or training."""
    code = ('import json,torch,numpy,PIL,sys; '
            'print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,pillow=PIL.__version__,'
            'cuda_build=torch.version.cuda,python=sys.version)))')
    result = subprocess.run([str(python), '-I', '-c', code], capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def prepare_training(runtime, design_run_id, *, dataset):
    material = load_draft(runtime, design_run_id)
    review_sha = paired._review(runtime, design_run_id, material)
    baseline, contract = material['baseline'], material['contract']
    built = build_patch({k: contract[k] for k in ('target', 'replacement', 'review_notes')}, baseline)
    source, candidate = baseline['source'], built['candidate_source']
    if sha(candidate.encode()) != contract['candidate_sha256']:
        raise ValueError('Reviewed candidate reconstruction changed')
    paired.check_ablation(source, candidate)
    original, old = reference.load_profile(runtime.workspace, baseline['binding_id'])
    # Freeze the exclusion set across all previously prepared smoke profiles.
    excluded = set()
    for path in (runtime.workspace/'method-profiles').glob('*/manifest.json'):
        entry = json.loads(path.read_bytes())
        excluded.update(s['sample_id'] for rows in entry['request']['samples'].values() for s in rows)
    selected, splits = select_samples(dataset, excluded)
    dataset = Path(dataset).resolve(strict=True)
    samples, files = {}, {'baseline.py': source.encode(), 'candidate.py': candidate.encode(),
        'LICENSE': (original/'LICENSE').read_bytes(),
        'support.py': Path(__file__).with_name('paired_snunet_worker.py').read_bytes(),
        'worker.py': Path(__file__).with_name('training_worker.py').read_bytes(),
        'training_state.py': Path(__file__).with_name('training_state.py').read_bytes()}
    from PIL import Image
    import io
    for split, names in selected.items():
        samples[split] = []
        for name in names:
            item = {'sample_id': name, 'hashes': {}}
            for role in ('time1', 'time2', 'label'):
                path = dataset/split/role/name
                if path.is_symlink() or not path.resolve().is_relative_to(dataset) or path.stat().st_size > 2_000_000:
                    raise ValueError('Unsafe SYSU sample')
                raw = path.read_bytes()
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (256, 256) or image.mode != ('L' if role == 'label' else 'RGB'):
                        raise ValueError('Unexpected SYSU image shape/mode')
                    if role == 'label' and not set(image.tobytes()).issubset({0, 255}):
                        raise ValueError('Expected binary SYSU labels')
                target = f'inputs/{split}/{role}/{name}'
                files[target], item[role], item['hashes'][role] = raw, target, sha(raw)
            samples[split].append(item)
    request = {'samples': samples, 'device': 'cuda', 'protocol_sha256': sha(encoded(protocol())),
               'expected_unused': paired.EXPECTED_UNUSED}
    files['request.json'] = encoded(request)
    manifest = {'version': 2, 'design_run_id': design_run_id, 'review_sha256': review_sha,
        'baseline_binding_id': baseline['binding_id'], 'draft_artifacts': material['draft_artifacts'],
        'python': old['python'], 'python_sha256': old['python_sha256'],
        'environment': environment(old['python']), 'protocol': protocol(), 'request': request,
        'splits': splits, 'excluded_ids': sorted(excluded),
        'files': {n: sha(raw) for n, raw in files.items()}}
    binding = sha(encoded(manifest))
    root = runtime.workspace/'paired-method-profiles'/binding
    for name, raw in {**files, 'manifest.json': encoded(manifest)}.items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe training profile path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('Refusing to overwrite frozen training assets')
        else:
            with path.open('xb') as stream:
                stream.write(raw)
    return binding


def load_training(runtime, binding, *, check_environment=True):
    if not isinstance(binding, str) or not re.fullmatch('[a-f0-9]{64}', binding):
        raise ValueError('Invalid training binding ID')
    root = runtime.workspace/'paired-method-profiles'/binding
    if not root.resolve().is_relative_to(runtime.workspace.resolve()):
        raise ValueError('Training profile escapes workspace')
    manifest = json.loads((root/'manifest.json').read_bytes())
    if sha(encoded(manifest)) != binding or manifest['version'] != 2 or manifest['protocol'] != protocol():
        raise ValueError('Training protocol changed; prepare and approve a new binding')
    for name, digest in manifest['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or file_sha(path) != digest:
            raise ValueError('Training source/data asset changed: '+name)
    material = load_draft(runtime, manifest['design_run_id'])
    if (paired._review(runtime, manifest['design_run_id'], material) != manifest['review_sha256']
            or material['draft_artifacts'] != manifest['draft_artifacts']
            or material['baseline']['binding_id'] != manifest['baseline_binding_id']
            or manifest['files']['baseline.py'] != reference.SOURCE_SHA
            or manifest['files']['candidate.py'] != material['contract']['candidate_sha256']
            or manifest['files']['LICENSE'] != reference.LICENSE_SHA
            or file_sha(manifest['python']) != manifest['python_sha256']):
        raise ValueError('Training authority or interpreter changed')
    for local, frozen in (('training_worker.py', 'worker.py'), ('training_state.py', 'training_state.py'),
                          ('paired_snunet_worker.py', 'support.py')):
        if file_sha(Path(__file__).with_name(local)) != manifest['files'][frozen]:
            raise ValueError('Training implementation changed; old checkpoint must not resume')
    request = manifest['request']
    if (json.loads((root/'request.json').read_bytes()) != request
            or request['protocol_sha256'] != sha(encoded(protocol())) or request['device'] != 'cuda'
            or request['expected_unused'] != paired.EXPECTED_UNUSED):
        raise ValueError('Training request changed')
    seen = set(manifest['excluded_ids'])
    expected_files = {'baseline.py', 'candidate.py', 'LICENSE', 'support.py', 'worker.py', 'training_state.py', 'request.json'}
    for split in ('train', 'val'):
        rows = request['samples'][split]
        if len(rows) != protocol()[split+'_images']:
            raise ValueError('Wrong fixed split size')
        for row in rows:
            if row['sample_id'] in seen:
                raise ValueError('Training/validation overlap or old diagnostic reused')
            seen.add(row['sample_id'])
            for role in ('time1', 'time2', 'label'):
                path = f"inputs/{split}/{role}/{row['sample_id']}"
                if row[role] != path or manifest['files'].get(path) != row['hashes'][role]:
                    raise ValueError('Unbound training sample')
                expected_files.add(path)
    if set(manifest['files']) != expected_files:
        raise ValueError('Unexpected training assets')
    if check_environment and environment(manifest['python']) != manifest['environment']:
        raise ValueError('Training package environment changed')
    paired.check_ablation((root/'baseline.py').read_text(encoding='utf-8'), (root/'candidate.py').read_text(encoding='utf-8'))
    return root, manifest


def describe_training(runtime, binding):
    _, manifest = load_training(runtime, binding)
    return {'binding_id': binding, 'design_run_id': manifest['design_run_id'],
        'review_sha256': manifest['review_sha256'], 'protocol': manifest['protocol'],
        'data_scope': {'dataset': 'SYSU-CD', 'train_images': 512, 'val_images': 256, 'test_images': 0},
        'limits': {'gpu_cap_seconds': 28800, 'checkpoint_interval': 25},
        'source_hashes': {k: manifest['files'][k] for k in ('baseline.py', 'candidate.py')},
        'requires_approval': True, 'scientific_hypothesis_supported': False,
        'network_isolated': False, 'filesystem_isolated': False,
        'output_location': str(runtime.workspace/'method-training'/'runs'/'<run-id>')}
