"""Export owned, verified predictions as a portable PNG example. No training/model calls."""
import argparse
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.research_assets import immutable, owned_folder
from paperpilot_agentic.research_diagnostics import freeze_experiment, load_bundle
from paperpilot_agentic.experiment_import import VERSION


def export(runtime, parent, candidate, destination, *, seeds=None):
    import numpy as np
    from PIL import Image
    from io import BytesIO
    source = freeze_experiment(runtime, parent, candidate)
    bundle = load_bundle(runtime, source)
    available = [p['seed'] for p in bundle['views']['paired']]
    chosen = sorted(seeds if seeds is not None else available)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(available):
        raise ValueError('Select existing source seeds exactly once')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = owned_folder(runtime, 'candidates', candidate)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    samples = []
    for sample in manifest['request']['samples']['val']:
        # Generated flat filenames avoid interpreting sample IDs as paths.
        name = 'labels/' + str(len(samples)) + '.png'
        immutable(destination, name, (root / sample['label']).read_bytes())
        samples.append({'sample_id': sample['sample_id'], 'label': name})
    trials = []
    for seed in chosen:
        for variant in ('baseline', 'candidate'):
            raw = json.loads(Path(bundle['source_receipts'][f'{seed}/{variant}']['path']).read_bytes())
            predictions = []
            for index, prediction in enumerate(raw['predictions']):
                bits = np.unpackbits(np.frombuffer(base64.b64decode(prediction['packed_bits'], validate=True),
                    dtype=np.uint8), bitorder='big')
                name = f'predictions/{seed}-{variant}/{index}.png'
                image = BytesIO()
                Image.fromarray((bits.reshape(256, 256) * 255).astype(np.uint8)).save(image, format='PNG')
                immutable(destination, name, image.getvalue())
                predictions.append({'sample_id': prediction['sample_id'], 'path': name})
            trials.append({'seed': seed, 'variant': variant, 'predictions': predictions,
                'training_provenance': {'kind': 'ADAPTATION_OF_EXISTING_VERIFIED_EXPERIMENT_NOT_NEW',
                    'source_bundle': source['identity'], 'parent_run_id': parent, 'candidate_id': candidate}})
    value = {'version': VERSION, 'task': 'binary_change_detection',
        'dataset': {'name': 'SYSU-CD', 'version': 'existing-frozen-256-validation-images',
            'split': 'development_validation', 'evaluation_scope': 'Previously used development data; not independent holdout'},
        'encoding': {'negative': 0, 'positive': 255}, 'samples': samples, 'trials': trials}
    immutable(destination, 'manifest.json', json.dumps(value, ensure_ascii=False, indent=2).encode())
    return value, bundle


if __name__ == '__main__':
    from paperpilot_agentic.orchestrator import ResearchAgentRuntime
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace', 'db', 'parent', 'candidate', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--seed', action='append', type=int)
    args = parser.parse_args()
    runtime = ResearchAgentRuntime(workspace=args.workspace, db_path=args.db)
    value, _ = export(runtime, args.parent, args.candidate, args.output, seeds=args.seed)
    print(json.dumps({'samples': len(value['samples']), 'trials': len(value['trials']),
                      'new_training': 0, 'model_calls': 0}))
