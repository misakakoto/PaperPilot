"""Zero-API infrastructure acceptance; this synthetic fixture is not research evidence."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.builtin_tools import _experiment_evaluate_handler
from paperpilot_agentic.docker_acceptance import build_docker_acceptance_protocol, resolve_local_docker_image
from paperpilot_agentic.sandbox import SandboxExperimentRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    image, _ = resolve_local_docker_image('python:3.11-slim')
    root = Path(__file__).resolve().parents[1]
    base = build_docker_acceptance_protocol(source_dir=root / 'examples/docker_acceptance',
                                            output_dir=args.output, image_id=image)
    command = list(base.command)
    command[command.index('--seed') + 1] = '__PAPERPILOT_SEED__'
    protocol = replace(base, command=command, seed=None, seed_sweep=[7, 11, 19],
                       study_design={'seeds': [7, 11, 19], 'dataset': 'synthetic XOR fixture'})
    result = SandboxExperimentRunner().run(protocol)
    evaluation = _experiment_evaluate_handler(None, {'protocol': protocol.to_dict(), 'result': result.to_dict()})
    report = {'purpose': 'Infrastructure only; not an autonomous scientific discovery',
              'api_requests': 0, 'result_path': result.result_path,
              'evaluation': evaluation.output, 'seeds': [
                  {'requested': r['seed'], 'status': r['status'],
                   'reported': r.get('result', {}).get('metrics', {}).get('seed'),
                   'seed_verified': r.get('seed_verified'),
                   'integrity_verified': r.get('integrity_verified')}
                  for r in result.seed_runs]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert evaluation.output['polarity'] == 'SUPPORTS'
    assert all(r['seed_verified'] and r['integrity_verified'] for r in report['seeds'])


if __name__ == '__main__':
    main()
