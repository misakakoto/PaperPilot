"""Reconcile a host-written receipt without invoking any execution backend.

This verifies ownership and consistency, not adversarial Python attestation.
Old RUNNING calls without a durable intent remain unknown.
"""
import json
from pathlib import Path

from .sandbox import ExperimentResult, _hash_tree

_IGNORE = {'.git', '.paperpilot_agentic', '__pycache__', '.pytest_cache'}


def execution_intent(protocol, execution_id):
    return {'version': 'prototype-execution-intent-v1', 'execution_id': execution_id,
            'protocol': protocol.to_dict(), 'protocol_hash': protocol.fingerprint,
            'source_hash': _hash_tree(Path(protocol.source_dir), ignore=_IGNORE)}


def recover_receipt(protocol, execution_id, intent):
    """Accept only a complete, successful host execution (worker may fail checks)."""
    if intent != execution_intent(protocol, execution_id):
        raise ValueError('Missing or changed execution intent')
    root = Path(protocol.output_dir)
    folder = root/('experiment_'+protocol.fingerprint.split(':')[1][:24])

    def owned_file(path, expected):
        path = Path(path)
        if path != expected or not path.is_file():
            raise ValueError('Receipt file missing or path changed')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Receipt path escapes owned output')
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise ValueError('Receipt path contains a symbolic link')
            if parent == root:
                break
        return path

    result_path = owned_file(folder/'result.json', folder/'result.json')
    data = json.loads(result_path.read_text(encoding='utf-8'))
    run = ExperimentResult(**data)
    if (run.experiment_id != folder.name or run.protocol_hash != protocol.fingerprint
            or run.status != 'COMPLETED' or run.exit_code != 0 or run.timed_out is not False
            or run.error is not None or run.backend != 'docker' or run.network_isolated is not True
            or run.source_unchanged is not True or run.source_hash_before != intent['source_hash']
            or run.source_hash_after != intent['source_hash']):
        raise ValueError('Receipt does not prove completed isolated execution')
    paths = {'result_path': 'result.json', 'preregistration_path': 'preregistration.json',
             'stdout_path': 'stdout.log', 'stderr_path': 'stderr.log'}
    for field, name in paths.items():
        owned_file(getattr(run, field), folder/name)
    prereg = json.loads((folder/'preregistration.json').read_text(encoding='utf-8'))
    if (prereg.get('experiment_id') != folder.name or prereg.get('protocol_hash') != protocol.fingerprint
            or prereg.get('protocol') != protocol.to_dict()):
        raise ValueError('Preregistration changed')
    # Prototype protocols have exactly one maintained JSON metrics output.
    if protocol.metric_files != ['metrics.json']:
        raise ValueError('Unsupported recovery metrics protocol')
    metrics_file = owned_file(folder/'metrics'/'metrics.json', folder/'metrics'/'metrics.json')
    metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
    if not isinstance(metrics, dict) or metrics != run.metrics:
        raise ValueError('Receipt metrics disagree with saved metrics')
    evidence = [a for a in run.artifacts if a.get('kind') == 'experiment_metrics']
    if (len(evidence) != 1 or evidence[0].get('path') != str(metrics_file.resolve())
            or evidence[0].get('sha256') != _hash_tree(metrics_file.parent)
            or evidence[0].get('metadata', {}).get('source_name') != 'metrics.json'):
        raise ValueError('Metric evidence changed')
    return run
