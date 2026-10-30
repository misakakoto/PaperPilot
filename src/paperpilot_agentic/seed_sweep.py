"""Bounded seed replication behind the existing sandbox runner interface.

The approved argv supplies the binding, not model-authored flags. A reported seed
is an execution contract check, not proof that arbitrary code used its RNG correctly.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Callable

from .models import ExperimentProtocol, utc_now
from .sandbox import ExperimentResult, _evaluate_criteria, _hash_tree

SEED_ARGUMENT = '__PAPERPILOT_SEED__'
_IGNORE = {'.git', '.paperpilot_agentic', '__pycache__', '.pytest_cache'}


def validate_seed_sweep(protocol: ExperimentProtocol) -> None:
    seeds = protocol.seed_sweep
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 8
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Seed sweep requires 1..8 distinct integer seeds')
    if protocol.command.count(SEED_ARGUMENT) != 1 or protocol.command[0] == SEED_ARGUMENT:
        raise ValueError('Approved argv requires one standalone __PAPERPILOT_SEED__ argument')
    if len(protocol.metric_files) != 1:
        raise ValueError('Seed sweep requires one metrics JSON containing an integer seed')
    if not protocol.acceptance_criteria:
        raise ValueError('Seed sweep requires preregistered acceptance criteria')
    if protocol.study_design.get('seeds', seeds) != seeds:
        raise ValueError('Study design seeds disagree with execution seeds')
    if not 1 <= protocol.policy.timeout_seconds * len(seeds) <= 480:
        raise ValueError('Seed sweep total execution timeout must be within 480 seconds')
    source, output = Path(protocol.source_dir).resolve(), Path(protocol.output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError('Seed sweep output must be outside the original source directory')


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_seed_sweep(
    protocol: ExperimentProtocol,
    run_one: Callable[[ExperimentProtocol], ExperimentResult],
) -> ExperimentResult:
    validate_seed_sweep(protocol)
    source = Path(protocol.source_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    initial_hash = _hash_tree(source, ignore=_IGNORE)
    experiment_id = 'sweep_' + protocol.fingerprint.split(':')[-1][:24]
    output = Path(protocol.output_dir).resolve() / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    journal_path, result_path = output / 'seed_runs.json', output / 'result.json'
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding='utf-8'))
        if journal['source_hash'] != initial_hash or journal['protocol_hash'] != protocol.fingerprint:
            raise ValueError('Existing sweep identity changed; create a new approved protocol')
        if result_path.exists():
            return ExperimentResult(**json.loads(result_path.read_text(encoding='utf-8')))
        raise RuntimeError('Unresolved seed sweep: inspect recorded attempts before recovery; automatic rerun forbidden')

    prereg = output / 'preregistration.json'
    _save(prereg, {'protocol': protocol.to_dict(), 'protocol_hash': protocol.fingerprint,
                  'source_hash': initial_hash, 'aggregation': 'all_seeds_must_meet_criteria',
                  'max_execution_seconds': len(protocol.seed_sweep) * protocol.policy.timeout_seconds})
    rows = [{'seed': seed, 'status': 'NOT_RUN'} for seed in protocol.seed_sweep]
    journal = {'source_hash': initial_hash, 'protocol_hash': protocol.fingerprint, 'seed_runs': rows}
    _save(journal_path, journal)
    started_at, started = utc_now(), time.monotonic()
    # Reserve at most 60 s for container setup/teardown, under the 600 s tool lease.
    deadline = started + len(rows) * protocol.policy.timeout_seconds + 60
    outcomes, artifacts = [], []
    for row in rows:
        if (time.monotonic() + protocol.policy.timeout_seconds > deadline
                or _hash_tree(source, ignore=_IGNORE) != initial_hash):
            row['error'] = 'Remaining budget insufficient or original source changed'
            _save(journal_path, journal)
            continue
        seed = row['seed']
        child = replace(protocol, seed_sweep=[], seed=seed,
                        command=[str(seed) if arg == SEED_ARGUMENT else arg for arg in protocol.command],
                        output_dir=str(output / f'seed-{seed}'))
        row.update(status='RUNNING', protocol_hash=child.fingerprint, protocol=child.to_dict())
        _save(journal_path, journal)  # An interruption cannot erase an attempted invocation.
        try:
            result = run_one(child)
            row.update(status=result.status, result=result.to_dict())
            metrics = result.metrics
            seed_ok = type(metrics.get('seed')) is int and metrics['seed'] == seed
            integrity = (result.protocol_hash == child.fingerprint and result.source_unchanged
                         and result.source_hash_before == initial_hash
                         and result.source_hash_after == initial_hash)
            criteria, accepted = _evaluate_criteria(metrics, protocol.acceptance_criteria)
            error = None
            if result.status != 'COMPLETED' or not seed_ok or not integrity or not result.network_isolated:
                error = 'Execution, reported seed, source identity or isolation check failed'
                accepted = None
            row.update(seed_verified=seed_ok, integrity_verified=integrity, accepted=accepted)
            if error:
                row['error'] = error
                outcomes.append({'seed': seed, 'passed': None, 'error': error})
            outcomes.extend({**item, 'seed': seed} for item in criteria)
            artifacts.extend(result.artifacts)
        except Exception as exc:
            row.update(status='FAILED', accepted=None, error=f'{type(exc).__name__}: {exc}')
            outcomes.append({'seed': seed, 'passed': None, 'error': row['error']})
        _save(journal_path, journal)

    for row in rows:
        if row['status'] == 'NOT_RUN':
            outcomes.append({'seed': row['seed'], 'passed': None, 'error': row.get('error', 'Not run')})
    complete = all(row['status'] == 'COMPLETED' for row in rows)
    evaluable = all(type(row.get('accepted')) is bool for row in rows)
    final_hash = _hash_tree(source, ignore=_IGNORE)
    accepted = all(row['accepted'] for row in rows) if evaluable and initial_hash == final_hash else None
    artifacts.extend([
        {'kind': 'experiment_preregistration', 'path': str(prereg), 'metadata': {}},
        {'kind': 'seed_execution_journal', 'path': str(journal_path), 'metadata': {}},
        {'kind': 'experiment_result', 'path': str(result_path), 'metadata': {}},
    ])
    result = ExperimentResult(
        experiment_id=experiment_id, status='COMPLETED' if complete else 'FAILED',
        backend='seed_sweep', exit_code=0 if complete else None,
        timed_out=any(row.get('result', {}).get('timed_out', False) for row in rows),
        started_at=started_at, completed_at=utc_now(), duration_seconds=time.monotonic() - started,
        protocol_hash=protocol.fingerprint, preregistration_path=str(prereg), result_path=str(result_path),
        stdout_path='', stderr_path='', metrics={'by_seed': {
            str(row['seed']): row.get('result', {}).get('metrics', {}) for row in rows}},
        criteria=outcomes, accepted=accepted, source_hash_before=initial_hash,
        source_hash_after=final_hash, source_unchanged=initial_hash == final_hash,
        network_isolated=all(row.get('result', {}).get('network_isolated') is True for row in rows),
        isolation_notes=['All seeds must meet criteria; not a statistical significance test.',
                         'See each seed result for logs and isolation backend.'],
        artifacts=artifacts, seed_runs=rows,
        error=None if complete else 'One or more registered seeds failed or were not run',
    )
    _save(result_path, result.to_dict())
    return result
