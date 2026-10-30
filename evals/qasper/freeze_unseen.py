"""Freeze a historically excluded internal holdout without provider requests."""
import argparse
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

from .oracle_reader import answer_type


TYPES = ('extractive', 'abstractive', 'boolean', 'unanswerable')


def audit_history(dataset, roots):
    """Audit stored IDs plus exact normalized questions in persisted run databases.

    Dataset downloads and model caches are inputs, not prior experiments. All
    other JSON/JSONL, scripts/docs/logs and runtime databases are audited. Failed
    reads are surfaced and prevent freezing rather than silently shrinking scope.
    """
    data = json.loads(dataset.read_bytes())
    known = {q['question_id'] for p in data.values() for q in p['qas']}
    by_text = {}
    for paper in data.values():
        for q in paper['qas']:
            by_text.setdefault(' '.join(q['question'].split()), set()).add(q['question_id'])
    used, records, errors, visited = set(), [], [], set()
    ignored = {'datasets', 'model-cache', '.git', 'node_modules', '__pycache__', '.venv'}
    suffixes = {'.json', '.jsonl', '.md', '.py', '.txt', '.log', '.csv', '.tsv', '.db', '.sqlite3'}
    for root in roots:
        for folder, dirs, files in os.walk(root, onerror=lambda exc: errors.append({'path':exc.filename, 'error':type(exc).__name__})):
            # Previous evaluation folders are history, regardless of their date.
            # Create the new output only AFTER this audit; never hide a whole
            # result directory just to avoid reading the current manifest.
            dirs[:] = [d for d in dirs if d not in ignored]
            for name in files:
                path = Path(folder) / name
                if path.suffix not in suffixes or path.resolve() in visited or path.resolve() == dataset.resolve():
                    continue
                visited.add(path.resolve())
                try:
                    digest = hashlib.sha256()
                    found = set()
                    with path.open('rb') as handle:
                        overlap = b''
                        while block := handle.read(1024 * 1024):
                            digest.update(block)
                            found.update(x.decode() for x in re.findall(rb'(?<![a-f0-9])[a-f0-9]{40}(?![a-f0-9])', overlap + block)
                                         if x.decode() in known)
                            overlap = block[-256:]
                    if path.suffix in {'.db', '.sqlite3'}:
                        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as conn:
                            tables = {x[0] for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                            if 'runs' in tables and 'question' in {x[1] for x in conn.execute('PRAGMA table_info(runs)')}:
                                for (question,) in conn.execute('SELECT question FROM runs'):
                                    found.update(by_text.get(' '.join((question or '').split()), set()))
                    used.update(found)
                    records.append({'path': str(path), 'sha256': digest.hexdigest(), 'question_ids': sorted(found)})
                except (OSError, sqlite3.Error) as exc:
                    errors.append({'path': str(path), 'error': type(exc).__name__})
    return used, {'roots': [str(p) for p in roots], 'ignored_directory_names': sorted(ignored),
                  'scanned_files': len(records), 'records': records, 'errors': errors,
                  'scope': 'Available local project and output history; not a claim about unknown/deleted external history.'}


def freeze_audited(dataset, roots, output):
    used, audit = audit_history(dataset, roots)
    selection = select_unseen(json.loads(dataset.read_bytes()), used)
    if audit['errors']:
        selection.update(status='HISTORY_AUDIT_INCOMPLETE', cases=[])
    result = {**selection, 'kind': 'qasper-train-internal-holdout-v1', 'official_test_set': False,
              'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
              'baseline_revision': '9a9e6b8', 'history_audit': audit}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def finalize_reviewed_audit(dataset, previous, output, exclusions):
    """Preserve failed reads; accept only an explicit, documented non-eval classification.

    This does not grant filesystem access or assume unreadable data is unused.
    The reviewer must supply provenance for every inaccessible path. Production
    evaluation history must never be excluded through this mechanism.
    """
    audit = json.loads(previous.read_bytes())
    paths = {row['path'] for row in audit['history_audit']['errors']}
    if set(exclusions) != paths or any(not r.get('reason') or not r.get('supporting_record') for r in exclusions.values()):
        raise ValueError('Every failed read needs explicit non-evaluation provenance')
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != audit['dataset_sha256']:
        raise ValueError('Dataset changed')
    result = {**audit, **select_unseen(json.loads(dataset.read_bytes()), set(audit['excluded_question_ids'])),
        'previous_audit_sha256': hashlib.sha256(previous.read_bytes()).hexdigest(),
        'reviewed_non_evaluation_paths': exclusions,
        'eligibility_scope': 'No match in audited available history; inaccessible paths classified below, not claimed to have been read.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def freeze_additional(dataset, previous, output, *, seed):
    """A newly authorized batch excludes the entire earlier frozen selection."""
    prior = json.loads(previous.read_bytes())
    if prior['status'] != 'FROZEN' or hashlib.sha256(dataset.read_bytes()).hexdigest() != prior['dataset_sha256']:
        raise ValueError('Previous selection or dataset is not valid')
    used = set(prior['excluded_question_ids']) | {r['question_id'] for r in prior['cases']}
    result = {**prior, **select_unseen(json.loads(dataset.read_bytes()), used, seed=seed),
        'previous_selection_sha256': hashlib.sha256(previous.read_bytes()).hexdigest(),
        'previous_selection_count': len(prior['cases']),
        'eligibility_scope': 'Reuse the documented historical audit, additionally exclude ALL earlier frozen cases, whether executed or not. New batch explicitly authorized by user.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def select_unseen(dataset, previously_used, *, per_type=10, seed='20260927'):
    rows = [{'paper_id': pid, 'question_id': qa['question_id'], 'question': qa['question'],
             'answer_type': answer_type(qa)} for pid, paper in dataset.items() for qa in paper['qas']]
    if len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate official question IDs')
    buckets = {kind: [] for kind in TYPES}
    for row in rows:
        if row['question_id'] not in previously_used and row['answer_type'] in buckets:
            buckets[row['answer_type']].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256((seed + ':' + r['question_id']).encode()).hexdigest())
    enough = all(len(bucket) >= per_type for bucket in buckets.values())
    return {'status': 'FROZEN' if enough else 'INSUFFICIENT_UNSEEN_CASES',
            'requested_count': per_type * 4, 'seed': seed,
            'available_by_type': {k: len(v) for k, v in buckets.items()},
            'excluded_question_ids': sorted({r['question_id'] for r in rows} & set(previously_used)),
            'cases': [r for k in TYPES for r in buckets[k][:per_type]] if enough else [],
            'eligibility': 'Exclude any question present in supplied historical diagnostic records, including retrieval-only runs.',
            'classification': 'Existing QASPER first-annotation answer_type; no outcome-based filtering',
            'billable_requests': 0}


def question_ids(value):
    if isinstance(value, dict):
        if isinstance(value.get('question_id'), str):
            yield value['question_id']
        for item in value.values():
            yield from question_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from question_ids(item)


def freeze(dataset, history, output):
    used, sources = set(), []
    for path in history:
        raw = path.read_bytes()
        values = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()] if path.suffix == '.jsonl' else [json.loads(raw)]
        ids = set(question_ids(values))
        used.update(ids)
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'question_count': len(ids)})
    data = dataset.read_bytes()
    result = {**select_unseen(json.loads(data), used), 'baseline_revision': '9a9e6b8',
              'dataset_sha256': hashlib.sha256(data).hexdigest(), 'history': sources,
              'history_is_exhaustive': False,
              'note': 'Provided records can disqualify cases. A successful selection still requires a complete history audit before real evaluation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--history', type=Path, nargs='+')
    parser.add_argument('--audit-roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = (freeze_audited(args.dataset, args.audit_roots, args.output) if args.audit_roots
              else freeze(args.dataset, args.history, args.output))
    print(json.dumps({k: result[k] for k in ('status', 'available_by_type', 'billable_requests')}))
