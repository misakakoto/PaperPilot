"""Freeze product outputs for source review; no model calls or answer rewriting.

Review packets omit model-audit verdicts. They cannot prove reviewer independence
or source truth: hashes bind artifacts, not facts. Pending grades never count as
passes. A frozen cohort includes every planned case, including missing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_new(path: Path, data: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(data)


def prepare(source: Path, output: Path, *, protocol: str, scope: str) -> dict:
    """Snapshot live_reliability_check outputs into a fresh private directory."""
    if not protocol.strip() or not scope.strip():
        raise ValueError('Declare protocol and evaluation scope')
    frozen = source / 'frozen-cases.json'
    raw_cases = frozen.read_bytes()
    cases = json.loads(raw_cases.decode('utf-8-sig'))
    cases = cases['cases'] if isinstance(cases, dict) else cases
    if not isinstance(cases, list) or not cases:
        raise ValueError('Expected nonempty frozen cases')
    ids, questions = set(), set()
    rows, snapshots = [], {'frozen-cases.json': raw_cases}
    for index, case in enumerate(cases, 1):
        cid, question = case.get('id'), case.get('question')
        if not isinstance(cid, str) or not cid.strip() or cid in ids:
            raise ValueError('Case IDs must be nonempty and unique')
        if not isinstance(question, str) or not question.strip() or question in questions:
            raise ValueError('Questions must be nonempty and unique')
        ids.add(cid)
        questions.add(question)
        filename = f'result-{index}.json'
        path = source / filename
        result, file_hash = {}, None
        if path.exists():
            raw = path.read_bytes()
            snapshots[filename] = raw
            file_hash = digest(raw)
            result = json.loads(raw.decode('utf-8-sig'))
            if result['run']['question'] != question:
                raise ValueError(f'Original question changed: {cid}')
        run = result.get('run', {})
        answer = (result.get('final_answer') or {}).get('answer') or ''
        if not isinstance(answer, str):
            raise ValueError('Expected unchanged textual answer')
        # Deliberately omit model confidence, claim polarities and audit verdicts.
        evidence = [{key: row.get(key) for key in (
            'evidence_id', 'content', 'source_uri', 'page', 'bbox', 'object_type',
            'artifact_path', 'content_hash')}
            for row in result.get('evidence', [])]
        binding = digest(encode({'case': case, 'result_sha256': file_hash}))
        rows.append({'case_id': cid, 'binding': binding, 'question': question,
            'criteria': case.get('review', ''), 'original_answer': answer,
            'run_status': run.get('status', 'MISSING'), 'evidence': evidence,
            'snapshot': filename if file_hash else None,
            'calls': run.get('used_llm_calls'), 'tokens': run.get('used_tokens')})
    expected = {f'result-{i}.json' for i in range(1, len(cases) + 1)}
    if {p.name for p in source.glob('result-*.json')} - expected:
        raise ValueError('Unexpected result files outside frozen cohort')
    packet = {'schema': 'paperpilot-source-review-v1', 'protocol': protocol,
        'scope': scope, 'cases': rows,
        'warning': 'Artifact hashes do not certify factual correctness or reviewer independence.'}
    packet_bytes = encode(packet)
    packet_hash = digest(packet_bytes)
    template = {'packet_sha256': packet_hash, 'reviewer_id': '',
        'reviewer_kind': 'UNSET', 'cases': [
            {'case_id': row['case_id'], 'binding': row['binding'], 'verdict': 'PENDING',
             'reason': '', 'source_checks': []} for row in rows]}
    manifest = {'schema': packet['schema'], 'packet_sha256': packet_hash,
        'files': {name: digest(raw) for name, raw in snapshots.items()}}
    # All input validation happens before creating the output; no overwrite.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in snapshots.items():
        write_new(output / name, raw)
    write_new(output / 'packet.json', packet_bytes)
    write_new(output / 'manifest.json', encode(manifest))
    write_new(output / 'review-template.json', encode(template))
    return {'count': len(rows), 'missing': sum(r['run_status'] == 'MISSING' for r in rows),
            'packet_sha256': packet_hash}


def verified_packet(folder: Path) -> tuple[dict, str]:
    manifest = read(folder / 'manifest.json')
    raw = (folder / 'packet.json').read_bytes()
    if digest(raw) != manifest['packet_sha256']:
        raise ValueError('Review packet changed')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest((folder / name).read_bytes()) != expected:
            raise ValueError('Source snapshot changed')
    packet = json.loads(raw)
    frozen = read(folder / 'frozen-cases.json')
    frozen = frozen['cases'] if isinstance(frozen, dict) else frozen
    if len(packet['cases']) != len(frozen):
        raise ValueError('Frozen denominator changed')
    for index, (row, case) in enumerate(zip(packet['cases'], frozen), 1):
        expected_file = f'result-{index}.json'
        expected_hash = manifest['files'].get(expected_file)
        if row['binding'] != digest(encode({'case': case, 'result_sha256': expected_hash})):
            raise ValueError('Case binding changed')
        if row['case_id'] != case['id'] or row['question'] != case['question']:
            raise ValueError('Original case changed')
        original = read(folder / expected_file) if expected_hash else {}
        answer = (original.get('final_answer') or {}).get('answer') or ''
        if row['original_answer'] != answer or row['run_status'] != original.get('run', {}).get('status', 'MISSING'):
            raise ValueError('Original output changed')
        original_evidence = [{key: item.get(key) for key in (
            'evidence_id', 'content', 'source_uri', 'page', 'bbox', 'object_type',
            'artifact_path', 'content_hash')} for item in original.get('evidence', [])]
        if row['evidence'] != original_evidence or row['criteria'] != case.get('review', ''):
            raise ValueError('Source evidence or review criteria changed')
    return packet, manifest['packet_sha256']


def score_review(packet: dict, packet_hash: str, review: dict) -> dict:
    if review.get('packet_sha256') != packet_hash:
        raise ValueError('Review belongs to another packet')
    if not str(review.get('reviewer_id') or '').strip() or review.get('reviewer_kind') not in {'human', 'ai'}:
        raise ValueError('Declare reviewer identity and kind (human or ai)')
    labels = review.get('cases', [])
    by_id = {row['case_id']: row for row in labels}
    if len(by_id) != len(labels) or set(by_id) != {row['case_id'] for row in packet['cases']}:
        raise ValueError('Review must contain every frozen case exactly once')
    counts = dict(PASS=0, PARTIAL=0, FAIL=0, PENDING=0, OPERATIONAL_FAILURE=0,
                  UNRESOLVED_EXECUTION=0)
    grades = {}
    for case in packet['cases']:
        label = by_id[case['case_id']]
        verdict = label.get('verdict')
        if label.get('binding') != case['binding'] or verdict not in {'PASS', 'PARTIAL', 'FAIL', 'PENDING'}:
            raise ValueError('Invalid verdict or answer binding')
        if case['run_status'] != 'COMPLETED':
            if verdict not in {'FAIL', 'PENDING'}:
                raise ValueError('Missing or noncompleted execution cannot pass')
            grade = ('OPERATIONAL_FAILURE' if case['run_status'] in {
                'FAILED', 'CANCELLED', 'BUDGET_EXHAUSTED'} else 'UNRESOLVED_EXECUTION')
        else:
            if verdict != 'PENDING':
                if not str(label.get('reason') or '').strip():
                    raise ValueError('Source review needs an explanation')
                if verdict in {'PASS', 'PARTIAL'} and not case['original_answer'].strip():
                    raise ValueError('Empty answer cannot pass')
                checks = label.get('source_checks', [])
                if case['original_answer'].strip() and not checks:
                    raise ValueError('Review an original source, not just model verdicts')
                for check in checks:
                    if not all(isinstance(check.get(k), str) and check[k].strip()
                               for k in ('source_uri', 'locator', 'note')):
                        raise ValueError('Each source check needs URI, location and note')
            grade = verdict
        counts[grade] += 1
        grades[case['case_id']] = grade
    total = len(packet['cases'])
    unreviewed = counts['PENDING'] + counts['UNRESOLVED_EXECUTION']
    complete = unreviewed == 0
    return {'reviewer_id': review['reviewer_id'], 'reviewer_kind': review['reviewer_kind'],
        'count': total, 'counts': counts, 'grades': grades, 'review_complete': complete,
        'strict_pass_rate': counts['PASS'] / total if complete else None,
        'pass_rate_lower_bound': counts['PASS'] / total,
        'pass_rate_upper_bound': (counts['PASS'] + unreviewed) / total,
        'interpretation': 'Source-review result on declared cohort, not overall accuracy. Reviewer independence is not verified.'}


def score(folder: Path, review_paths: list[Path]) -> dict:
    packet, packet_hash = verified_packet(folder)
    if not 1 <= len(review_paths) <= 2:
        raise ValueError('Supply one or two reviews')
    results = [score_review(packet, packet_hash, read(path)) for path in review_paths]
    report = {'packet_sha256': packet_hash, 'scope': packet['scope'], 'protocol': packet['protocol'],
        'reviews': results, 'agreement': None}
    if len(results) == 2:
        a, b = results
        if a['reviewer_id'] == b['reviewer_id']:
            raise ValueError('Two-review comparison requires distinct reviewer IDs')
        comparable = [cid for cid, grade in a['grades'].items()
            if grade in {'PASS', 'PARTIAL', 'FAIL'} and b['grades'][cid] in {'PASS', 'PARTIAL', 'FAIL'}]
        disagree = [cid for cid in comparable if a['grades'][cid] != b['grades'][cid]]
        report['agreement'] = {'jointly_reviewed': len(comparable), 'disagreements': disagree,
            'rate': 1 - len(disagree) / len(comparable) if comparable else None,
            'warning': 'Agreement is not correctness; no claim of independent human review.'}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--protocol', required=True)
    prep.add_argument('--scope', required=True)
    grade = sub.add_parser('score')
    grade.add_argument('--packet', type=Path, required=True)
    grade.add_argument('--review', type=Path, action='append', required=True)
    grade.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    private = Path('.paperpilot').resolve()
    destination = args.output.resolve()
    if not destination.is_relative_to(private) or destination == private:
        parser.error('Output must stay in a new path under .paperpilot')
    if args.command == 'prepare':
        report = prepare(args.source, destination, protocol=args.protocol, scope=args.scope)
    else:
        report = score(args.packet, args.review)
        write_new(destination, encode(report))
    print(json.dumps({'output': str(destination), 'count': report.get('count'),
                      'packet_sha256': report['packet_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
