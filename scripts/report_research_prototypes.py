"""Read-only per-run report for a frozen prototype campaign; never resumes it."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.prototype_workspace import digest  # noqa: E402
from paperpilot_agentic.prototype_outcomes import validated_candidates  # noqa: E402


def assess_state(state):
    work = state['work']
    artifacts = work['artifacts']
    if any(digest(value) != identity for identity, value in artifacts.items()):
        raise ValueError('Artifact content ID changed')
    done = [t for t in work['tasks'].values() if t['status'] == 'DONE']
    if any(t.get('output') not in artifacts for t in done):
        raise ValueError('Completed task output missing')
    for task in done:
        dependencies = [work['tasks'][d] for d in task['depends_on']]
        artifact = artifacts[task['output']]
        if (any(d['status'] != 'DONE' for d in dependencies)
                or artifact['inputs'] != sorted(d['output'] for d in dependencies)
                or artifact['kind'] != task['capability'] or artifact['name'] != task['id']):
            raise ValueError('Completed task dependency version changed')
    codes = {t['output'] for t in done if t['capability'] == 'implement'}
    validations = [artifacts[t['output']] for t in done if t['capability'] == 'validate']
    covered, outcomes = set(), []
    for artifact in validations:
        value = artifact['content']
        code = value.get('validated_code_id')
        current = (code in codes and code in artifact['inputs']
                   and value.get('validation_contract') == state.get('validation_contract'))
        outcomes.append({'code_id': code, 'artifact_id': digest(artifact), 'current_binding': current,
                         'passed': value.get('passed'), 'failure_kind': value.get('failure_kind'),
                         'metrics': value.get('execution', {}).get('metrics', {}),
                         'execution_snapshot_id': value.get('execution_snapshot_id')})
        if current and value.get('passed') is True:
            covered.add(code)
    checks = [t for t in work['tasks'].values() if t['capability'] == 'validate']
    implementations = [t for t in work['tasks'].values() if t['capability'] == 'implement']
    all_current = bool(codes) and all(t['status'] == 'DONE' for t in implementations+checks)
    passed = (all_current and codes <= covered and bool(validations)
              and all(v['current_binding'] and v['passed'] is True for v in outcomes))
    engineering = ('PASSED_CURRENT' if passed else 'FAILED_OR_STALE' if validations else 'NOT_EXECUTED')
    reviews = [artifacts[t['output']]['content'] for t in done if t['capability'] == 'review']
    read_ids = sorted({artifacts[t['output']]['content'].get('evidence_id') for t in done
                       if t['capability'] == 'read' and isinstance(artifacts[t['output']]['content'], dict)
                       and artifacts[t['output']]['content'].get('evidence_id')})
    return {'phase': state['phase'], 'task_completed': state.get('completed', False),
            'engineering': engineering, 'current_code_ids': sorted(codes), 'validations': outcomes,
            'validated_candidates': validated_candidates(state),
            'historical_validation_ids': sorted(i for i, a in artifacts.items() if a['kind'] == 'validate'
                                                 and i not in {t['output'] for t in done}),
            'model_reviews': reviews, 'model_review_facts': 'NOT_HOST_VERIFIED',
            'source_read_ids': read_ids, 'selected_sources_not_read': sorted(set(state.get('sources', {}))-set(read_ids)),
            'requirements_satisfied': 'NOT_ESTABLISHED_BY_ENGINEERING_PASS',
            'implementation_attempts': work['repairs'], 'code_repairs': max(0, work['repairs']-1),
            'repair_count_scope': 'Includes static-rejected implementation attempts',
            'scientific_validity': 'NOT_VERIFIED', 'stop_reason': state.get('stop_reason')}


def campaign_report(folder):
    folder = Path(folder).resolve()
    manifest_raw = (folder/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    ledger = json.loads((folder/'ledger.json').read_text(encoding='utf-8'))
    snapshot = folder/'frozen-source'
    actual = {}
    for path in (snapshot/'src').rglob('*.py'):
        if path.is_symlink() or not path.resolve().is_relative_to(snapshot):
            raise ValueError('Snapshot source escapes campaign')
        actual[path.relative_to(snapshot).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest['source_files']:
        raise ValueError('Frozen engine snapshot changed')
    cases = {c['id']: c for c in manifest['cases']}
    rows = []
    # No RuntimeDatabase construction/schema migration and no write connection.
    with sqlite3.connect((folder/'runtime.sqlite3').as_uri()+'?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        for job in manifest['jobs']:
            row = {'job_id': job['id'], 'case_id': job['case_id'], 'mode': job['mode'],
                   'repeat': job['repeat'], 'phase': 'NOT_STARTED'}
            receipt = conn.execute('SELECT * FROM research_episodes WHERE request_key=?',
                                   (manifest['campaign']+'-'+job['id'],)).fetchone()
            if receipt is None:
                rows.append(row)
                continue
            rid = receipt['run_id']
            run = dict(conn.execute('SELECT * FROM runs WHERE run_id=?', (rid,)).fetchone())
            state = json.loads(receipt['state_json'])
            case = cases[job['case_id']]
            expected_contract = {'kind': 'unbound', **case['constraints']}
            expected = {'goal': case['question'], 'parent': ledger['parent_run_id'],
                        'sources': sorted(ledger['source_ids']), 'conversation_id': None,
                        'mode': job['mode'], 'profile_id': None, 'validation_contract': expected_contract}
            if (receipt['fingerprint'] != digest(expected) or state['goal'] != case['question']
                    or state['mode'] != job['mode'] or state['validation_contract'] != expected_contract
                    or sorted(state['sources']) != sorted(ledger['source_ids'])):
                raise ValueError('Campaign request binding changed: '+job['id'])
            for sid, field in [('snunet_code', 'source_sha256'), ('snunet_license', 'license_sha256')]:
                if field in manifest and hashlib.sha256(state['sources'][sid]['content'].encode('utf-8')).hexdigest() != manifest[field]:
                    raise ValueError('Registered source content changed: '+sid)
            row.update(assess_state(state), run_id=rid, run_status=run['status'],
                       used_llm_calls=run['used_llm_calls'], used_tokens=run['used_tokens'],
                       budget=json.loads(run['budget_json']))
            calls = conn.execute('SELECT model,status,total_tokens FROM llm_calls WHERE run_id=?', (rid,)).fetchall()
            row['model_receipts'] = [dict(c) for c in calls]
            row['model_names_match'] = all(c['model'] == manifest['model'] for c in calls)
            attempts = conn.execute('SELECT state,reserved_tokens,usage_known FROM llm_attempts WHERE run_id=?',
                                    (rid,)).fetchall()
            row['unknown_results'] = sum(a['state'] in {'STARTED', 'UNKNOWN'} for a in attempts)
            row['unknown_usage_reserved_tokens'] = sum(a['reserved_tokens'] for a in attempts if not a['usage_known'])
            approvals = conn.execute('SELECT approval_id,status,decision_json FROM approvals WHERE run_id=?',
                                     (rid,)).fetchall()
            row['approvals'] = [{'approval_id': a['approval_id'], 'status': a['status'],
                                 'decision': json.loads(a['decision_json']) if a['decision_json'] else None}
                                for a in approvals]
            row['human_approval_decisions'] = sum(a['decision'] is not None and
                a['decision'].get('actor') == 'human' for a in row['approvals'])
            row['delegated_approval_decisions'] = sum(a['decision'] is not None and
                a['decision'].get('actor') == 'codex-delegated' for a in row['approvals'])
            row['other_human_interventions'] = None  # Not inferred from missing events.
            tools = conn.execute("SELECT attempt FROM tool_calls WHERE run_id=? AND tool_name='prototype.validate'",
                                 (rid,)).fetchall()
            row['execution_attempts'] = sum(t['attempt'] for t in tools)
            rows.append(row)
    return {'campaign': manifest['campaign'], 'model': manifest['model'],
            'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(), 'engine_snapshot_matches': True,
            'rows': rows, 'scope': 'Database snapshot; generation ledger phases are not authoritative. '
            'Reports saved host receipts without re-execution or adversarial attestation. '
            'Per-run outcomes only; no aggregate superiority claim. Different models must not be pooled.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New report path; never overwritten')
    args = parser.parse_args()
    report = campaign_report(args.campaign)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
