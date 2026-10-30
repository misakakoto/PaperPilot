"""Prepare paired requests; --execute explicitly submits tasks, never approves code.

Manifest: a list of {id, question, conversation_id, parent_run_id, source_ids},
optionally with prototype_profile_id and prototype_constraints.
--execute requires an explicit --base-url.
Freeze sources, model configuration and cases before executing. Default is dry run.
"""
import argparse
import copy
import json
import os
import tempfile
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path


def requests(cases, campaign):
    if not isinstance(cases, list) or not cases or not isinstance(campaign, str) or not campaign.strip():
        raise ValueError('Supply nonempty cases and campaign')
    result = []
    seen = set()
    for case in cases:
        required = {'id', 'question', 'conversation_id', 'parent_run_id', 'source_ids'}
        if (not isinstance(case, dict) or not required <= set(case)
                or set(case) - required - {'prototype_profile_id', 'prototype_constraints'}):
            raise ValueError('Invalid comparison case')
        if any(not isinstance(case[k], str) or not case[k].strip() for k in required - {'source_ids'}):
            raise ValueError('Invalid comparison case fields')
        if (not isinstance(case['source_ids'], list) or not case['source_ids']
                or any(not isinstance(s, str) or not s for s in case['source_ids'])):
            raise ValueError('Select comparison evidence')
        if 'prototype_profile_id' in case and (not isinstance(case['prototype_profile_id'], str)
                                                or not case['prototype_profile_id']):
            raise ValueError('Invalid prototype profile')
        if case['id'] in seen:
            raise ValueError('Duplicate comparison case ID')
        seen.add(case['id'])
        for repeat in (1, 2):
            for mode in ('simple_loop', 'adaptive'):
                payload = {k: case[k] for k in ('question', 'conversation_id', 'parent_run_id', 'source_ids')}
                if 'prototype_profile_id' in case:
                    payload['prototype_profile_id'] = case['prototype_profile_id']
                if 'prototype_constraints' in case:
                    if not isinstance(case['prototype_constraints'], dict):
                        raise ValueError('Invalid prototype constraints')
                    payload['prototype_constraints'] = copy.deepcopy(case['prototype_constraints'])
                result.append(payload | {
                    'request_id': f'{campaign}-{case["id"]}-{repeat}-{mode}',
                    'action': 'research_prototype', 'prototype_mode': mode})
    return result


def _save(output, records):
    # Never leave a partially rewritten receipt if the process dies during save.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                     prefix=output.name+'.', suffix='.tmp', delete=False) as handle:
        temp = Path(handle.name)
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def submit_jobs(jobs, output, *, base_url=None):
    output = Path(output)
    if base_url is not None:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {'http', 'https'} or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or parsed.username or parsed.password or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
            raise ValueError('Use an explicit local PaperPilot base URL')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents a second launch from replacing UNKNOWN or
    # partially submitted campaigns. An empty reserved file also requires inspection.
    with output.open('x', encoding='utf-8') as handle:
        handle.write('[]')
        handle.flush()
        os.fsync(handle.fileno())
    records = []
    for payload in jobs:
        record = {'request': payload, 'status': 'NOT_RUN'}
        records.append(record)
        if base_url is not None:
            request = urllib.request.Request(base_url.rstrip('/')+'/api/agentic/turns',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
            record['status'] = 'STARTED'
            _save(output, records)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    record.update(status='SUBMITTED', response=json.load(response))
            except Exception as exc:
                record.update(status='UNKNOWN', error=type(exc).__name__)
                _save(output, records)
                raise
        _save(output, records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--base-url', help='Explicit local service origin, e.g. http://127.0.0.1:8014')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; preserve original receipts')
    jobs = requests(json.loads(args.manifest.read_text(encoding='utf-8')), args.campaign)
    if args.execute and not args.base_url:
        parser.error('--execute requires --base-url; do not guess the active service port')
    submit_jobs(jobs, args.output, base_url=args.base_url if args.execute else None)


if __name__ == '__main__':
    main()
