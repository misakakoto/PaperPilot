"""One explicitly authorized extra case execution, without changing allowance."""
from .campaign import sha


def submissions(folder, case):
    return [p for p in [folder/(case+'-submission.json'),
            *sorted((folder/'revisions').glob('*/'+case+'-submission.json'))] if p.exists()]


def grant_extra_case(folder, identity, case, instruction):
    paths = submissions(folder, case)
    if not instruction or len(paths) != 2:
        raise ValueError('Extra case revalidation requires explicit instruction and two preserved submissions')
    return {'version': 'one-extra-case-v1', 'case': case,
            'source_sha256': identity['source_sha256'], 'human_instruction': instruction,
            'max_submissions': 3, 'allowance_changed': False,
            'preserved_submissions': [{'path': str(p.relative_to(folder)), 'sha256': sha(p)} for p in paths]}


def submission_ceiling(folder, protocol, case):
    grant = protocol.get('extra_case_revalidation')
    if not grant or grant.get('case') != case:
        return 2
    if (grant.get('version') != 'one-extra-case-v1' or grant.get('max_submissions') != 3
            or grant.get('allowance_changed') is not False or not grant.get('human_instruction')
            or grant.get('source_sha256') != protocol['identity']['source_sha256']
            or len(grant.get('preserved_submissions', [])) != 2):
        raise ValueError('Invalid extra-case authorization identity')
    owned = {str(p.relative_to(folder)): p for p in submissions(folder, case)}
    if len({r['path'] for r in grant['preserved_submissions']}) != 2:
        raise ValueError('Extra-case authorization must preserve two distinct submissions')
    for row in grant['preserved_submissions']:
        if row['path'] not in owned or sha(owned[row['path']]) != row['sha256']:
            raise ValueError('Previously preserved submission changed')
    return 3
