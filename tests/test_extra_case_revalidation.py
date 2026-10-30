import pytest
from evals.interview.budget import save
from evals.interview.retry_grant import grant_extra_case, submission_ceiling


def test_extra_case_has_one_bounded_execution_and_preserves_originals(tmp_path):
    case = 'case_a'
    (tmp_path/'revisions/0001').mkdir(parents=True)
    (tmp_path/'revisions/0002').mkdir()
    save(tmp_path/(case+'-submission.json'), {'original': True})
    save(tmp_path/'revisions/0001'/(case+'-submission.json'), {'repaired': True})
    identity = {'source_sha256': 'frozen-code'}
    grant = grant_extra_case(tmp_path, identity, case, 'Human authorized one more case within existing allowance')
    protocol = {'identity': identity, 'extra_case_revalidation': grant}
    assert submission_ceiling(tmp_path, protocol, case) == 3
    assert submission_ceiling(tmp_path, protocol, 'other') == 2
    save(tmp_path/'revisions/0002'/(case+'-submission.json'), {'extra': True})
    assert submission_ceiling(tmp_path, protocol, case) == 3
    # The ceiling never grows from re-opening the same grant.
    with pytest.raises(ValueError):
        grant_extra_case(tmp_path, identity, case, 'another')
    (tmp_path/(case+'-submission.json')).write_text('{"tampered":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        submission_ceiling(tmp_path, protocol, case)


def test_missing_or_changed_authorization_never_expands_default(tmp_path):
    identity = {'source_sha256': 'code'}
    assert submission_ceiling(tmp_path, {'identity': identity}, 'a') == 2
    with pytest.raises(ValueError):
        grant_extra_case(tmp_path, identity, 'a', None)
    save(tmp_path/'a-submission.json', {})
    (tmp_path/'revisions/0001').mkdir(parents=True)
    save(tmp_path/'revisions/0001/a-submission.json', {})
    grant = grant_extra_case(tmp_path, identity, 'a', 'explicit')
    with pytest.raises(ValueError, match='identity'):
        submission_ceiling(tmp_path, {'identity': {'source_sha256': 'changed'},
                                     'extra_case_revalidation': grant}, 'a')
