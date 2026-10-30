import pytest
from evals.interview.budget import CampaignBudget, save


@pytest.mark.parametrize('status,keyword,http_status', [('FAILED', False, 400),
    ('UNKNOWN', False, 400), ('FAILED', True, 400), ('FAILED', False, 403)])
def test_only_proven_local_json_contract_rejection_can_carry_cost(tmp_path, status, keyword, http_status):
    identity = {'model': 'fixed', 'provider': 'qwen'}
    old = CampaignBudget(tmp_path, identity, max_tokens=10000)
    with old.transaction() as c:
        c.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?)',
                  (1, status, 3000, 0, 'fixed', 'patch', 0, None))
    save(tmp_path/'failure-001.json', {'http_status': http_status, 'provider_code': 'invalid_parameter_error'})
    save(tmp_path/'request-001.json', {'model': 'fixed', 'json_mode': True,
        'messages': [{'role': 'system', 'content': 'Return JSON' if keyword else 'Return a facts object'}]})
    new_identity = {**identity, 'revision': '1'}
    before = old.snapshot()['attempts']
    if status != 'FAILED' or keyword or http_status != 400:
        with pytest.raises(ValueError):
            old.authorize_json_rejection_revision(new_identity, evidence={'human_instruction': 'Fix and test in approved allowance'})
    else:
        old.authorize_json_rejection_revision(new_identity, evidence={'human_instruction': 'Fix and test in approved allowance'})
        new = CampaignBudget(tmp_path, new_identity, max_tokens=10000)
        assert old.snapshot()['blocked'] is True and new.snapshot()['blocked'] is False
        assert new.snapshot()['unsettled_upper_bound'] == 3000
        assert new.snapshot()['reported_tokens'] == 0 and new.snapshot()['actual_requests'] == 1
        assert new.snapshot()['attempts'] == before
