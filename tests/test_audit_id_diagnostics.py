import json

import pytest

from paperpilot_agentic.answer_audit import validate_audit


@pytest.mark.parametrize('ids,missing,unexpected,duplicates', [
    (['a'], ['b'], [], []),
    (['a', 'a'], ['b'], [], ['a']),
    (['a', 'b', 'x'], [], ['x'], []),
])
def test_id_contract_reports_exact_repair_targets(ids, missing, unexpected, duplicates):
    draft = {'facts': [{'id': 'a'}, {'id': 'b'}]}
    with pytest.raises(ValueError, match='exactly once') as exc:
        validate_audit({'assessments': [{'fact_id': fid} for fid in ids]}, draft, [])
    details = json.loads(str(exc.value).split(': ', 1)[1])
    assert details == {'expected': ['a', 'b'], 'missing': missing,
                       'unexpected': unexpected, 'duplicates': duplicates}


def test_invalid_id_is_a_repairable_contract_error_not_type_error():
    with pytest.raises(ValueError, match='string fact_id'):
        validate_audit({'assessments': [{'fact_id': []}]}, {'facts': [{'id': 'a'}]}, [])
