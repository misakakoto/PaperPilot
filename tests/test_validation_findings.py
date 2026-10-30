from copy import deepcopy

from paperpilot_agentic.validation_findings import AuditTrail
from paperpilot_agentic.table_claims import apply_row_guard


def test_transition_trace_preserves_input_and_records_exact_fact_and_validator():
    draft = {'facts': [{'id': 'a', 'evidence_ids': ['source']}, {'id': 'b', 'evidence_ids': ['source2']}]}
    audit = {'assessments': [{'fact_id': 'a', 'verdict': 'SUPPORTED'},
                             {'fact_id': 'b', 'verdict': 'SUPPORTED'}], 'coverage': 'COMPLETE'}
    original = deepcopy(audit)
    trail = AuditTrail(audit, draft)
    after = deepcopy(audit)
    after['assessments'][1].update(verdict='UNSUPPORTED', reason='source row mismatch')
    trail.capture(apply_row_guard, after)
    assert audit == original and after['assessments'][0] == audit['assessments'][0]
    assert len(trail.findings) == 1
    record = trail.findings[0]
    assert record['fact_id'] == 'b' and record['evidence_ids'] == ['source2']
    assert record['validator'].endswith('table_claims.apply_row_guard')
    assert record['validator_version'] and record['validator_source_sha256']
    assert record['action'] == 'WITHHOLD' and record['after']['reason'] == 'source row mismatch'
    trail.capture(apply_row_guard, after)
    assert len(trail.findings) == 1
