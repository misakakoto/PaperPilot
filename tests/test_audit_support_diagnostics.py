from copy import deepcopy
import pytest

from paperpilot_agentic.answer_contracts import AuditSupportError, conservative_audit_contract
from paperpilot_agentic.execution_diagnostics import execution_fault


def test_completed_rejection_is_not_mislabeled_as_invalid_supported_citations():
    draft = {'facts': [{'id': 'f1', 'kind': 'SOURCE_FACT', 'text': 'Claim',
                        'evidence_ids': ['e']}], 'limitations': ['Unreviewed caveat']}
    spans = [{'span_id': 's0', 'evidence_id': 'e', 'text': 'Source'}]
    audit = {'assessments': [{'fact_id': 'f1', 'verdict': 'UNSUPPORTED', 'relevance': 'DIRECT',
                             'span_ids': [], 'reason': 'Values absent in the source'}],
             'coverage': 'PARTIAL', 'missing_aspects': [], 'omitted_answer_spans': [],
             'limitation_assessments': [{'limitation_index': 0, 'role': 'REQUEST_GAP',
                 'reason': 'Synthetic invalid caveat mapping', 'requirement': 'undeclared requirement'}]}
    before = deepcopy(audit)
    with pytest.raises(AuditSupportError, match='No source-supported facts'):
        conservative_audit_contract(audit, draft, spans, question='What value?')
    assert audit == before
    fault = execution_fault({'status': 'FAILED', 'error': 'AuditSupportError', 'metadata': {}}, {})
    assert fault['code'] == 'NO_SUPPORTED_FACTS'
    audit['assessments'][0].update(verdict='SUPPORTED', span_ids=['unknown'])
    with pytest.raises(ValueError, match='All repaired audit support relied on invalid citation spans'):
        conservative_audit_contract(audit, draft, spans, question='What value?')
