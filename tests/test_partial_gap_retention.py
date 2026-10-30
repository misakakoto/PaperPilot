from paperpilot_agentic.claim_graph import derive_claim_evidence_state


def test_weak_evidence_does_not_close_gap_while_claim_is_partial():
    claim = {'gaps': ['primary_source'], 'required_evidence_types': ['TEXT'], 'metadata': {}}
    row = {'evidence_id': 'e1', 'evidence_type': 'TEXT', 'confidence': .82,
           'polarity': 'SUPPORTS', 'provenance': {'gap': 'primary_source'}}
    result = derive_claim_evidence_state(claim, [row, {**row, 'evidence_id': 'e2'}])
    assert result['status'].value == 'PARTIAL'
    assert result['gaps'] == ['primary_source']
    result = derive_claim_evidence_state(claim, [row, {**row, 'evidence_id': 'e2'}, {**row, 'evidence_id': 'e3'}])
    assert result['status'].value == 'SUPPORTED'
    assert result['gaps'] == []


def test_other_gap_evidence_cannot_close_weak_primary_source():
    claim = {'gaps': ['primary_source', 'numeric_results'], 'required_evidence_types': ['TEXT'], 'metadata': {}}
    rows = [{'evidence_type': 'TEXT', 'confidence': .82, 'polarity': 'SUPPORTS',
             'provenance': {'gap': 'primary_source'}},
            {'evidence_type': 'TABLE', 'confidence': .99, 'polarity': 'SUPPORTS',
             'content': 'Accuracy 95%', 'provenance': {'gap': 'numeric_results'}}]
    result = derive_claim_evidence_state(claim, rows)
    assert result['gaps'] == ['primary_source']
