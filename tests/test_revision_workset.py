import pytest

from paperpilot_agentic.reader_workset import revision_workset, source_aware_workset


def test_revision_keeps_whole_original_evidence_even_if_new_allocation_would_shrink():
    rows = [{'evidence_id': 'a', 'source_uri': 'paper://a/c', 'content': 'A' * 500},
            {'evidence_id': 'b', 'source_uri': 'paper://b/c', 'content': 'B' * 500}]
    working, context = source_aware_workset(rows, max_bytes=2000)
    shrunk, _ = source_aware_workset(rows, max_bytes=700)
    assert len(shrunk) < len(working)
    restored, info = revision_workset(working, context)
    assert restored == rows
    assert info['selected_evidence_ids'] == ['a', 'b']
    assert info['budget_enforcement'] == 'per-request-input-output-runtime-guard'
    assert context['policy'] != info['policy']


def test_revision_refuses_a_different_source_window():
    with pytest.raises(ValueError, match='differ'):
        revision_workset([{'evidence_id': 'new'}], {'selected_evidence_ids': ['old']})
