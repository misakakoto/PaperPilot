import pytest

from test_diagnosis_content_review import revised_provider
from test_imported_diagnosis_harness import setup


@pytest.mark.parametrize('mode', ['strict', 'assisted'])
def test_delivery_mode_preserves_review_and_bounds_revision(tmp_path, mode):
    _, harness, reader, binding = setup(tmp_path)
    reader.complete = revised_provider(reader, reject=2)
    rid = harness.start_imported_diagnosis(
        binding['bundle_id'], 'delivery-review-001', comparison_mode='summary',
        report_protocol='metric-facts-v3', delivery_mode=mode)['run']['run_id']
    result = harness.advance(rid)
    state = result['research_episode']
    assert state['phase'] == ('COMPLETED' if mode == 'assisted' else 'PARTIAL')
    assert state['content_review']['verdict'] == 'NEEDS_REVISION'
    assert state['analysis_revisions'] == 1
    assert state['assurance']['facts'] == 'HOST_RECOMPUTED'
    assert state['new_gpu_starts'] == 0
    answer = result['final_answer']['answer']
    assert ('待人工复核草稿' in answer) == (mode == 'assisted')
    calls = len(reader.messages)
    harness.advance(rid)
    assert len(reader.messages) == calls
    with pytest.raises(ValueError):
        harness.start_imported_diagnosis(
            binding['bundle_id'], 'delivery-review-001', comparison_mode='summary',
            report_protocol='metric-facts-v3',
            delivery_mode='strict' if mode == 'assisted' else 'assisted')


def test_assisted_does_not_publish_forged_observations(tmp_path):
    _, harness, reader, binding = setup(tmp_path)
    reader.complete = revised_provider(reader, invalid_ref=True)
    rid = harness.start_imported_diagnosis(
        binding['bundle_id'], 'delivery-invalid-001', comparison_mode='summary',
        report_protocol='metric-facts-v3', delivery_mode='assisted')['run']['run_id']
    result = harness.advance(rid)
    assert result['research_episode']['phase'] == 'PARTIAL'
    assert '待人工复核草稿' not in result['final_answer']['answer']


@pytest.mark.parametrize('mode,protocol', [('invalid', 'metric-facts-v3'), ('assisted', 'legacy-v1')])
def test_delivery_mode_rejects_invalid_contract(tmp_path, mode, protocol):
    _, harness, _, binding = setup(tmp_path)
    with pytest.raises(ValueError):
        harness.start_imported_diagnosis(binding['bundle_id'], 'invalid-delivery-001',
            delivery_mode=mode, report_protocol=protocol)
