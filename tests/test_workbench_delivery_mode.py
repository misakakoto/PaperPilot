import pytest

from paperpilot_agentic.workbench import ResearchWorkbench
from test_imported_diagnosis_harness import setup
from test_workbench import Background


@pytest.mark.parametrize('mode', [None, 'strict', 'assisted'])
def test_workbench_accepts_delivery_mode_at_submission_boundary(tmp_path, mode):
    runtime, harness, reader, binding = setup(tmp_path)
    wb = ResearchWorkbench(runtime, Background())
    payload = {'action': 'imported_diagnosis', 'request_id': 'delivery-web-001',
               'bundle_id': binding['bundle_id']}
    if mode is not None:
        payload['delivery_mode'] = mode
    result = wb.submit(payload)
    state = harness.status(result['run_id'])['research_episode']
    assert state['delivery_mode'] == (mode or 'assisted')
    assert reader.messages == []
    assert wb.submit(payload)['reused']
