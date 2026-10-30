import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.workbench import ResearchWorkbench
from test_workbench import Background, completed
from test_prototype_workspace import Reader


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_workbench_prototype_submission_is_owned_persistent_and_visible(tmp_path, monkeypatch, mode):
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    background = Background()
    wb = ResearchWorkbench(runtime, background)
    parent = completed(wb, cid='owned-conversation')
    monkeypatch.setattr(runtime.db, 'list_evidence', lambda rid, **kw: [
        {'evidence_id': 'selected-source', 'content': 'source'}] if rid == parent else [])
    payload = {'action': 'research_prototype', 'prototype_mode': mode,
               'request_id': 'prototype-web-001', 'conversation_id': 'owned-conversation',
               'parent_run_id': parent, 'source_ids': ['selected-source'], 'question': 'build model'}
    payload['prototype_constraints'] = {'input_cases': [[1,48,80]], 'max_parameters': 1000}
    result = wb.submit(payload)
    assert wb.submit(payload)['run_id'] == result['run_id']
    assert background.submitted == [result['run_id']]
    turn = wb.turn(result['run_id'])
    assert turn['research']['prototype']['validation_contract']['max_parameters'] == 1000
    assert turn['research']['prototype']['goal'] == 'build model'
    assert runtime.db.get_run(result['run_id'])['metadata']['conversation_id'] == 'owned-conversation'
    assert runtime.db.list_tool_calls(result['run_id']) == []


def test_workbench_prototype_rejects_foreign_conversation(tmp_path):
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=Reader())
    wb = ResearchWorkbench(runtime, Background())
    parent = completed(wb, cid='foreign-conversation')
    with pytest.raises(ValueError):
        wb.submit({'action': 'research_prototype', 'request_id': 'prototype-web-foreign',
                   'conversation_id': 'other-conversation', 'parent_run_id': parent,
                   'source_ids': ['selected-source'], 'question': 'build model'})


@pytest.mark.parametrize('running', [True, False])
def test_workbench_prototype_waits_for_stable_reading_task(tmp_path, running):
    from paperpilot_agentic.models import RunStatus
    runtime=ResearchAgentRuntime(workspace=tmp_path,db_path=tmp_path/'runtime.db',llm=Reader())
    background=Background()
    wb=ResearchWorkbench(runtime,background)
    parent=completed(wb,cid='handoff-conversation')
    if running:
        runtime.db.update_run_status(parent,RunStatus.RUNNING)
    else:
        background.submitted.append(parent)
    with pytest.raises(ValueError,match='reading task to finish'):
        wb.submit({'action':'research_prototype','request_id':'premature-prototype',
            'conversation_id':'handoff-conversation','parent_run_id':parent,
            'source_ids':['selected-source'],'question':'build model'})
    assert len(runtime.list_runs())==1


def test_unknown_prototype_is_not_presented_as_resumable(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    from paperpilot_agentic.models import RunStatus
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    state['phase'] = 'UNKNOWN'
    harness._save(rid, state)
    runtime.db.update_run_status(rid, RunStatus.RUNNING)
    view = ResearchWorkbench(runtime, Background()).turn(rid)
    assert view['can_resume'] is False
    assert '禁止自动重发' in view['state_text']


def test_pending_approval_view_preserves_code_environment_and_legacy_tool_name(tmp_path, monkeypatch):
    import json
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path, monkeypatch)
    wb = ResearchWorkbench(runtime, Background())
    state = harness._load(rid)
    view = wb.turn(rid)
    approval = view['approvals'][0]
    assert approval['tool_name'] == 'prototype.validate'
    assert approval['arguments']['execution_environment']['image_id'].startswith('sha256:')
    assert approval['context']['content']['source'].startswith('def build_model')
    saved = runtime.db.get_approval(state['pending']['approval'])['request']
    saved.pop('tool')
    with runtime.db.transaction() as connection:
        connection.execute('UPDATE approvals SET request_json=? WHERE approval_id=?',
                           (json.dumps(saved), state['pending']['approval']))
    assert wb.turn(rid)['approvals'][0]['tool_name'] == 'prototype.validate'
