import copy
import json
import pytest

from paperpilot_agentic.prototype_workspace import plan, complete
from test_prototype_workspace import waiting, tasks


@pytest.mark.parametrize('mode', ['adaptive', 'simple_loop'])
def test_review_receives_exact_metrics_without_replaying_raw_receipts(tmp_path, monkeypatch, mode):
    _, harness, rid = waiting(tmp_path, monkeypatch)
    state = harness._load(rid)
    state['mode'] = mode
    work = state['work']
    plan(work, tasks()+[{'id': 'review', 'goal': 'review exact implementation', 'capability': 'review',
                        'depends_on': ['design', 'code', 'check'], 'acceptance': 'consistent'}])
    code = work['tasks']['code']['output']
    raw = {'passed': True, 'validated_code_id': code, 'scientific_validity': 'NOT_VERIFIED',
           'execution': {'experiment_id': 'exact-execution', 'status': 'COMPLETED', 'exit_code': 0,
                         'metrics': {'parameters': 123, 'peak_rss_kib': 456,
                                     'checks': [{'shape': [1, 1, 64, 64], 'loss': 0.75}]},
                         'stderr_path': 'PRIVATE_HOST_PATH/'+'a'*8000}}
    complete(work, 'check', 'validate', raw)
    state['validation_feedback'] = {'code_id': code, 'result': raw}
    state['last_observation'] = raw
    before = copy.deepcopy(work['artifacts'])
    _, messages = harness._context(rid, state)
    wire = messages[-1]['content']
    payload = json.loads(wire)
    deps = payload['ready_contexts']['review']['dependencies']
    if payload.get('context_policy') == 'artifact-table-v1':
        deps = {i: payload['context_artifacts'][v['artifact_ref']] for i, v in deps.items()}
    receipt = next(a['content'] for a in deps.values() if a['kind'] == 'validate')
    assert receipt['validated_code_id'] == code
    assert receipt['execution']['metrics']['parameters'] == 123
    assert receipt['execution']['metrics']['checks'][0]['loss'] == 0.75
    assert 'PRIVATE_HOST_PATH' not in wire
    assert len(wire) < 10000
    assert work['artifacts'] == before  # Full original remains addressable by ID.
