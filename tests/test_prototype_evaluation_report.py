import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paperpilot_agentic.prototype_workspace import complete, digest, plan
from test_prototype_completion import finished

spec = importlib.util.spec_from_file_location('prototype_report', Path(__file__).parents[1]/'scripts/report_research_prototypes.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_engineering_pass_does_not_attest_model_review_or_research_value():
    _, state, _ = finished()
    state.update(phase='COMPLETED', completed=True)
    state['work']['repairs'] = 2
    result = report.assess_state(state)
    assert result['engineering'] == 'PASSED_CURRENT'
    assert result['code_repairs'] == 1
    assert result['requirements_satisfied'] == 'NOT_ESTABLISHED_BY_ENGINEERING_PASS'
    assert result['model_review_facts'] == 'NOT_HOST_VERIFIED'
    assert result['scientific_validity'] == 'NOT_VERIFIED'


def test_changed_code_keeps_old_receipt_historical_without_counting_pass():
    _, state, rows = finished()
    state['phase'] = 'READY'
    old_check = state['work']['tasks']['check']['output']
    rows[2]['goal'] = 'replace implementation'
    plan(state['work'], rows)
    complete(state['work'], 'code', 'implement', {'source': 'new version', 'config': {}})
    result = report.assess_state(state)
    assert result['engineering'] != 'PASSED_CURRENT'
    assert old_check in result['historical_validation_ids']


@pytest.mark.parametrize('field', ['validated_code_id', 'validation_contract'])
def test_wrong_code_or_contract_receipt_cannot_count_as_current_pass(field):
    _, state, _ = finished()
    state['phase'] = 'READY'
    work = state['work']
    check = work['tasks']['check']['output']
    artifact = copy.deepcopy(work['artifacts'][check])
    artifact['content'][field] = 'different version'
    new_id = digest(artifact)
    work['artifacts'][new_id] = artifact
    work['tasks']['check']['output'] = new_id
    work['tasks']['review'].update(status='PENDING', output=None)
    assert report.assess_state(state)['engineering'] == 'FAILED_OR_STALE'


def test_tampered_artifact_stops_report_instead_of_silently_counting_it():
    _, state, _ = finished()
    state['phase'] = 'READY'
    state['work']['artifacts'][state['work']['tasks']['check']['output']]['content']['passed'] = False
    with pytest.raises(ValueError, match='content ID'):
        report.assess_state(state)


def test_completed_task_cannot_keep_receipt_bound_to_old_dependency():
    _, state, _ = finished()
    state['phase'] = 'READY'
    work = state['work']
    code = copy.deepcopy(work['artifacts'][work['tasks']['code']['output']])
    code['content']['source'] = 'different implementation'
    identity = digest(code)
    work['artifacts'][identity] = code
    work['tasks']['code']['output'] = identity
    with pytest.raises(ValueError, match='dependency version'):
        report.assess_state(state)


def test_report_uses_database_after_generation_ledger_stopped(tmp_path, monkeypatch):
    from test_prototype_workspace import waiting
    runtime, harness, rid = waiting(tmp_path/'runtime', monkeypatch)
    state = harness._load(rid)
    run = runtime.db.get_run(rid)
    case = {'id': 'case', 'question': state['goal'],
            'constraints': {k: v for k, v in state['validation_contract'].items() if k != 'kind'}}
    job = {'id': 'job', 'case_id': 'case', 'mode': 'adaptive', 'repeat': 1}
    manifest = {'campaign': 'fixture', 'model': runtime.llm_runtime.model.model,
                'cases': [case], 'jobs': [job, {**job, 'id': 'not-started'}], 'source_files': {}}
    (tmp_path/'frozen-source/src').mkdir(parents=True)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (tmp_path/'ledger.json').write_text(json.dumps({'parent_run_id': 'parent', 'source_ids': ['source'],
        'jobs': [{'run_id': rid, 'phase': 'COMPLETED'}]}), encoding='utf-8')
    request = {'goal': state['goal'], 'parent': 'parent', 'sources': ['source'], 'conversation_id': None,
               'mode': 'adaptive', 'profile_id': None, 'validation_contract': state['validation_contract']}
    # Copy a maintained real SQLite database, with this fixture's submission key.
    import sqlite3
    with sqlite3.connect(tmp_path/'runtime.sqlite3') as target, runtime.db.connect() as source:
        source.backup(target)
        target.execute('UPDATE research_episodes SET request_key=?,fingerprint=? WHERE run_id=?',
                       ('fixture-job', digest(request), rid))
    before = (tmp_path/'runtime.sqlite3').read_bytes()
    result = report.campaign_report(tmp_path)
    assert result['rows'][0]['phase'] == 'WAITING_APPROVAL'
    assert result['rows'][0]['used_llm_calls'] == run['used_llm_calls']
    assert result['rows'][0]['execution_attempts'] == 0
    assert result['rows'][0]['other_human_interventions'] is None
    assert result['rows'][1]['phase'] == 'NOT_STARTED'
    output = tmp_path/'cli-report.json'
    process = subprocess.run([sys.executable, str(Path(__file__).parents[1]/'scripts/report_research_prototypes.py'),
                              str(tmp_path), '--output', str(output)], capture_output=True, text=True,
                             timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text(encoding='utf-8'))['rows'][0]['phase'] == 'WAITING_APPROVAL'
    assert (tmp_path/'runtime.sqlite3').read_bytes() == before
