"""Report a real durable reservation as RUNNING; do not confuse it with NOT_RUN."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from paperpilot_agentic.training_state import atomic_json, GpuBudget
from paperpilot_agentic.training_profiles import protocol


def test_active_launch_is_visible_without_inventing_verified_metrics(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts'/'report_research_training.py'
    spec = importlib.util.spec_from_file_location('training_report_fixture', path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    workspace = tmp_path/'work'
    folder = workspace/'method-training'/'runs'/'run-fixed'
    trial = folder/'trials'/'7-candidate'
    (trial/'checkpoints').mkdir(parents=True)
    atomic_json(trial/'checkpoints'/'latest.json', {'step': 25})
    atomic_json(folder/'state.json', {'checks': {}, 'trials': {}})
    ledger = GpuBudget(workspace/'method-training'/'gpu-budget.sqlite3')
    token, _ = ledger.reserve(str(trial), 60)
    ledger.attach(token)  # Synthetic reservation only; no CUDA or subprocess.
    manifest = {'protocol': protocol(), 'environment': {'fixture': True}, 'files': {
        'baseline.py': 'b', 'candidate.py': 'c', 'worker.py': 'w'}, 'python_sha256': 'p',
        'review_sha256': 'r', 'design_run_id': 'd', 'baseline_binding_id': 'old', 'splits': {}, 'excluded_ids': ['old-id']}
    monkeypatch.setattr(report, 'load_training', lambda *_: (tmp_path, manifest))
    db = SimpleNamespace(get_run=lambda _: {'status': 'RUNNING'}, list_llm_calls=lambda _: [])
    data = report.facts(SimpleNamespace(workspace=workspace, db=db), {'binding_id': 'bound', 'run_id': 'run-fixed'})
    assert data['trials'][0]['candidate'] == {'status': 'RUNNING', 'committed_step': 25}
    assert data['verified_trials'] == 0 and data['trials'][0]['baseline']['status'] == 'NOT_RUN'
    assert 'RUNNING' in report.page(data)
    assert all(v is False for v in data['claims'].values())
