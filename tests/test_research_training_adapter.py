"""Protocol/asset tests, no generated code execution, data or GPU quality claim."""
import json
import sys

import pytest

from paperpilot_agentic import research_training as training
from paperpilot_agentic.reference_methods import encoded, sha
from paperpilot_agentic.training_state import file_sha, atomic_json
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from test_research_harness import SOURCE, FUSION, PROPOSAL


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    runtime = ResearchAgentRuntime(workspace=tmp_path/'work', db_path=tmp_path/'runtime.sqlite3')
    root = tmp_path/'base'
    root.mkdir()
    contents = {'baseline.py': SOURCE.encode(), 'LICENSE': b'fixture', 'support.py': b'# fixture',
        'training_state.py': b'# fixture', 'worker.py': b"ledger = state.GpuBudget(job['ledger'])\n"}
    for name, raw in contents.items():
        (root/name).write_bytes(raw)
    from paperpilot_agentic.training_profiles import protocol
    base = {'protocol': protocol(), 'request': {'samples': {'train': [], 'val': []}, 'device': 'cuda'},
        'python': sys.executable, 'python_sha256': file_sha(sys.executable),
        'environment': {'fixture': True}, 'files': {n: sha(v) for n, v in contents.items()}}
    recipe = {'training_binding': 'b'*64,
              'validation_scope': 'Previously used development subset, NOT an independent holdout'}
    monkeypatch.setattr(training, 'load_recipe', lambda *args, **kwargs: (recipe, root, base))
    card = training.freeze_candidate(runtime, 'run_fixture', 'a'*64, 1, PROPOSAL, FUSION)
    return runtime, root, base, card


def test_candidate_uses_new_authority_and_only_changes_gpu_cap_in_trusted_worker(frozen):
    runtime, base_root, base, card = frozen
    root, manifest, loaded = training.load_candidate(runtime, card['binding_id'])
    assert loaded == card
    assert manifest['version'] == 3
    assert manifest['protocol']['gpu_cap_seconds'] == 14400
    assert manifest['request']['expected_unused'] == []
    assert base['protocol']['gpu_cap_seconds'] == 28800
    assert (base_root/'worker.py').read_text() == "ledger = state.GpuBudget(job['ledger'])\n"
    assert 'cap_seconds=14400' in (root/'worker.py').read_text()
    assert (root/'baseline.py').read_bytes() == (base_root/'baseline.py').read_bytes()
    assert loaded['requires_approval'] is True


@pytest.mark.parametrize('asset', ['candidate.py', 'baseline.py', 'worker.py', 'request.json'])
def test_changed_candidate_source_or_protocol_is_rejected(frozen, asset):
    runtime, _, _, card = frozen
    root, _, _ = training.load_candidate(runtime, card['binding_id'])
    with (root/asset).open('ab') as stream:
        stream.write(b'\n# changed')
    with pytest.raises(ValueError, match='changed'):
        training.load_candidate(runtime, card['binding_id'])


def test_unknown_gpu_ledger_rejected_before_worker_launch(frozen):
    runtime, _, _, card = frozen
    ledger = training.gpu_ledger(runtime, 'run_fixture')
    ledger.reserve('unfinished', 50)
    with pytest.raises(training.TrainingBusy):
        training.execute_candidate(runtime, 'run_fixture', card['binding_id'], 'run_child',
                                   runner=lambda *args, **kwargs: pytest.fail('Unknown process relaunched'))
    assert ledger.snapshot()['reserved_seconds'] == 50


def test_card_tampering_is_rejected(frozen):
    runtime, _, _, card = frozen
    root, _, _ = training.load_candidate(runtime, card['binding_id'])
    altered = json.loads((root/'card.json').read_bytes())
    altered['hypothesis'] = {'fake': 'Do not silently approve another hypothesis'}
    atomic_json(root/'card.json', altered)
    with pytest.raises(ValueError, match='card'):
        training.load_candidate(runtime, card['binding_id'])
