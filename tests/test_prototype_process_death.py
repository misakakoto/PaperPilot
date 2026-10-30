"""Actual OS process termination; fixtures do not call APIs or execute models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from paperpilot_agentic.research_training import episode_folder
from test_prototype_workspace import Reader


@pytest.mark.parametrize('position', ['before_response_file', 'after_response_file', 'unknown_response'])
def test_real_process_death_releases_lock_and_preserves_dispatch_state(tmp_path, monkeypatch, position):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(root/'src'), str(root/'tests')]),
           'PYTHONUTF8': '1'}
    with (tmp_path/'child.log').open('wb') as log:
        child = subprocess.Popen([sys.executable, str(root/'tests/fixtures/prototype_process_child.py'),
                                  str(tmp_path), position], cwd=root, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            deadline = time.monotonic()+60
            while not (tmp_path/'barrier.json').is_file():
                assert child.poll() is None, (tmp_path/'child.log').read_text(encoding='utf-8')
                assert time.monotonic() < deadline, 'Child did not reach durable barrier'
                time.sleep(0.05)
            barrier = json.loads((tmp_path/'barrier.json').read_text(encoding='utf-8'))
            assert barrier['pid'] == child.pid and barrier['position'] == position
            child.kill()  # Only this exact Popen-owned fixture process.
            assert child.wait(timeout=10) != 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    reader = Reader()
    monkeypatch.setattr(reader, 'complete', lambda *a, **k: pytest.fail('Saved/unknown request must not resend'))
    runtime = ResearchAgentRuntime(workspace=tmp_path, db_path=tmp_path/'runtime.db', llm=reader)
    harness = PrototypeHarness(runtime)
    rid = barrier['run_id']
    before = runtime.db.get_run(rid)
    assert before['used_llm_calls'] == 1
    assert before['metadata']['research_active_since'] is not None
    snapshot = episode_folder(runtime, rid)/'turn-01-response.json'
    saved_bytes = snapshot.read_bytes() if snapshot.exists() else None
    assert snapshot.exists() is (position == 'after_response_file')
    result = harness.advance(rid, max_turns=1)
    state = harness._load(rid)
    after = runtime.db.get_run(rid)
    assert after['used_llm_calls'] == before['used_llm_calls']
    assert after['used_tokens'] == before['used_tokens']
    assert after['budget'] == before['budget'] and after['created_at'] == before['created_at']
    assert after['metadata']['research_active_since'] is None
    assert after['metadata']['research_active_seconds'] >= before['metadata']['research_active_seconds']
    if position == 'unknown_response':
        assert state['phase'] == 'UNKNOWN' and state['turn'] == 0
        assert runtime.llm_runtime.budget_snapshot(rid)['unknown_results'] == 1
        assert runtime.db.list_tool_calls(rid) == []
        assert not snapshot.exists()
    else:
        assert state['phase'] == 'READY' and state['turn'] == 1
        assert result['research_episode']['work']['tasks']
        assert snapshot.is_file()
        assert len(runtime.db.list_tool_calls(rid)) == 1
        assert after['used_steps'] == 1
        if saved_bytes is not None:
            assert snapshot.read_bytes() == saved_bytes
