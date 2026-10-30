"""Maintained fixture held at a durable boundary for the owning test to kill."""
import json
import os
from pathlib import Path
import sys
import time

from paperpilot_agentic import episode_runner
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.prototype_harness import PrototypeHarness
from test_prototype_workspace import Reader

folder, position = Path(sys.argv[1]), sys.argv[2]
reader = Reader()
runtime = ResearchAgentRuntime(workspace=folder, db_path=folder/'runtime.db', llm=reader)
runtime.db.list_evidence = lambda _: [{'evidence_id': 'source', 'content': 'maintained source'}]
harness = PrototypeHarness(runtime)
rid = harness.start('build', 'process-death-fixture', 'parent', ['source'])['run']['run_id']


def hold():
    temp = folder/'barrier.tmp'
    with temp.open('w', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'run_id': rid, 'position': position}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, folder/'barrier.json')
    while True:
        time.sleep(0.1)


original = episode_runner.immutable


def snapshot(boundary_folder, name, raw):
    if name.endswith('-response.json'):
        if position == 'after_response_file':
            original(boundary_folder, name, raw)
        hold()
    return original(boundary_folder, name, raw)


original_complete = reader.complete


def complete(messages, *, before_attempt=None, **kwargs):
    # Exercise the production providers' dispatch hook, not Reader's legacy
    # return-time fallback accounting. A killed request must already be held.
    if before_attempt is not None:
        before_attempt()
    if position == 'unknown_response':
        hold()
    return original_complete(messages, **kwargs)


reader.complete = complete
if position != 'unknown_response':
    episode_runner.immutable = snapshot
harness.advance(rid, max_turns=1)
raise AssertionError('Parent must kill this fixture at the barrier')
