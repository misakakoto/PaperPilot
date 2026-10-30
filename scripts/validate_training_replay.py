"""Verify real API replay against the existing campaign, without starting a new job."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime  # noqa: E402
from paperpilot_agentic.paired_methods import enqueue_comparison  # noqa: E402
from paperpilot_agentic.paired_training import campaign  # noqa: E402
from paperpilot_agentic.training_state import atomic_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--db', required=True)
    args = parser.parse_args()
    target = Path(args.db).parent
    record = json.loads((target/'campaign.json').read_bytes())
    runtime = ResearchAgentRuntime(workspace=args.workspace, db_path=args.db)
    rid, binding = record['run_id'], record['binding_id']
    before = campaign(runtime).snapshot()
    run = runtime.db.get_run(rid)
    active = [r for r in before['launches'] if r['status'] == 'RUNNING']
    if run['status'] != 'COMPLETED' and not active:
        raise ValueError('Replay only while an owned worker is running or the run is completed')
    counts = {'launches': len(before['launches']), 'tool_calls': len(runtime.db.list_tool_calls(rid)),
              'llm_calls': len(runtime.db.list_llm_calls(rid))}
    repeated = enqueue_comparison(runtime, binding)
    assert repeated['run']['run_id'] == rid
    restored = ResearchAgentRuntime(workspace=args.workspace, db_path=args.db).resume(rid)
    after = campaign(runtime).snapshot()
    current = {'launches': len(after['launches']), 'tool_calls': len(runtime.db.list_tool_calls(rid)),
               'llm_calls': len(runtime.db.list_llm_calls(rid))}
    if counts != current:
        raise ValueError('Counts changed during replay; keep the record and inspect background progress before attributing it')
    report = {'binding_id': binding, 'run_id': rid, 'before': counts, 'after': current,
        'status': restored['run']['status'], 'same_run_id': True,
        'completed_response_reused': run['status'] == 'COMPLETED',
        'running_worker_blocked': bool(active), 'new_gpu_launches': 0, 'new_llm_calls': 0}
    atomic_json(target/('replay-completed.json' if run['status'] == 'COMPLETED' else 'replay-running.json'), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
