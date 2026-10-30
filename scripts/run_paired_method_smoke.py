"""Explicit host opt-in to a reviewed paired SNUNet diagnostic; no model API calls."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.paired_methods import prepare_comparison, enqueue_comparison
from paperpilot_agentic.research_notebook import ResearchNotebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--design-run', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'training_executed': False, 'api_calls': 0,
            'seeds': args.seeds, 'steps_per_variant': args.steps,
            'requires': 'Existing hash-bound READY_FOR_EXECUTION_REVIEW host review, then --execute',
            'scope': 'paired exploratory diagnostic; no non-inferiority or scientific acceptance'}))
        return
    workspace = args.workspace.resolve(strict=True)
    if not (workspace/'runtime.sqlite3').is_file():
        raise ValueError('Use the original reviewed design workspace')
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=workspace/'runtime.sqlite3')
    binding = prepare_comparison(runtime, args.design_run, seeds=args.seeds, steps=args.steps)
    output = Path(tempfile.mkdtemp(prefix='paired-method-smoke-', dir=workspace))
    rid = enqueue_comparison(runtime, binding)['run']['run_id']
    waiting = runtime.resume(rid)
    if waiting['run']['status'] != 'WAITING_APPROVAL' or len(waiting['pending_approvals']) != 1:
        raise RuntimeError('Expected a separate runtime execution approval')
    runtime.approve(waiting['pending_approvals'][0]['approval_id'], actor='host-cli',
        comment='Explicit --execute for the frozen paired protocol; reviewed host-local sources, diagnostics only')
    result = runtime.resume(rid)
    with (output/'run.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    before = runtime.db.list_tool_calls(rid)
    replay = runtime.resume(rid)
    book = ResearchNotebook(runtime.db, args.design_run)
    book.import_experiment_run(rid)
    summary = {'output': str(output), 'binding_id': binding, 'run_id': rid,
        'status': result['run']['status'], 'error': result['run'].get('error'), 'api_calls': 0,
        'tool_attempts': [c['attempt'] for c in runtime.db.list_tool_calls(rid)],
        'replay_same_calls': before == runtime.db.list_tool_calls(rid),
        'replay_same_artifacts': result['artifacts'] == replay['artifacts'],
        'comparison': result['run']['metadata'].get('method_comparison')}
    with (output/'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if result['run']['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
