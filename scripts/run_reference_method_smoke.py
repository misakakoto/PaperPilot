"""Opt-in, host-authorized reference method smoke. No LLM, downloads or installations."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime
from paperpilot_agentic.reference_methods import prepare_snunet, enqueue_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='Pinned public source/license JSON from design smoke')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--execute', action='store_true', help='Explicitly approve this bounded local diagnostic')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'training_executed': False, 'api_calls': 0,
            'next': 'Use --execute to approve pinned source, copied train/val samples and 1-4 optimizer steps; no hard sandbox.'}))
        return
    output_root = Path(__file__).resolve().parents[1] / 'outputs'
    output_root.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='reference-method-smoke-', dir=output_root))
    binding = prepare_snunet(workspace, snapshot=args.snapshot, dataset=args.dataset,
        python=args.python, steps=args.steps, seed=args.seed, device=args.device)
    runtime = ResearchAgentRuntime(workspace=workspace, db_path=workspace/'runtime.sqlite3')
    run_id = enqueue_validation(runtime, binding)['run']['run_id']
    waiting = runtime.resume(run_id)
    if waiting['run']['status'] != 'WAITING_APPROVAL' or len(waiting['pending_approvals']) != 1:
        raise RuntimeError('Expected explicit runtime approval before execution')
    runtime.approve(waiting['pending_approvals'][0]['approval_id'], actor='host-cli',
        comment='Explicit --execute approval for fixed profile; local-only diagnostic, no scientific claim')
    result = runtime.resume(run_id)
    with (workspace/'run.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    summary = {'workspace': str(workspace), 'run_id': run_id, 'status': result['run']['status'],
        'error': result['run'].get('error'), 'api_calls': 0,
        'validation': result['run']['metadata'].get('method_validation')}
    with (workspace/'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if result['run']['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
