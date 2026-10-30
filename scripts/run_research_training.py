"""Host CLI for an already reviewed draft; runtime APIs accept only its binding ID.

No cloud calls. Prepare a new protocol before --approve-plan can record the user's
explicit eight-hour authorization. Source DB is read-only; work uses a SQLite backup.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paperpilot_agentic.orchestrator import ResearchAgentRuntime  # noqa: E402
from paperpilot_agentic.paired_methods import enqueue_comparison  # noqa: E402
from paperpilot_agentic.paired_training import campaign  # noqa: E402
from paperpilot_agentic.training_profiles import describe_training, prepare_training  # noqa: E402
from paperpilot_agentic.training_state import atomic_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--source-db')
    parser.add_argument('--db', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--design-run')
    parser.add_argument('--binding')
    parser.add_argument('--approve-plan', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    target = Path(args.db).resolve()
    if args.source_db:
        source = Path(args.source_db).resolve(strict=True)
        if source == target:
            parser.error('Use a separate DB; never write the historic source database')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(source.as_uri()+'?mode=ro', uri=True) as original, sqlite3.connect(target) as destination:
                original.backup(destination)
    runtime = ResearchAgentRuntime(workspace=args.workspace, db_path=target)
    record_path = target.parent/'campaign.json'
    record = json.loads(record_path.read_bytes()) if record_path.exists() else {}
    if args.prepare:
        if not args.dataset or not args.design_run:
            parser.error('--prepare requires host-selected --dataset and --design-run')
        binding = prepare_training(runtime, args.design_run, dataset=args.dataset)
        if record and record['binding_id'] != binding:
            parser.error('Campaign already frozen; keep previous record and choose a separate report directory')
        card = describe_training(runtime, binding)
        state = enqueue_comparison(runtime, binding)
        record = {'binding_id': binding, 'run_id': state['run']['run_id'], 'card': card}
        atomic_json(record_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    binding = args.binding or record.get('binding_id')
    if args.approve_plan or args.resume or args.status:
        if not binding:
            parser.error('Prepare a reviewed binding first')
        state = enqueue_comparison(runtime, binding)
        run_id = state['run']['run_id']
        if args.approve_plan:
            for approval in state['pending_approvals']:
                runtime.approve(approval['approval_id'], actor='codex-under-user-plan',
                    comment='User explicitly accepted frozen v2 local experiments, cumulative GPU cap eight hours, on 2026-10-04')
        if args.resume:
            state = runtime.resume(run_id)
        print(json.dumps({'run_id': run_id, 'status': state['run']['status'],
            'error': state['run'].get('error'), 'training_state': state['run']['metadata'].get('training_state'),
            'budget': campaign(runtime).snapshot()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
