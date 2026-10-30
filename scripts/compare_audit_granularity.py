"""Replay frozen whole/individual/atomic audits; dry-run unless --execute-paid.

Private snapshots, annotations and outputs belong in the ignored .paperpilot/
directory. Existing outputs are never resumed or overwritten. Start a new,
explicit experiment after an interruption; never silently replay a paid call.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.audit_ablation import build_plan, digest, frozen_inputs, run_plan  # noqa: E402
from paperpilot_agentic.direct_answer import prepare_evidence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--max-requests', type=int, default=0)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8-sig'))
    rows, draft = frozen_inputs(snapshot)
    question = snapshot.get('question') or snapshot['run']['question']
    if not args.annotations:
        if args.execute_paid:
            parser.error('Human annotations are required before any paid experiment')
        print(json.dumps({'input_sha256': digest({'question': question,
                         'evidence': prepare_evidence(rows), 'draft': draft}),
                         'fact_ids': [f['id'] for f in draft['facts']], 'paid_requests': 0}))
        return
    annotations = json.loads(args.annotations.read_text(encoding='utf-8-sig'))
    plan = build_plan(question, rows, draft, annotations)
    if not args.execute_paid:
        print(json.dumps({'input_sha256': plan['input_sha256'], 'paid_requests': 0,
                          'planned_max_requests': plan['planned_max_requests']}))
        return
    if not args.env_file or not args.output or args.max_requests < plan['planned_max_requests']:
        parser.error('Paid run requires env file, new output and a sufficient explicit request budget')
    if args.output.exists():
        raise FileExistsError('Preserve the old experiment; choose a new output')
    from dotenv import load_dotenv
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    load_dotenv(args.env_file, override=False)
    settings = RuntimeSettings.from_env(require_api_key=True)
    model = OpenAICompatibleLLM(settings, max_retries=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents concurrent jobs from replacing one another.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump({'status': 'RESERVED', 'provider_attempts': 0}, handle)

    def save(report):
        report['model'] = settings.model
        report['temperature'] = settings.temperature
        temporary = args.output.with_suffix(args.output.suffix + '.pending')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(args.output)

    report = run_plan(plan, model, max_requests=args.max_requests, on_progress=save)
    print(json.dumps({'status': report['status'], 'provider_attempts': report['provider_attempts'],
                      'known_tokens': report['known_tokens'], 'output': str(args.output)}))
    if report['status'] in {'FAILED', 'PARTIAL_FAILURE'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
