"""Answer/audit diagnostic with at most one repair on frozen retrieved evidence."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dotenv import load_dotenv  # noqa: E402

from paperpilot_agentic.answer_audit import answer_then_audit  # noqa: E402
from paperpilot_agentic.audit_ablation import frozen_inputs  # noqa: E402
from paperpilot_agentic.config import RuntimeSettings  # noqa: E402
from paperpilot_agentic.providers import OpenAICompatibleLLM  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--execute-paid', action='store_true')
    parser.add_argument('--freeze-saved-draft', action='store_true',
                        help='Audit the saved candidate without generating a new answer')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; preserve previous attempts')
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8-sig'))
    question = snapshot.get('question') or snapshot['run']['question']
    frozen_draft = None
    evidence = snapshot['evidence']
    if args.freeze_saved_draft:
        evidence, frozen_draft = frozen_inputs(snapshot)
    if not args.execute_paid:
        print(json.dumps({'question': question, 'paid_requests': 0,
                          'evidence_count': len(evidence)}))
        return
    load_dotenv(args.env_file, override=False)
    settings = RuntimeSettings.from_env(require_api_key=True)
    model = OpenAICompatibleLLM(settings, max_retries=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        payload = {**report, 'source_snapshot': str(args.snapshot),
                   'model': settings.model,
                   'source_run_id': snapshot.get('run', {}).get('run_id'),
                   'evaluation_scope': 'FROZEN_RETRIEVED_EVIDENCE_NOT_END_TO_END'}
        temp = args.output.with_suffix('.pending')
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(args.output)

    result = answer_then_audit(question, evidence, model, on_progress=save,
                               frozen_draft=frozen_draft)
    print(json.dumps({'status': result['status'], 'attempts': result['provider_attempts'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
