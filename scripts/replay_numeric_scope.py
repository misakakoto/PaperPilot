"""Apply a deterministic scope guard to SAVED responses; zero provider calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.answer_audit import audited_presentation, evidence_spans  # noqa: E402
from paperpilot_agentic.audit_ablation import score  # noqa: E402
from paperpilot_agentic.direct_answer import prepare_evidence  # noqa: E402
from paperpilot_agentic.numeric_scope import VERSION, apply_scope_guard  # noqa: E402


def replay(saved):
    rows = prepare_evidence(saved['plan']['evidence'])
    before, after = [], []
    for result in saved['results']:
        original = result['report']
        if original.get('status') == 'FAILED' or not original.get('audit'):
            raise ValueError('Paired replay requires valid saved audit responses for every arm')
        if original['source_spans'] != evidence_spans(rows):
            raise ValueError('Frozen source spans have drifted')
        # Re-apply to the original model decision. If a prior report was
        # produced by an enforced runtime, `model_audit` preserves that choice.
        model_audit = original.get('model_audit') or original['audit']
        effective, diagnostics = apply_scope_guard(result['draft'], model_audit, rows)
        before.append(result)
        after.append({**result, 'report': {**original, 'model_audit': model_audit,
                     'audit': effective, 'numeric_scope': diagnostics,
                     **audited_presentation(result['draft'], effective)}})
    return {'protocol': VERSION, 'scope': 'OFFLINE_SAVED_OUTPUT_COUNTERFACTUAL_NOT_NEW_MODEL_EVAL',
            'additional_provider_attempts': 0, 'additional_tokens': 0,
            'before_metrics': score(before), 'after_metrics': score(after), 'results': after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.saved.read_bytes()
    report = replay(json.loads(raw.decode('utf-8-sig')))
    report['saved_response_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'additional_provider_attempts': 0, 'output': str(args.output),
                      'before': report['before_metrics'], 'after': report['after_metrics']}))


if __name__ == '__main__':
    main()
