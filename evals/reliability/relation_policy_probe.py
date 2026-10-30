"""Frozen audit-only diagnostic; labels are never sent to the model.

This is NOT QASPER end-to-end scoring. All treatments share unchanged source
passages and candidate claims; only an appended audit instruction varies.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from paperpilot_agentic.answer_audit import answer_then_audit, RELATION_POLICY
from paperpilot_agentic.audit_ablation import digest, frozen_inputs, score


def synthetic_cases():
    specs = [
        ('relation', 'Which language is used?',
         'The first corpus contains Spanish texts. The second corpus was presented at Aurora '
         'Conference and tokenized with LexiTools.',
         'The first corpus uses Spanish.',
         'The second corpus uses English because it was presented at Aurora Conference and tokenized with LexiTools.'),
        ('hardware', 'Describe the compute used.',
         'Training used two GPUs. The implementation uses ArrayEngine 2.1.',
         'Two GPUs were used for training.',
         'Each GPU had 80 GB of memory because the implementation uses ArrayEngine 2.1.'),
        ('causality', 'What does the experiment establish?',
         'System A uses module M with optimizer P and scores 72. System B uses module N '
         'with optimizer Q and scores 75. Both scores are on the same validation set.',
         'System B scores 75 on the validation set.',
         'Changing module M to N alone caused the score improvement from 72 to 75.'),
        ('scope', 'What evidence is there of robustness?',
         'The model reached F1 0.8 on the clean validation set. Noise levels were not part of this experiment.',
         'The reported clean-validation F1 is 0.8.',
         'The model is robust to unseen noise because its clean-validation F1 is 0.8.'),
    ]
    return [{'id': name, 'question': question,
             'evidence': [{'evidence_id': 'source', 'content': content}],
             'draft': {'facts': [{'id': f'f{i}', 'kind': 'SOURCE_FACT', 'text': text,
                                  'evidence_ids': ['source'], 'depends_on': []}
                                 for i, text in enumerate((positive, negative), 1)],
                       'direct_answer_id': 'f1', 'limitations': []},
             'expected': {'f1': 'SUPPORTED', 'f2': 'UNSUPPORTED'},
             'origin': 'synthetic diagnostic, assistant-authored labels'}
            for name, question, content, positive, negative in specs]


def freeze(snapshot):
    rows, draft = frozen_inputs(snapshot)
    if [fact['id'] for fact in draft['facts']] != ['f1', 'f2', 'f3']:
        raise ValueError('This diagnostic expects the recorded three-claim language case')
    cases = synthetic_cases() + [{
        'id': 'observed-language-overreach', 'question': snapshot['run']['question'],
        'evidence': rows, 'draft': draft,
        'expected': {'f1': 'SUPPORTED', 'f2': 'UNSUPPORTED', 'f3': 'UNSUPPORTED'},
        'origin': 'previously inspected public QASPER dev failure; debugging case, NOT holdout',
    }]
    return {'version': 1, 'cases': cases, 'policy': RELATION_POLICY,
            'planned_requests': 2 * len(cases), 'input_sha256': digest(cases),
            'scoring': 'assistant-authored entailment labels, not independent ground truth',
            'scope': 'frozen audit only; no writer/retrieval quality claim; no repairs/retries'}


def run(plan, model, output):
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (output / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    save('protocol.json', {**plan, 'model': getattr(model, 'model', None),
                          'temperature': getattr(model, 'temperature', None)})
    results, attempts, tokens = [], 0, 0
    treatment = plan.get('treatment_name', 'relation')
    for index, case in enumerate(plan['cases']):
        # Counterbalance request ordering, not outcome-dependent retries.
        for arm in (['baseline', treatment] if index % 2 == 0 else [treatment, 'baseline']):
            name = f'{index:02}-{arm}'
            class Meter:
                def complete(self, messages, **kwargs):
                    nonlocal attempts, tokens
                    request = [dict(item) for item in messages]
                    if arm == treatment:
                        request[0]['content'] += '\n' + plan['policy']
                    # Preserve the actual request but never credentials/headers.
                    save(name + '-request.json', request)
                    guard = kwargs.pop('before_attempt')
                    def counted():
                        nonlocal attempts
                        if attempts >= plan['planned_requests']:
                            raise RuntimeError('Diagnostic request cap')
                        guard()
                        attempts += 1
                        save(name + '-started.json', {'attempt': attempts})
                    start = time.monotonic()
                    try:
                        response = model.complete(request, before_attempt=counted, **kwargs)
                        tokens += response.total_tokens
                        save(name + '-response.json', {**asdict(response),
                             'seconds': time.monotonic() - start})
                        return response
                    except Exception as exc:
                        save(name + '-failure.json', {'error_type': type(exc).__name__,
                             'seconds': time.monotonic() - start, 'usage_unknown': True})
                        raise RuntimeError(type(exc).__name__) from None
            report = answer_then_audit(case['question'], case['evidence'], Meter(),
                frozen_draft=case['draft'], allow_repair=False, numeric_scope_mode='off',
                relation_policy=plan.get('base_relation_policy', False))
            result = {'arm': arm, 'case': case['id'], 'expected': case['expected'], 'report': report}
            if 'expected_direct' in case:
                result['expected_direct'] = case['expected_direct']
            save(name + '-result.json', result)
            results.append(result)
    summary = {'attempts': attempts, 'known_tokens': tokens, 'metrics': score(results),
               'limitations': plan['scoring'] + '; ' + plan['scope']}
    if any('expected_direct' in result for result in results):
        summary['direct_answer_metrics'] = direct_answer_metrics(results)
    save('summary.json', summary)
    return summary


def direct_answer_metrics(results):
    metrics = {}
    for result in results:
        if 'expected_direct' not in result:
            continue
        counts = metrics.setdefault(result['arm'], dict(total=0, correct=0, failed=0,
                    false_direct=0, missed_direct=0, expected_direct=0, expected_not_direct=0))
        report = result['report']
        assessments = {row['fact_id']: row for row in report.get('audit', {}).get('assessments', [])}
        for fid, expected in result['expected_direct'].items():
            counts['total'] += 1
            counts['expected_direct' if expected else 'expected_not_direct'] += 1
            row = assessments.get(fid)
            if report.get('status') == 'FAILED' or row is None:
                counts['failed'] += 1
                continue
            predicted = row.get('verdict') == 'SUPPORTED' and row.get('relevance') == 'DIRECT'
            counts['correct'] += predicted == expected
            counts['false_direct'] += predicted and not expected
            counts['missed_direct'] += expected and not predicted
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute-paid', action='store_true')
    args = p.parse_args()
    plan = freeze(json.loads(args.snapshot.read_text(encoding='utf-8-sig')))
    if not args.execute_paid:
        print(json.dumps({'planned_requests': plan['planned_requests'], 'fingerprint': plan['input_sha256']}))
        return
    from paperpilot_agentic.config import RuntimeSettings, resolve_provider_api_key
    from paperpilot_agentic.providers import OpenAICompatibleLLM
    settings = replace(RuntimeSettings.from_env(), provider='qwen',
        api_key=resolve_provider_api_key('qwen'), model='qwen-plus', thinking_enabled=False,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', temperature=0)
    print(json.dumps(run(plan, OpenAICompatibleLLM(settings, max_retries=0), args.output)))


if __name__ == '__main__':
    main()
