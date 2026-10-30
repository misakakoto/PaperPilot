"""Zero-provider replay of the saved comparison-loss and gap-display failure.

Reads unchanged receipts; writes a separate diagnostic, never upgrades live scores.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.answer_contracts import audited_presentation
from paperpilot_agentic.answer_isolation import validate_patch_operations


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def user_payload(request):
    return json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))


def response_payload(response):
    return json.loads(response['choices'][0]['message']['content'])


def replay(receipts):
    paths = {name: receipts / name for name in (
        'request-027.json', 'response-027.json', 'request-028.json', 'response-028.json')}
    patch_request = user_payload(load(paths['request-027.json']))
    patch = response_payload(load(paths['response-027.json']))
    audit_request = user_payload(load(paths['request-028.json']))
    audit = response_payload(load(paths['response-028.json']))
    original = {'facts': patch_request['read_only_source_premises'] +
                         patch_request['failed_facts_untrusted']}
    before = deepcopy(original)
    error = None
    try:
        validate_patch_operations(original, patch, {'errors': patch_request['contract_errors']})
    except ValueError as exc:
        error = str(exc)
    assert error and 'lost inference work' in error, 'Saved degraded patch was not rejected'
    draft = {key: audit_request[key] for key in ('facts', 'direct_answer_id', 'limitations')}
    legacy = audited_presentation(draft, deepcopy(audit))
    current = audited_presentation(draft, deepcopy(audit),
                                  question=audit_request['original_question'])
    assert any(raw in legacy['answer_units'] for raw in audit['missing_aspects'])
    assert all(raw not in current['answer_units'] for raw in audit['missing_aspects'])
    assert current['answer_scope'] == 'PARTIAL' and current['limitations']
    assert current['internal_gap_diagnostics']['missing_aspects'] == audit['missing_aspects']
    assert before == original
    return {
        'mode': 'OFFLINE_SAVED_RESPONSE_REPLAY', 'actual_provider_calls': 0,
        'source_receipt_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in paths.items()},
        'comparison_patch': {'status': 'REJECTED_BEFORE_REAUDIT', 'error': error,
                             'remaining_required_work': 'COMPARISON_NOT_REPAIRED_BY_THIS_REPLAY'},
        'gap_display': {'old_answer_units': legacy['answer_units'],
                        'new_answer_units': current['answer_units'],
                        'new_limitations': current['limitations'],
                        'new_answer_scope': current['answer_scope'],
                        'internal_gap_diagnostics': current['internal_gap_diagnostics']},
        'live_acceptance': 'NOT_RUN_NO_BUDGET_FOR_GENERATION_AND_AUDIT',
        'history_and_live_scores_unchanged': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.receipts_dir.resolve())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve the previous replay: use a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(output), 'actual_provider_calls': 0,
                      'comparison_patch': report['comparison_patch']['status'],
                      'answer_scope': report['gap_display']['new_answer_scope']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
