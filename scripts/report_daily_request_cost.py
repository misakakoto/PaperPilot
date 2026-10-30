"""Zero-provider-call replay: compare complete saved requests and value bindings."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# scripts/paperpilot_agentic.py must not shadow the active source package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.context_window import prompt_upper_bound
from paperpilot_agentic.request_packet import compact_payload, restore_value_references, INSTRUCTION, VERSION
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.llm_runtime import PersistentLLMRuntime
from paperpilot_agentic.models import Budget
from paperpilot_agentic.context_window import enforce_window


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def replay(request, *, prepared_requests=None):
    before = deepcopy(request['messages'])
    after = deepcopy(before)
    changes = []
    for message in after:
        if message['role'] != 'user':
            continue
        try:
            payload = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or 'question' not in payload:
            continue
        compact = compact_payload(payload)
        restored = restore_value_references(compact)
        fields = [field for field in ('evidence', 'sources', 'facts', 'request_history', 'request_contract',
                   'table_rows', 'source_roles', 'original_question', 'question') if field in payload]
        for field in fields:
            if payload[field] != restored.get(field):
                raise AssertionError(f'Protected value changed: {field}')
        message['content'] = json.dumps(compact, ensure_ascii=False, separators=(',', ':'))
        changes.append({'protected_sha256': {field: digest(payload[field]) for field in fields},
                        'navigation_before': len(payload.get('source_inventory_navigation_only', [])),
                        'navigation_after': len(compact.get('source_inventory_navigation_only', [])),
                        'cell_references': sum('cells_source_ref' in row for row in compact.get('table_rows', []))})
    if changes:
        after[0]['content'] = INSTRUCTION.lstrip() + '\n' + after[0]['content']
    output = request['max_tokens']
    if prepared_requests is not None:
        # A saved SDK request already has its old advisory budget. The current
        # preflight reconstructs that field from the new ledger, exactly once.
        prepared_requests.append({'messages': [m for m in after if not (
            m['role'] == 'system' and m['content'].startswith('[PAPERPILOT_RUNTIME_BUDGET]'))],
            'max_tokens': output})
    return {'before_request_bytes': len(json.dumps(before, ensure_ascii=False).encode()),
            'after_request_bytes': len(json.dumps(after, ensure_ascii=False).encode()),
            'before_input_output_upper': prompt_upper_bound(before) + output,
            'after_input_output_upper': prompt_upper_bound(after) + output,
            'equivalence': 'PASS', 'changes': changes, 'actual_new_usage': 'NOT_RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--request-ids', nargs='+', type=int, default=[10, 11])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    prepared_requests = []
    actual_old = 0
    for number in args.request_ids:
        path = args.receipts / f'request-{number:03}.json'
        row = {'request_id': number, 'original_request_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               **replay(json.loads(path.read_text(encoding='utf-8')), prepared_requests=prepared_requests)}
        response_path = args.receipts / f'response-{number:03}.json'
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding='utf-8'))
            usage = response.get('usage') or {}
            if usage.get('total_tokens') is not None:
                actual_old += int(usage['total_tokens'])
        results.append(row)
    result = {'protocol': VERSION, 'provider_calls': 0, 'results': results,
        'before_pair_upper': sum(r['before_input_output_upper'] for r in results),
        'after_pair_upper': sum(r['after_input_output_upper'] for r in results),
        'before_pair_bytes': sum(r['before_request_bytes'] for r in results),
        'after_pair_bytes': sum(r['after_request_bytes'] for r in results),
        'historical_confirmed_tokens_if_present': actual_old,
        'daily_cap': 200000, 'actual_new_usage': 'NOT_RUN', 'answer_quality': 'NOT_EVALUATED'}
    result['request_byte_reduction'] = 1 - result['after_pair_bytes'] / result['before_pair_bytes']
    with TemporaryDirectory(prefix='paperpilot-cost-admission-') as folder:
        db = RuntimeDatabase(Path(folder) / 'runtime.sqlite3')
        rid = db.create_run('Zero-call saved-request admission.', Budget(max_tokens=200000))
        runtime = PersistentLLMRuntime(db, None)
        runtime.ensure_capacity(rid, prepared_requests, reserve_class='synthesis')
        current = [runtime._with_budget(r['messages'], runtime.budget_snapshot(rid)) for r in prepared_requests]
        for messages, request in zip(current, prepared_requests):
            enforce_window(messages, request['max_tokens'], limit=131072)
        result['current_runtime_admission'] = {'status': 'PASS',
            'pair_upper': sum(prompt_upper_bound(m) + r['max_tokens']
                              for m, r in zip(current, prepared_requests)),
            'physical_requests': db.get_run(rid)['used_llm_calls']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
