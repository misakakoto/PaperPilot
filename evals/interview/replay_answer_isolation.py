"""Zero-provider replay of a saved writer request/response and repair admission.

This diagnoses contracts and constructed input bytes, never scores factual truth
or supplies a fabricated audit. Original receipts and stopped ledgers are read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.answer_contracts import validate_draft
from paperpilot_agentic.answer_pipeline import (AnswerOptions, generate_draft, initialize,
    prepared_messages, repair_isolated_facts, writer_messages)
from paperpilot_agentic.llm_runtime import LLMBudgetExhausted
from paperpilot_agentic.request_packet import restore_value_references


class NoDispatch:
    def __init__(self):
        self.requests = []

    def ensure_capacity(self, requests):
        self.requests = requests
        raise LLMBudgetExhausted('Offline replay: no live authorization or dispatch')

    def complete(self, *args, **kwargs):
        raise AssertionError('Offline replay must never dispatch')


def replay(request_path, response_path):
    request_bytes = Path(request_path).read_bytes()
    response_bytes = Path(response_path).read_bytes()
    request, response = json.loads(request_bytes), json.loads(response_bytes)
    payload = next(json.loads(m['content']) for m in request['messages']
                   if m['role'] == 'user' and isinstance(json.loads(m['content']), dict)
                   and 'evidence' in json.loads(m['content']))
    payload = restore_value_references(payload)
    raw = json.loads(response['choices'][0]['message']['content'])
    model = NoDispatch()
    context = initialize(payload['question'], payload['evidence'], model,
        AnswerOptions(original_question=payload.get('original_question'),
                      request_history=payload.get('request_history'),
                      source_inventory=payload.get('source_inventory_navigation_only')), None)
    try:
        validate_draft(raw, context.rows, question=payload.get('original_question', payload['question']))
        before = 'VALID'
    except (ValueError, TypeError) as exc:
        before = str(exc)
    draft = generate_draft(context, lambda *args, **kwargs: deepcopy(raw))
    context.report['draft'] = draft
    if context.report.get('draft_isolation'):
        repair_isolated_facts(context, draft, None)
    assert Path(request_path).read_bytes() == request_bytes
    assert Path(response_path).read_bytes() == response_bytes
    return {
        'protocol': 'offline-answer-isolation-replay-v1',
        'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
        'response_sha256': hashlib.sha256(response_bytes).hexdigest(),
        'original_contract_result': before,
        'isolation': context.report.get('draft_isolation'),
        'eligible_fact_ids': [f['id'] for f in draft['facts']],
        'candidate_facts_unchanged': all(f in raw['facts'] for f in draft['facts']),
        'original_receipts_unchanged': True,
        'writer_constructed_message_utf8_bytes': len(json.dumps(
            prepared_messages(context, writer_messages(context)), ensure_ascii=False).encode()),
        'local_repair_workset': context.report.get('local_repair_workset'),
        'audit_floor_constructed_message_utf8_bytes': len(json.dumps(
            model.requests[1]['messages'], ensure_ascii=False).encode()) if len(model.requests) > 1 else None,
        'physical_provider_requests': 0, 'audit_status': 'NOT_RUN',
        'factual_correctness': 'NOT_EVALUATED', 'live_acceptance_unchanged': True,
        'cost_scope': 'constructed_inputs_without_runtime_countdown_not_paid_token_savings',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Outputs must never overwrite either preserved receipt, even via a symlink.
    if args.output.resolve() in {args.request.resolve(), args.response.resolve()}:
        parser.error('Output must be separate from the preserved receipts')
    result = replay(args.request, args.response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('eligible_fact_ids', 'physical_provider_requests',
        'candidate_facts_unchanged', 'audit_status', 'factual_correctness')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
