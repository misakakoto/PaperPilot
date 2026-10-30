"""Measure constructed inputs for an explicit offline selection, not model quality."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.answer_pipeline import (AnswerOptions, initialize, prepared_messages,
                                               writer_messages, audit_floor_request)
from paperpilot_agentic.initial_evidence_read import (mandatory_ids, packet_bytes,
    resolve_selection, selection_messages)
from paperpilot_agentic.request_packet import restore_value_references


def replay(path, ids):
    original = path.read_bytes()
    request = json.loads(original)
    payload = next(json.loads(m['content']) for m in request['messages']
                   if m['role'] == 'user' and 'evidence' in json.loads(m['content']))
    payload = restore_value_references(payload)
    context = initialize(payload['question'], payload['evidence'], None,
        AnswerOptions(original_question=payload.get('original_question'),
                      request_history=payload.get('request_history'),
                      source_inventory=payload.get('source_inventory_navigation_only'),
                      requested_tables=payload.get('request_contract', {}).get('table_requirements'),
                      evidence_read_policy='navigation-v1'), None)
    required = mandatory_ids(context.rows, context.report['table_rows'], context.report['request_contract'])
    selected = resolve_selection({'evidence_ids': ids or required}, context.rows, required, 24000)
    narrowed = dataclasses.replace(context, writer_rows=selected)
    def size(messages):
        return len(json.dumps(messages, ensure_ascii=False).encode())
    full_writer = size(prepared_messages(context, writer_messages(context)))
    read_writer = size(prepared_messages(narrowed, writer_messages(narrowed)))
    selection = size(prepared_messages(context, selection_messages(context, required)))
    audit = size(audit_floor_request(context)['messages'])
    assert audit_floor_request(narrowed) == audit_floor_request(context)
    assert path.read_bytes() == original
    return {'protocol': 'offline-initial-read-sizing-v1',
        'request_sha256': hashlib.sha256(original).hexdigest(), 'original_receipt_unchanged': True,
        'scenario': 'HOST_REQUIRED_SOURCES_PLUS_EXPLICIT_IDS_NOT_A_MODEL_SELECTION',
        'required_evidence_ids': required, 'selected_evidence_ids': [r['evidence_id'] for r in selected],
        'candidate_count': len(context.rows), 'read_count': len(selected),
        'full_source_utf8_bytes': packet_bytes(context.rows), 'read_source_utf8_bytes': packet_bytes(selected),
        'constructed_input_bytes': {'full_writer': full_writer, 'selection': selection,
            'selected_writer': read_writer, 'all_candidate_audit_floor': audit,
            'full_two_stage_floor': full_writer + audit,
            'selection_three_stage_floor': selection + read_writer + audit},
        'scope': 'NO_RUNTIME_COUNTDOWN_NO_GENERATED_FACTS_NO_REPAIR_NO_PRESENTATION',
        'paid_token_savings': 'NOT_MEASURED', 'semantic_coverage': 'NOT_PROVEN',
        'factual_review': 'NOT_RUN', 'physical_provider_requests': 0, 'live_acceptance_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--evidence-id', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.request.resolve():
        parser.error('Output must be separate from the preserved receipt')
    result = replay(args.request, args.evidence_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('candidate_count', 'read_count',
        'constructed_input_bytes', 'physical_provider_requests', 'semantic_coverage')}))


if __name__ == '__main__':
    main()
