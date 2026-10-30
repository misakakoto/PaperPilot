"""Zero-call diagnostics on saved answer results; never rewrite live scores."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.answer_contracts import validate_draft
from paperpilot_agentic.answer_coverage import apply_final_coverage
from paperpilot_agentic.citation_bindings import quarantine_misbound_support
from paperpilot_agentic.reader_recovery import citation_revision_plan
from paperpilot_agentic.table_claims import apply_row_guard


def replay(path):
    raw = path.read_bytes()
    result = json.loads(raw)
    report = result['run']['metadata']['reader_report']
    if 'draft' not in report:
        return {'file': str(path.resolve()), 'result_sha256': hashlib.sha256(raw).hexdigest(),
                'actual_requests': 0, 'saved_outcome': result['run']['research_outcome'],
                'replay_status': 'NO_SAVED_DRAFT', 'failure_detail': report.get('failure_detail'),
                'live_scores_unchanged': True}
    draft, audit = report['draft'], report['audit']
    wanted = {s['evidence_id'] for s in report['source_spans']}
    evidence = [row for row in result['evidence'] if row['evidence_id'] in wanted]
    question = result['run']['question']
    history = result['run']['metadata'].get('answer_request_history', [])
    language_error = None
    try:
        validate_draft(deepcopy(draft), evidence, question=question, request_history=history,
                       table_rows=report.get('table_rows', []))
    except ValueError as exc:
        language_error = str(exc)
    _, coverage = apply_final_coverage(draft, audit, evidence, report.get('table_rows', []),
                                       report['request_contract'], request_history=history)
    _, binding = quarantine_misbound_support(report['raw_audit'], draft, report['source_spans'], question=question)
    _, rows = apply_row_guard(draft, report['raw_audit'], report.get('table_rows', []))
    diagnostic = {**report, 'citation_bindings': binding}
    plan = citation_revision_plan(diagnostic, evidence, question=question, request_history=history)
    return {'file': str(path.resolve()), 'result_sha256': hashlib.sha256(raw).hexdigest(),
            'run_id': result['run']['run_id'], 'actual_requests': 0,
            'saved_answer_scope': report['answer_scope'], 'saved_outcome': result['run']['research_outcome'],
            'draft_validation_error': language_error,
            'misbound_fact_ids': binding['changed_fact_ids'],
            'saved_row_rejections': report.get('row_bindings', {}).get('changed_fact_ids', []),
            'current_row_guard': rows,
            'coverage_diagnostic': coverage,
            'citation_revision_plan': plan,
            'requires_fresh_boundary_mapping': [row['requirement'] for row in audit.get('requirement_assessments', [])
                if not row['fact_ids'] and not row.get('gap') and not row.get('limitation_indices')],
            'factual_review': 'NOT_PERFORMED_BY_REPLAY', 'live_scores_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob('*-result.json'))
    paths = [p for p in paths if p.name != 'preflight-result.json']
    report = {'version': 'boundary-citation-replay-v2-source-rows', 'actual_requests': 0,
              'rows': [replay(p) for p in paths]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output), 'cases': len(paths), 'actual_requests': 0}))


if __name__ == '__main__':
    main()
