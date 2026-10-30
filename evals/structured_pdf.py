"""Frozen local PDF-region evaluation. No LLM/provider entry point exists here."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from paperpilot_agentic.docling_adapter import DoclingEvidenceAdapter
from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion, config_sha, file_sha

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'outputs/structured-pdf-20261002'

# Coordinates were selected from original PDF renders, before parser output.
TABLES = [
    (0, 16, 'XII', [50, 193, 300, 325], True),
    (1, 11, 'VII', [48, 270, 565, 435], True),
    (0, 9, 'I', [60, 215, 553, 322], False),
    (0, 9, 'II', [48, 342, 303, 490], False),
    (0, 16, 'XI', [96, 55, 515, 175], False),
    (0, 17, 'XIII', [151, 55, 455, 165], False),
    (0, 13, 'VIII', [324, 523, 535, 710], False),
    (1, 6, 'I', [48, 383, 303, 555], False),
    (1, 6, 'II', [321, 383, 552, 480], False),
    (1, 11, 'VIII', [48, 445, 565, 610], False),
]
FORMULAS = [
    (0, 7, 'inline-token-exponent', [48, 268, 302, 298], True),
    (0, 4, '1', [365, 72, 565, 108], False),
    (0, 4, '2', [365, 222, 565, 257], False),
    (0, 4, '3', [365, 286, 565, 320], False),
    (1, 4, '2', [365, 270, 565, 286], False),
    (1, 4, '3', [365, 286, 565, 302], False),
    (1, 5, '9', [365, 170, 565, 186], False),
    (1, 5, '10', [365, 302, 565, 320], False),
    (1, 7, '11', [75, 393, 302, 424], False),
    (1, 7, '13', [75, 577, 302, 611], False),
]


def save_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def freeze():
    import fitz
    from paperpilot_agentic.config import RuntimeSettings
    from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
    from paperpilot_retrieval import PaperKnowledgeStore
    from evals.interview.cases import CHANGE, FLOW
    settings = RuntimeSettings.from_env()
    adapter = PaperPilotKnowledgeSearchAdapter(PaperKnowledgeStore(settings.memory_db,
        namespace=settings.memory_namespace), paper_roots=settings.paper_roots)
    FOLDER.mkdir(parents=True, exist_ok=True)
    if (FOLDER / 'cases.json').exists():
        raise ValueError('Cases already frozen; do not overwrite or retune holdouts')
    documents = [adapter.resolve_pdf_paths({sha})[0] for sha in [CHANGE, FLOW]]
    cases = []
    for kind, specs in [('table', TABLES), ('equation', FORMULAS)]:
        for index, (paper, page, label, box, known) in enumerate(specs):
            cid = f'{kind}-{index + 1:02d}'
            crop = FOLDER / f'{cid}.png'
            with fitz.open(documents[paper]) as doc:
                doc[page - 1].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(box)).save(crop)
            region = SourceRegion(paper_sha256=file_sha(documents[paper]), object_id=cid,
                label=f'Table {label}' if kind == 'table' else f'Equation {label}',
                page=page, bbox=box, crop_sha256=file_sha(crop))
            cases.append({'id': cid, 'kind': kind, 'known_failure': known,
                'pdf_path': documents[paper], 'crop': str(crop), 'source': region.model_dump(),
                'review_dimensions': ['all_cells_and_conditions', 'spans', 'source_location'] if kind == 'table'
                    else ['all_symbols', 'subscripts_exponents', 'denominator', 'source_location']})
    save_new(FOLDER / 'cases.json', {'cases': cases, 'case_sha256': config_sha(cases),
        'reference': 'Original PDF crops, not parser output', 'reviewer': 'Independent original-PDF review required'})
    print(json.dumps({'frozen_tables': 10, 'frozen_formula_regions': 10}))


def execute(case_ids=None):
    cases = json.loads((FOLDER / 'cases.json').read_text(encoding='utf-8'))
    adapter = DoclingEvidenceAdapter.from_env()
    code = {name: file_sha(ROOT / 'src/paperpilot_agentic' / name) for name in
            ['docling_worker.py', 'docling_adapter.py', 'structured_extraction.py']}
    identity = {'cases': cases['case_sha256'], 'parser': adapter.identity, 'code': code}
    folder = FOLDER / f'run-{config_sha(identity)[:16]}'
    folder.mkdir(exist_ok=True)
    if not (folder / 'identity.json').exists():
        save_new(folder / 'identity.json', identity)
    for case in cases['cases']:
        if case_ids and case['id'] not in case_ids:
            continue
        target = folder / f'{case["id"]}.json'
        if target.exists():
            continue
        started = time.perf_counter()
        try:
            result = adapter.analyze(case['crop'], '', {'object_type': case['kind'],
                'pdf_path': case['pdf_path'], 'source_region': case['source']})
            record = {'status': 'PARSED', 'result': result}
        except Exception as exc:
            record = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}',
                      'local_parse_attempts': getattr(exc, 'local_parse_attempts', 0),
                      'local_cache_hit': getattr(exc, 'local_cache_hit', False),
                      'local_sdk_requests': getattr(exc, 'local_sdk_requests', None),
                      'raw_artifact_path': getattr(exc, 'raw_artifact_path', None)}
        record.update(case_id=case['id'], source=case['source'], elapsed_seconds=time.perf_counter() - started,
                      original_pdf_review='PENDING', provider_attempts=0)
        save_new(target, record)
        print(json.dumps({'case': case['id'], 'status': record['status'],
                          'seconds': round(record['elapsed_seconds'], 2), 'error': record.get('error')}, ensure_ascii=False), flush=True)
    print(str(folder))


def structured_result_is_bound(case, record):
    if record.get('status') != 'PARSED':
        return False
    try:
        artifact = ParsedArtifact.model_validate(record['result']['metadata']['structured_extraction'])
        source = SourceRegion.model_validate(case['source'])
    except (KeyError, TypeError, ValueError):
        return False
    return (artifact.source == source and artifact.kind == case['kind'] and
            artifact.parse_status == 'PARSED' and artifact.validation_status == 'VALIDATED' and
            bool(artifact.table_rows and artifact.column_headers if case['kind'] == 'table'
                 else artifact.latex and artifact.latex.strip()))


def quality_summary(cases, records, reviews):
    groups = {}
    for kind in ['table', 'equation']:
        subset = [c for c in cases if c['kind'] == kind]
        held = [c for c in subset if not c['known_failure']]
        passed = [c for c in subset if structured_result_is_bound(c, records.get(c['id'], {}))
                  and reviews.get(c['id'], {}).get('verdict') == 'PASS'
                  and reviews[c['id']].get('result_sha256') == config_sha(records[c['id']])
                  and reviews[c['id']].get('crop_sha256') == c['source']['crop_sha256']]
        passed_ids = {c['id'] for c in passed}
        hold_pass = sum(c['id'] in passed_ids for c in held)
        known = [c for c in subset if c['known_failure']]
        known_ok = len(known) >= (2 if kind == 'table' else 1) and all(c['id'] in passed_ids for c in known)
        groups[kind] = {'total': len(subset), 'passed': len(passed), 'holdout_total': len(held),
            'holdout_passed': hold_pass, 'required_holdout_passes': math.ceil(.8 * len(held)),
            'known_failures_passed': known_ok,
            'gate_passed': len(subset) == 10 and known_ok and hold_pass >= math.ceil(.8 * len(held))}
    return {'groups': groups, 'promote_default': all(g['gate_passed'] for g in groups.values()),
            'assurance': 'Small frozen region sample; not whole-Agent accuracy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--execute-local', action='store_true')
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--summarize', type=Path, help='Existing run folder; requires crop/result-bound reviews.json')
    parser.add_argument('--reviews', type=Path, help='Separate immutable review revision for --summarize')
    parser.add_argument('--summary-output', type=Path, help='Separate summary path; existing contents cannot change')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.execute_local:
        execute(args.case_id)
    elif args.summarize:
        cases = json.loads((FOLDER / 'cases.json').read_text(encoding='utf-8'))['cases']
        records = {c['id']: json.loads((args.summarize / f'{c["id"]}.json').read_text(encoding='utf-8'))
                   for c in cases if (args.summarize / f'{c["id"]}.json').is_file()}
        review_path = args.reviews or args.summarize / 'reviews.json'
        reviews = json.loads(review_path.read_text(encoding='utf-8'))['reviews']
        summary = quality_summary(cases, records, reviews)
        if args.reviews:
            summary['review_sha256'] = file_sha(review_path)
        target = args.summary_output or args.summarize / 'quality-summary.json'
        if target.exists():
            if json.loads(target.read_text(encoding='utf-8')) != summary:
                raise ValueError('Saved review summary changed; preserve it and create a new review revision')
        else:
            save_new(target, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print('Zero-call default. Use --freeze, then --execute-local. Paid APIs are not supported.')


if __name__ == '__main__':
    main()
