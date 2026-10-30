"""Score frozen oracle-reader dev outputs using the downloaded official evaluator.

Oracle inputs are not retrieved evidence, so Evidence F1 is deliberately not reported.
Missing/failed/truncated predictions remain in the frozen denominator with zero credit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location('official_qasper_evaluator', path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load official evaluator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(report: dict, dataset: dict, evaluator: Any) -> dict:
    if report.get('kind') != 'qasper_v0.3_oracle_evidence_reader_dev_smoke':
        raise ValueError('Only oracle dev reports are accepted')
    if report.get('dataset_split') != 'dev':
        raise ValueError('Expected dev split')
    references = evaluator.get_answers_and_evidence(dataset, False)
    cases = report['cases']
    ids = [case['question_id'] for case in cases]
    if not ids or len(set(ids)) != len(ids) or any(qid not in references for qid in ids):
        raise ValueError('Empty, duplicate or unknown frozen question IDs')
    questions = {qa['question_id']: (str(pid), qa['question'])
                 for pid, paper in dataset.items() for qa in paper['qas']}
    predicted = {}
    rows = []
    for case in cases:
        qid = case['question_id']
        if (case['paper_id'], case['question']) != questions[qid]:
            raise ValueError('Question changed after selection')
        answer = case.get('prediction')
        valid = (isinstance(answer, str) and bool(answer.strip())
                 and case.get('usage', {}).get('finish_reason') == 'stop'
                 and not case.get('error'))
        f1 = em = 0.0
        if valid:
            predicted[qid] = {'answer': answer, 'evidence': []}
            f1 = max(evaluator.token_f1_score(answer, ref['answer']) for ref in references[qid])
            em = float(any(evaluator.normalize_answer(answer) == evaluator.normalize_answer(ref['answer'])
                           for ref in references[qid]))
        rows.append({'question_id': qid, 'selection_type': case['answer_type'],
                     'valid_prediction': valid, 'answer_f1': f1, 'exact_match': em})
    official = evaluator.evaluate({qid: references[qid] for qid in ids}, predicted)
    mean_f1 = sum(row['answer_f1'] for row in rows) / len(rows)
    assert abs(mean_f1 - official['Answer F1']) < 1e-12
    groups = {}
    for kind in sorted({row['selection_type'] for row in rows}):
        subset = [row for row in rows if row['selection_type'] == kind]
        groups[kind] = {'count': len(subset), 'answer_f1': sum(r['answer_f1'] for r in subset) / len(subset)}
    return {'kind': 'oracle_reader_dev_diagnostic_score', 'count': len(rows),
            'answer_f1': mean_f1, 'exact_match': sum(r['exact_match'] for r in rows) / len(rows),
            'missing_or_failed': official['Missing predictions'], 'by_selection_type': groups,
            'evidence_f1': None, 'official_benchmark_comparable': False,
            'limitations': ['Gold evidence is supplied; retrieval and complete Agent are not evaluated.',
                           'Empty oracle evidence can reveal unanswerability; do not claim abstention capability.',
                           'Small type-balanced dev subset, not the natural full-test distribution.',
                           'F1 is token overlap, not a factual correctness judgment.'], 'rows': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output path')
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report['dataset_sha256']:
        raise ValueError('Dataset fingerprint mismatch')
    result = score(report, json.loads(raw.decode('utf-8-sig')), load_evaluator(args.evaluator))
    result['evaluator_sha256'] = hashlib.sha256(args.evaluator.read_bytes()).hexdigest()
    result['report_sha256'] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
