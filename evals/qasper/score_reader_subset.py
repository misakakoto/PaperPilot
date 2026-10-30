"""Score saved non-oracle reader outputs; no model calls or answer rewriting."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def score(workspaces, dataset_path, evaluator_path):
    dataset = json.loads(dataset_path.read_text(encoding='utf-8-sig'))
    spec = importlib.util.spec_from_file_location('official_qasper', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    gold = evaluator.get_answers_and_evidence(dataset, False)
    questions = {}
    for pid, paper in dataset.items():
        for qa in paper['qas']:
            if qa['question_id'] in questions:
                raise ValueError('Duplicate question ID in dataset')
            questions[qa['question_id']] = (str(pid), qa['question'])
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    rows, seen = [], set()
    for workspace in workspaces:
        manifest = json.loads((workspace/'manifest.json').read_text(encoding='utf-8'))
        result_path = workspace/'reader-result.json'
        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
        qid = manifest['question_id']
        if qid in seen or qid not in gold:
            raise ValueError('Duplicate or unknown question')
        seen.add(qid)
        if (manifest.get('paper_id'), manifest['question']) != questions[qid]:
            raise ValueError('Manifest differs from original dataset question or paper')
        if manifest.get('dataset_sha256', dataset_sha256) != dataset_sha256:
            raise ValueError('Dataset fingerprint mismatch')
        if manifest.get('gold_answers_sent') is not False or manifest.get('gold_evidence_sent') is not False:
            raise ValueError('Expected non-oracle inputs')
        if result is not None and result['question'] != manifest['question']:
            raise ValueError('Question changed')
        reader = result['reader'] if result is not None else {}
        original = reader.get('answer') or ''
        if not isinstance(original, str):
            raise ValueError('Answer must be text')
        citation_free = re.sub(r'\[evidence_[A-Za-z0-9_-]+\]', '', original).strip()
        # Do not pick a favorable sentence, turn prose abstention into a gold
        # label, or select a numeric answer using the reference.
        references = [r['answer'] for r in gold[qid]]
        run = result['persisted_run'] if result is not None else {'status': 'MISSING'}
        valid = (run['status'] == 'COMPLETED' and bool(original.strip())
                 and not reader.get('error_type') and reader.get('status') not in {'FAILED', 'ROUTING_FAILED'})
        rows.append({'question_id': qid, 'question': manifest['question'], 'original_answer': original,
            'reference_answers': references, 'raw_f1': max(evaluator.token_f1_score(original, g) for g in references) if valid else 0.0,
            'citation_free_f1': max(evaluator.token_f1_score(citation_free, g) for g in references) if valid else 0.0,
            'valid_prediction': valid, 'run_status': run['status'],
            'calls': run.get('used_llm_calls'), 'tokens': run.get('used_tokens'),
            'manifest_sha256': hashlib.sha256((workspace/'manifest.json').read_bytes()).hexdigest(),
            'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest() if result is not None else None})
    if not rows:
        raise ValueError('Empty subset')
    return {'scope': 'non-oracle small dev diagnostic; not full benchmark',
        'scoring_protocol': 'non-oracle-reader-subset-v2-failure-zero-original-question',
        'denominator_scope': 'all explicitly supplied workspaces; not proof of a prospectively frozen cohort',
        'official_benchmark_comparable': False,
        'dataset_sha256': dataset_sha256,
        'evaluator_sha256': hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        'count': len(rows), 'raw_f1': sum(r['raw_f1'] for r in rows)/len(rows),
        'citation_free_f1': sum(r['citation_free_f1'] for r in rows)/len(rows),
        'invalid_predictions': sum(not row['valid_prediction'] for row in rows),
        'evidence_f1': None, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, action='append', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior scores')
    report = score(args.workspace, args.dataset, args.evaluator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
