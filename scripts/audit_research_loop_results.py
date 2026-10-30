"""Read-only recomputation over saved live runs; no provider or experiment calls."""
import argparse
import json
from pathlib import Path


def audit(root):
    runs = []
    for directory in sorted(root.glob('research-real-20260913-*')):
        count_path = directory / 'attempt-count.json'
        if not count_path.exists():
            continue  # Active run has not finalized its accounting yet.
        saved = directory / 'result.json'
        if not saved.exists():
            saved = directory / 'before-approval.json'
        result = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {}
        summaries = []
        for call in result.get('tool_calls', []):
            if call['tool_name'] != 'experiment.run':
                continue
            experiment = (call.get('result') or {}).get('output', {})
            for row in experiment.get('seed_runs', []):
                metrics = row['result']['metrics']
                questions = metrics['rows']
                baseline = sum(q['baseline'] for q in questions) / len(questions)
                candidate = sum(q['candidate'] for q in questions) / len(questions)
                assert abs(baseline - metrics['baseline_recall_at_3']) < 1e-12
                assert abs(candidate - metrics['candidate_recall_at_3']) < 1e-12
                assert abs(candidate - baseline - metrics['improvement']) < 1e-12
                summaries.append({'seed': row['seed'], 'questions': len(questions),
                    'baseline_recall_at_3': baseline, 'candidate_recall_at_3': candidate,
                    'delta_percentage_points': (candidate - baseline) * 100,
                    'unique_question_ids': [q['question_id'] for q in questions]})
        final = result.get('final_answer') or {}
        runs.append({'directory': str(directory),
                     'actual_requests': json.loads(count_path.read_text())['actual_requests'],
                     'request_files': len(list((directory/'api').glob('*-request.json'))),
                     'status': result.get('run', {}).get('status'),
                     'outcome': result.get('run', {}).get('research_outcome'),
                     'answer_quality_passed': final.get('answer_quality', {}).get('passed'),
                     'claims': [{'status': c['status'], 'statement': c['statement']}
                                for c in result.get('claim_graph', {}).get('claims', [])],
                     'unique_scored_questions': len({qid for row in summaries for qid in row['unique_question_ids']}),
                     'seed_results': [{k: v for k, v in row.items() if k != 'unique_question_ids'} for row in summaries]})
    assert all(row['actual_requests'] == row['request_files'] for row in runs)
    return {'total_actual_requests': sum(row['actual_requests'] for row in runs), 'runs': runs,
            'scope': 'Recompute metric arithmetic from persisted rows, not an independent scientific review'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.paperpilot'))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))
