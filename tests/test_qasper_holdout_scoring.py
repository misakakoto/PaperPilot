"""Evaluation accounting must not drop failed or unfinished workbench turns."""
import hashlib
import json
from types import SimpleNamespace

from evals.qasper import workbench_acceptance


def test_failed_missing_and_approval_turns_remain_in_score_denominator(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'p': {'abstract': '', 'full_text': []}}), encoding='utf-8')
    evaluator = tmp_path / 'evaluator.py'
    evaluator.write_text('# test-owned scorer boundary', encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    cases = [{'question_id': str(i), 'question': 'Original?', 'paper_id': 'p'} for i in range(4)]
    manifest.write_text(json.dumps({'kind': 'accounting-test', 'cases': cases,
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest()}), encoding='utf-8')
    # Keep the official-metric boundary trivial: this test covers our accounting,
    # not a replacement implementation of official F1.
    monkeypatch.setattr(workbench_acceptance, 'load_evaluator', lambda _: SimpleNamespace(
        get_answers_and_evidence=lambda *_: {str(i): [{'answer': 'A', 'evidence': []}] for i in range(4)},
        token_f1_score=lambda *_: 1.0, paragraph_f1_score=lambda *_: 1.0))
    workspace = tmp_path / 'runs'
    for i, status in enumerate(('COMPLETED', 'FAILED', 'WAITING_APPROVAL')):
        folder = workspace / f'case-{i:02}'
        folder.mkdir(parents=True)
        (folder / 'result.json').write_text(json.dumps({'run': {'status': status,
            'used_llm_calls': i+1, 'used_tokens': 10}, 'final_answer': {'answer': 'A'}}), encoding='utf-8')
    result = workbench_acceptance.score(dataset, manifest, workspace, evaluator)
    assert result['frozen_count'] == 4
    assert result['failures'] == 3
    assert set(result['metrics'].values()) == {0.25}
    assert result['total_calls'] == 6 and result['total_tokens'] == 30
    assert [r['run_status'] for r in result['rows']] == ['COMPLETED', 'FAILED', 'WAITING_APPROVAL', 'MISSING']
