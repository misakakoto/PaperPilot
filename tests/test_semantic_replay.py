import hashlib
import json

import pytest

from evals.interview import replay_semantic


def test_replay_preserves_inputs_emits_original_answer_and_never_adds_live_pass(tmp_path, monkeypatch):
    source = tmp_path / 'original'
    source.mkdir()
    output = tmp_path / 'replay'
    case = {'id': 'c', 'question': 'Explain the mechanism.', 'paper_ids': ['p']}
    monkeypatch.setattr(replay_semantic, 'acceptance_cases', lambda root: [case, {**case, 'id': 'unrun'}])
    result = {'run': {'run_id': 'r', 'question': case['question'], 'status': 'COMPLETED',
                     'research_outcome': 'PARTIAL', 'metadata': {'reader_report': {
        'draft': {'facts': [{'id': 'f', 'text': 'The mechanism combines features.',
                            'evidence_ids': ['evidence_a']}], 'direct_answer_id': 'f', 'limitations': []},
        'model_audit': {'assessments': [{'fact_id': 'f', 'verdict': 'SUPPORTED',
                         'relevance': 'DIRECT', 'span_ids': ['s0']}],
                        'coverage': 'COMPLETE', 'missing_aspects': []},
        'source_spans': [{'span_id': 's0', 'evidence_id': 'evidence_a'}]}}},
        'final_answer': {'answer': 'Original saved answer.'},
        'evidence': [{'evidence_id': 'evidence_a', 'content': 'The mechanism combines features.'}]}
    path = source / 'c-result.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = replay_semantic.replay(tmp_path, source, output)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert report['actual_new_requests'] == report['live_passes_added'] == 0
    assert report['cases'][0]['original_answer'] == 'Original saved answer.'
    assert report['cases'][0]['current_guard_diagnostic']['provider_attempts'] == 0
    assert report['cases'][1]['status'] == 'NOT_RUN'
    assert 'Original saved answer.' in (output / 'questions-answers-diff.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        replay_semantic.replay(tmp_path, source, output)


def test_replay_cannot_write_under_preserved_results(tmp_path):
    with pytest.raises(ValueError):
        replay_semantic.replay(tmp_path, tmp_path / 'original', tmp_path / 'original/replay')
