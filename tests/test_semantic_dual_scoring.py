import hashlib

from evals.interview.semantic import dual_observations, VERSION, DIMENSIONS


def fixture(tmp_path):
    source = tmp_path / 'source.png'
    source.write_bytes(b'fixture source identity, not a real paper')
    answer = '对应位置的两个时相特征相邻，以此类推。F1 为 62.3%。'
    case = {'id': 'fixture', 'paper_ids': ['p'], 'answer_checks': {'required_substrings': ['交替排列']}}
    result = {'run': {'run_id': 'r', 'status': 'COMPLETED', 'research_outcome': 'ANSWERED', 'metadata': {}},
              'final_answer': {'answer': answer}, 'evidence': []}
    review = {'rubric_version': VERSION, 'review_kind': 'INDEPENDENT_ORIGINAL_SOURCE_REVIEW',
              'reviewer': 'synthetic harness only', 'case': 'fixture', 'run_id': 'r', 'result_sha256': 'result',
              'checks': [{'dimension': d, 'result': 'PASS', 'reason': 'Synthetic wiring test.',
                          'answer_excerpt': answer} for d in DIMENSIONS],
              'source_artifacts': [{'path': str(source), 'page': 1, 'paper_sha256': 'p',
                                   'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}
    return case, result, review


def test_legacy_keywords_remain_visible_but_not_a_factual_verdict(tmp_path):
    case, result, review = fixture(tmp_path)
    observed = dual_observations(case, result, result_sha256='result', review=review, replay=True)
    assert 'missing_term:交替排列' in observed['frozen_v1']['structural_issues']
    assert observed['semantic_v2']['dimensions']['factual_correctness'] == 'PASS'
    assert not observed['semantic_v2']['live_pass']


def test_no_review_and_changed_result_or_source_cannot_become_semantic_pass(tmp_path):
    case, result, review = fixture(tmp_path)
    assert not dual_observations(case, result, result_sha256='result')['semantic_v2']['source_review_pass']
    assert not dual_observations(case, result, result_sha256='changed', review=review)['semantic_v2']['review_valid']
    (tmp_path / 'source.png').write_bytes(b'changed')
    assert not dual_observations(case, result, result_sha256='result', review=review)['semantic_v2']['review_valid']


def test_format_failure_is_not_recorded_as_factual_failure(tmp_path):
    case, result, review = fixture(tmp_path)
    next(c for c in review['checks'] if c['dimension'] == 'presentation')['result'] = 'FAIL'
    output = dual_observations(case, result, result_sha256='result', review=review)['semantic_v2']
    assert output['dimensions']['factual_correctness'] == 'PASS'
    assert output['dimensions']['presentation'] == 'FAIL'
    assert not output['source_review_pass']


def test_unfinished_answer_is_not_run_even_if_a_draft_was_available(tmp_path):
    case, result, review = fixture(tmp_path)
    result['run']['status'] = 'BUDGET_EXHAUSTED'
    result['final_answer']['answer'] = ''
    output = dual_observations(case, result, result_sha256='result', review=review)['semantic_v2']
    assert set(output['dimensions'].values()) == {'NOT_RUN'}
    assert not output['live_pass']
