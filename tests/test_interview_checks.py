"""Freeze original questions and retain content checks for the default Reader."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.interview.cases import acceptance_cases, CHANGE, FLOW  # noqa: E402
from evals.interview.checks import structural_observations  # noqa: E402


def test_original_three_questions_and_controls_are_not_rewritten():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'demo/interview_scenarios.json').read_text(encoding='utf-8'))['scenarios']
    cases = acceptance_cases(root)
    assert len(cases) == 9
    for before, after in zip(original, cases[:3]):
        for field in ('id', 'question', 'answer_checks', 'success_requirements'):
            assert after[field] == before[field]
    assert {c['dimension'] for c in cases[3:]} == {
        'mechanism', 'numeric_ownership', 'comparability', 'scope', 'missing_evidence', 'history'}


def test_source_ownership_is_checked_even_when_model_says_supported():
    case = {'paper_ids': [CHANGE], 'answer_checks': {'minimum_supported_claims': 1}}
    result = {'run': {'status': 'COMPLETED', 'metadata': {'reader_report': {
        'audit': {'coverage': 'COMPLETE', 'assessments': [{'verdict': 'SUPPORTED'}]}}}},
        'final_answer': {'answer': 'An asserted fact. [evidence_one]'},
        'evidence': [{'evidence_id': 'evidence_one', 'provenance': {'paper_sha256': FLOW}}]}
    check = structural_observations(case, result)
    assert check['structural_issues'] == ['citation_outside_allowed_papers']
    assert check['model_supported_facts'] == 1
    assert check['independent_source_review'] == 'PENDING' and check['live_demo_pass'] is False
