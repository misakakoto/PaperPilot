"""Supplemental semantic-v2 review; original literal observations stay intact."""
import hashlib
from pathlib import Path

from .checks import structural_observations

VERSION = 'semantic-v2-20261005'
DIMENSIONS = {'factual_correctness', 'completeness', 'citation_ownership',
              'conclusion_scope', 'presentation'}
HARD_ISSUES = {'process_not_completed', 'empty_answer', 'unowned_citation',
               'citation_outside_allowed_papers', 'citation_source_identity_missing',
               'required_paper_not_cited', 'followup_lineage_invalid'}


def dual_observations(case, result, *, result_sha256, parent=None, review=None, replay=False):
    frozen = structural_observations(case, result, parent)
    answer = (result.get('final_answer') or {}).get('answer') or ''
    executed = result['run']['status'] == 'COMPLETED' and bool(answer)
    default = 'PENDING' if executed else 'NOT_RUN'
    dimensions = {name: default for name in sorted(DIMENSIONS)}
    valid, errors = False, []
    if review:
        checks = review.get('checks', [])
        valid = (review.get('rubric_version') == VERSION and
                 review.get('review_kind') == 'INDEPENDENT_ORIGINAL_SOURCE_REVIEW' and
                 bool(review.get('reviewer')) and review.get('case') == case['id'] and
                 review.get('run_id') == result['run']['run_id'] and
                 review.get('result_sha256') == result_sha256 and
                 len(checks) == len(DIMENSIONS) and
                 {c.get('dimension') for c in checks} == DIMENSIONS)
        valid = valid and all(c.get('result') in {'PASS', 'FAIL', 'PENDING', 'NOT_RUN'} and
                              isinstance(c.get('reason'), str) and c['reason'].strip() and
                              isinstance(c.get('answer_excerpt'), str) and
                              c['answer_excerpt'] and c['answer_excerpt'] in answer for c in checks)
        artifacts = review.get('source_artifacts', [])
        valid = valid and bool(artifacts)
        for artifact in artifacts:
            try:
                path = Path(artifact['path'])
                artifact_ok = (artifact.get('paper_sha256') in case['paper_ids'] and
                               isinstance(artifact.get('page'), int) and artifact['page'] > 0 and
                               hashlib.sha256(path.read_bytes()).hexdigest() == artifact.get('sha256'))
            except (KeyError, OSError, TypeError):
                artifact_ok = False
            if not artifact_ok:
                valid = False
        if valid and executed:
            dimensions = {c['dimension']: c['result'] for c in checks}
        else:
            errors.append('semantic_review_not_bound_to_final_answer_and_source_artifacts')
    passed = executed and valid and all(value == 'PASS' for value in dimensions.values())
    # The protocol still requires a completed ANSWERED outcome, not just prose.
    completed_outcome = result['run'].get('research_outcome') == 'ANSWERED'
    live = (passed and completed_outcome and frozen['citation_count'] > 0 and not replay
            and not (set(frozen['structural_issues']) & HARD_ISSUES))
    return {'frozen_v1': frozen, 'semantic_v2': {'rubric_version': VERSION,
        'evaluation_mode': 'REPLAY' if replay else 'LIVE', 'dimensions': dimensions,
        'review_valid': valid, 'review_errors': errors, 'source_review_pass': passed,
        'live_pass': bool(live), 'independent_source_review': default if not valid else 'RECORDED',
        'note': 'Literal keyword observations do not establish semantic correctness; replay is never live acceptance.'}}
