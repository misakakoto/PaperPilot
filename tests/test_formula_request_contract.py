import json

import pytest

from paperpilot_agentic.answer_contracts import draft_repair_diagnostics, validate_draft
from paperpilot_agentic.answer_pipeline import AnswerOptions, initialize, writer_messages


BAD = {'facts': [{'id': 'sequence', 'kind': 'SOURCE_FACT',
    'text': 'The tokens form F = [x(1), ..., x(HW/(21+j))].',
    'evidence_ids': ['method'], 'depends_on': []}], 'direct_answer_id': 'sequence', 'limitations': []}
EVIDENCE = [{'evidence_id': 'method', 'content': 'First use all tokens of the earlier image, then all tokens of the later image.'}]


def test_captured_failure_shape_identifies_the_fact_to_rewrite_without_deleting_it():
    with pytest.raises(ValueError, match='Unrequested structural formula') as exc:
        validate_draft(BAD, EVIDENCE, question='Explain how tokens are organized.')
    diagnostic = draft_repair_diagnostics(BAD, question='Explain how tokens are organized.', primary=exc.value)
    assert 'fact_id=sequence' in diagnostic
    assert 'Rewrite' in diagnostic and 'preserve' in diagnostic
    correct = json.loads(json.dumps(BAD))
    correct['facts'][0]['text'] = EVIDENCE[0]['content']
    assert validate_draft(correct, EVIDENCE, question='Explain how tokens are organized.') == correct


@pytest.mark.parametrize('question,requested', [
    ('Explain how tokens are organized.', False),
    ('Give the mathematical equation for the token sequence.', True),
])
def test_writer_and_validator_share_resolved_formula_requirement(question, requested):
    context = initialize(question, EVIDENCE, None, AnswerOptions(), None)
    payload = json.loads(writer_messages(context)[-1]['content'])
    contract = payload['output_contract']
    assert contract['formulas_requested'] is requested
    if requested:
        assert validate_draft(BAD, EVIDENCE, question=question) == BAD
    else:
        assert 'token order' in contract['prose']
        with pytest.raises(ValueError):
            validate_draft(BAD, EVIDENCE, question=question)


def test_writer_contract_separates_measured_rows_and_their_explanations():
    context = initialize('Compare the modules.', EVIDENCE, None, AnswerOptions(), None)
    contract = json.loads(writer_messages(context)[-1]['content'])['output_contract']
    assert 'separate facts for baseline and candidate' in contract['source_fact_scope']
    assert 'Correct numbers do not support' in contract['source_fact_scope']
