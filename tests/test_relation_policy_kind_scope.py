import json
from copy import deepcopy

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from test_inference_citation_recovery import DRAFT, EVIDENCE, Model


@pytest.mark.parametrize('verdict', ['SUPPORTED', 'UNSUPPORTED'])
def test_relation_policy_does_not_overrule_inference_derivation_contract(verdict):
    model = Model(deepcopy(DRAFT), verdict)
    result = answer_then_audit('Compare representations.', EVIDENCE, model,
                              relation_policy=True, allow_repair=False)
    instruction = model.messages[1][0]['content']
    assert 'source-only entailment check to EVERY clause of every fact' not in instruction
    assert 'every SOURCE_FACT' in instruction
    assert 'For INFERENCE' in instruction
    payload = json.loads(model.messages[1][-1]['content'])
    assert payload['facts'][-1]['kind'] == 'INFERENCE'
    assert payload['inference_review']['author_comparison_required'] is False
    assert result['direct_answer_supported'] == (verdict == 'SUPPORTED')
