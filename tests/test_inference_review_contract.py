from copy import deepcopy
import json

import pytest

from paperpilot_agentic.answer_audit import answer_then_audit
from paperpilot_agentic.answer_pipeline import AnswerOptions, initialize, repair_isolated_facts
from paperpilot_agentic.providers import LLMResponse
from test_inference_citation_recovery import DRAFT, EVIDENCE, Model


@pytest.mark.parametrize('verdict', ['SUPPORTED', 'UNSUPPORTED'])
def test_actual_audit_message_separates_derived_support_from_author_attribution(verdict):
    model = Model(deepcopy(DRAFT), verdict)
    result = answer_then_audit('Compare representations.', EVIDENCE, model, allow_repair=False)
    payload = json.loads(model.messages[1][-1]['content'])
    contract = payload['inference_review']
    assert contract['premise_bindings'] == [{'fact_id': 'f3', 'premise_fact_ids': ['f1', 'f2']}]
    assert contract['author_comparison_required'] is False
    assert contract['premises_require_independent_support'] is True
    assert 'not need to appear verbatim' in model.messages[1][0]['content']
    assert result['direct_answer_supported'] == (verdict == 'SUPPORTED')
    if verdict == 'UNSUPPORTED':
        assert DRAFT['facts'][2]['text'] not in result['answer']
        assert DRAFT['facts'][0]['text'] in result['answer']


def test_actual_local_patch_has_no_model_owned_identity_or_copyable_goal_example():
    seen = []

    class Capture:
        def ensure_capacity(self, requests):
            pass

        def complete(self, messages, *, before_attempt, **kwargs):
            before_attempt()
            seen.append(messages)
            return LLMResponse('{}', finish_reason='stop')

    draft = deepcopy(DRAFT)
    draft['facts'][2]['evidence_ids'] = ['left', 'right']
    context = initialize('Compare representations.', EVIDENCE, Capture(), AnswerOptions(), None)
    context.report.update(pre_isolation_draft=draft,
        draft_isolation={'quarantined_fact_ids': ['f3'],
                         'errors': [{'fact_id': 'f3', 'error': 'Unsupported comparison'}]})
    repair_isolated_facts(context, draft, lambda *args, **kwargs: pytest.fail('No valid patch'))
    assert len(seen) == 1
    instruction = seen[0][0]['content']
    assert 'WITHOUT id' in instruction
    assert '"id":"m"' not in instruction
    assert 'Authors aim to model temporal relations.' not in instruction
