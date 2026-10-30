import pytest

from paperpilot_agentic.follow_up_reader import resolve_follow_up
from test_followup_reference_repair import Resolver


CONTEXT = {'prior_question': '比较《RadarNet》和《OpticNet》的机制。'}


@pytest.mark.parametrize('question', [
    '比较前者（RadarNet）和后者（OpticNet）的机制，不比较精度。',
    '基于《RadarNet》和《OpticNet》这两篇论文，仅比较机制。',
    'Compare the former (RadarNet) with the latter (OpticNet), not accuracy.',
])
def test_explicitly_bound_subjects_do_not_trigger_clarification(question):
    model = Resolver([question, question])
    result = resolve_follow_up('比较上述方法的机制。', CONTEXT, model)
    assert result['question'] == question  # Never rewrite or invent the binding.
    assert not result['needs_clarification']
    assert result['provider_attempts'] == 1


@pytest.mark.parametrize('question', [
    '比较前者与后者。',
    '比较前者（UnknownNet）和后者（OpticNet）。',
    '提到RadarNet和OpticNet，再比较这两个方法。',
    '《UnknownNet》和《OpticNet》这两篇论文有什么不同？',
    'Compare the former (83.2) with the latter (OpticNet).',
])
def test_unbound_or_foreign_aliases_still_require_clarification(question):
    result = resolve_follow_up('比较上述方法。', CONTEXT, Resolver([question, question]))
    assert result['needs_clarification']
    assert result['provider_attempts'] == 2

