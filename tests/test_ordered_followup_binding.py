"""An ordered local title pair binds former/latter without rewriting the model text."""
import pytest
from paperpilot_agentic.follow_up_reader import _has_unbound_subject

CONTEXT = {'prior_question': 'Compare 《Synthetic Radar Model》 and 《Synthetic Optical Model》.'}


def test_local_ordered_pair_is_a_binding_not_a_bare_historical_pronoun():
    question = ('基于《Synthetic Radar Model》和《Synthetic Optical Model》这两篇论文的证据，'
                '哪一种设计（前者空间重排或后者坐标形变）更直接处理错位？')
    assert not _has_unbound_subject(question, CONTEXT)


@pytest.mark.parametrize('question', [
    'Which is better, 前者 or 后者?',
    '比较《Unknown Model》和《Synthetic Optical Model》，前者是否更好？',
    '《Synthetic Radar Model》和《Synthetic Optical Model》有区别。前者如何实现？',
    '《Synthetic Radar Model》和《Synthetic Optical Model》与《Third》和《Fourth》，前者如何？',
])
def test_unknown_or_distant_antecedent_remains_unresolved(question):
    assert _has_unbound_subject(question, CONTEXT)
