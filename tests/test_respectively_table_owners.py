import pytest

from paperpilot_agentic.request_scope import automatic_table_targets


def resolve(text):
    return {name for name in ('Alpha', 'Beta') if name in text}


def test_explicit_ordered_titles_and_respectively_tables_have_unique_owners():
    question = '仅依据《Alpha: Paper A》和《Beta: Paper B》，对比前者和后者；再分别引用 Table XII 和 Table VII 中一个消融结果。不得混用证据。'
    targets, gaps = automatic_table_targets(question, {'Alpha', 'Beta'}, resolve)
    assert not gaps
    assert targets == [{'paper_sha256': 'Alpha', 'label': 'Table XII'},
                       {'paper_sha256': 'Beta', 'label': 'Table VII'}]


@pytest.mark.parametrize('question,scope', [
    ('依据《Alpha》和《Beta》，引用 Table XII 和 Table VII。', {'Alpha', 'Beta'}),
    ('依据《Alpha》和《Beta》，不要分别引用 Table XII 和 Table VII。', {'Alpha', 'Beta'}),
    ('依据《Alpha》和《Beta》，如果分别引用 Table XII 和 Table VII 呢？', {'Alpha', 'Beta'}),
    ('依据《Alpha》和《Beta》，分别引用 Table XII 和 Table VII。', {'Alpha'}),
    ('依据《Alpha》和《Alpha》，分别引用 Table XII 和 Table VII。', {'Alpha', 'Beta'}),
    ('依据《Alpha》和《Unknown》，分别引用 Table XII 和 Table VII。', {'Alpha', 'Beta'}),
])
def test_ambiguous_negative_or_unselected_titles_are_not_order_mapped(question, scope):
    targets, gaps = automatic_table_targets(question, scope, resolve)
    assert not (len(targets) == 2 and targets[0]['paper_sha256'] != targets[1]['paper_sha256'])
