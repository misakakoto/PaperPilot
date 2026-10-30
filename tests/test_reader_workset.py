import json
from paperpilot_agentic.reader_workset import source_aware_workset


def row(i, paper, content):
    return {'evidence_id': f'e{i}', 'source_uri': f'paper://{paper}/c{i}', 'page': i, 'content': content}


def test_round_robin_preserves_second_paper_and_whole_source_text():
    rows = [row(1, 'a', 'A' * 400), row(2, 'a', 'B' * 400), row(3, 'b', 'C' * 400)]
    original = json.dumps(rows)
    selected, report = source_aware_workset(rows, max_bytes=1100)
    assert [r['evidence_id'] for r in selected] == ['e1', 'e3']
    assert report['omitted_evidence_ids'] == ['e2']
    assert not report['unrepresented_sources']
    assert selected[0]['content'] == rows[0]['content']
    assert json.dumps(rows) == original


def test_unicode_byte_budget_does_not_silently_drop_source():
    selected, report = source_aware_workset([row(1, 'a', '中文' * 1000)], max_bytes=1000)
    assert selected == []
    assert report['unrepresented_sources'] == ['a']
    assert report['omitted_evidence_ids'] == ['e1']


def test_fit_preserves_every_item_and_empty_input_is_valid():
    rows = [row(1, 'a', 'first'), row(2, 'b', 'second')]
    selected, report = source_aware_workset(rows, max_bytes=10000)
    assert selected == rows
    assert not report['omitted_evidence_ids']
    assert source_aware_workset([], max_bytes=0)[0] == []
