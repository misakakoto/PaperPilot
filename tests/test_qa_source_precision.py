"""Source intervals must not become exact pages or false budget failures."""
from types import SimpleNamespace

from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_agentic.execution_diagnostics import execution_fault
from paperpilot_agentic.read_packet import build_read_packet


def hit(text, section='Methods', start=1, end=20):
    return SimpleNamespace(channels=('bm25',), title='Example', section=section,
        text=text, paper_sha256='paper', stable_id=section, page_start=start,
        page_end=end, score=1., file_name='example.pdf')


def test_document_interval_does_not_trigger_bibliography_page_cutoff():
    rows = []
    for i, source in enumerate([hit('References\n[1] Work', 'References'),
                               hit('The alignment loss compares aligned features.')]):
        row = PaperPilotKnowledgeSearchAdapter._hit_to_item(source, rank=i+1, memory_ids=[])
        row['evidence_id'] = str(i)
        rows.append(row)
    assert rows[1]['page'] is None
    assert rows[1]['provenance']['page_range'] == [1, 20]
    packet = build_read_packet('Explain alignment loss', rows)
    assert [r['evidence_id'] for r in packet.rows] == ['1']


def test_exact_single_page_remains_exact():
    row = PaperPilotKnowledgeSearchAdapter._hit_to_item(hit('Method', start=4, end=4),
                                                       rank=1, memory_ids=[])
    assert row['page'] == 4
    assert row['provenance']['page_precision'] == 'EXACT'


def test_section_read_preserves_interval_precision():
    from paperpilot_agentic.section_router import read_selected_sections
    source = hit('Original full paragraph')
    source.stable_id = 'chunk'
    store = SimpleNamespace(get_chunks=lambda ids: [source])
    outline = [{'section_id': 's', 'chunk_ids': ['chunk']}]
    rows, _ = read_selected_sections(store, 'paper', outline, ['s'])
    assert rows[0]['page'] is None and rows[0]['provenance']['page_range'] == [1, 20]


def test_source_coverage_failure_is_not_a_budget_failure():
    run = {'status': 'FAILED', 'metadata': {'reader_report': {'error_type': 'SourceCoverageError'}}}
    assert execution_fault(run, {})['code'] == 'SOURCE_COVERAGE_INCOMPLETE'
