from paperpilot_agentic.paper_document import document_units
from paperpilot_agentic.paper_tables import literal_reports, validate_grid
from paperpilot_agentic.paper_catalog import frontmatter_observations
from test_paper_document import raw_document


def test_table_rows_are_numeric_only_after_grid_and_header_validation():
    units = document_units(raw_document(), 'a' * 64, table_rows=2)
    for unit in units[1:]:
        assert unit['object']['structure_status'] == 'GRID_VALIDATED_NOT_FACT_VERIFIED'
        validate_grid(unit['object'])
        reports = literal_reports(unit)
        assert reports and all(r['metric'] == 'F1' and r['unit'] == '%' and r['dataset'] == 'LEVIR-CD' for r in reports)
        assert all(r['pretraining'] is None for r in reports)
        assert all(v in unit['text'] for r in reports for v in r['context_quotes'].values())


def test_missing_cells_in_successful_parser_output_never_become_experiment_facts():
    raw = raw_document()
    raw['document']['tables'][0]['data']['table_cells'].pop()
    unit = document_units(raw, 'a' * 64)[1]
    assert unit['object']['parse_status'] == 'INVALID_STRUCTURE'
    assert unit['object']['structure_errors'] and literal_reports(unit) == []


def test_first_page_doi_requires_label_and_preserves_wrapped_chapter_suffix():
    rows = frontmatter_observations('A related URL 10.1234/unlabelled\nhttps://doi.org/10.1007/book\n_49', 'a' * 64)
    assert next(r['value'] for r in rows if r['field'] == 'doi') == '10.1007/book_49'
