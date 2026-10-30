import copy

import pytest

from paperpilot_agentic.paper_document import document_units, parse_document
from paperpilot_agentic.paper_structure import digest


def raw_document():
    prov = [{'page_no': 2, 'bbox': {'l': 1, 't': 2, 'r': 3, 'b': 4, 'coord_origin': 'TOPLEFT'}, 'charspan': [0, 10]}]
    texts = [
        {'self_ref': '#/texts/0', 'label': 'section_header', 'level': 1, 'text': 'Experiments', 'prov': prov},
        {'self_ref': '#/texts/1', 'label': 'text', 'text': 'The original experimental settings.', 'prov': prov},
        {'self_ref': '#/texts/2', 'label': 'caption', 'text': 'Table 1. F1 (%) on LEVIR-CD.', 'prov': prov},
        {'self_ref': '#/texts/3', 'label': 'footnote', 'text': '* author reported test split', 'prov': prov}]
    cells = [{'start_row_offset_idx': r, 'end_row_offset_idx': r+1, 'start_col_offset_idx': c,
              'end_col_offset_idx': c+1, 'text': t, 'column_header': r == 0} for r, row in enumerate([
                  ['Method', 'F1 (%)'], ['A', '93.5'], ['B', '94.1'], ['C', '94.8']]) for c, t in enumerate(row)]
    return {'document': {'texts': texts, 'groups': [], 'pictures': [], 'tables': [{
        'self_ref': '#/tables/0', 'label': 'table', 'prov': prov, 'data': {'table_cells': cells, 'num_rows': 4, 'num_cols': 2},
        'captions': [{'$ref': '#/texts/2'}], 'footnotes': [{'$ref': '#/texts/3'}]}],
        'body': {'children': [{'$ref': '#/texts/0'}, {'$ref': '#/texts/1'}, {'$ref': '#/tables/0'}]}}}


def test_hierarchy_table_groups_retain_headers_cells_captions_and_footnotes():
    raw = raw_document()
    before = copy.deepcopy(raw)
    units = document_units(raw, 'a' * 64, table_rows=2)
    assert raw == before and len(units) == 3
    assert units[0]['section'] == 'Experiments' and units[0]['source']['page_range'] == [2, 2]
    tables = units[1:]
    assert [u['object']['row_group'] for u in tables] == [[1, 2], [3]]
    for u in tables:
        assert u['kind'] == 'TABLE' and u['object']['header_repeated']
        assert u['object']['column_headers'] == ['Method', 'F1 (%)']
        assert u['object']['footnotes'] == ['* author reported test split']
        assert 'Table 1. F1 (%) on LEVIR-CD.' in u['text']
        assert u['source']['spans'][0]['bbox']['coord_origin'] == 'TOPLEFT'
    assert [u['id'] for u in document_units(raw, 'a' * 64, table_rows=2)] == [u['id'] for u in units]


def test_missing_provenance_and_cycles_are_not_silently_accepted():
    raw = raw_document()
    raw['document']['texts'][1]['children'] = [{'$ref': '#/texts/1'}]
    with pytest.raises(ValueError, match='cyclic'):
        document_units(raw, 'a' * 64)


def test_unlinked_starred_note_is_attached_only_with_its_original_pdf_region():
    raw = raw_document()
    table = raw['document']['tables'][0]
    table['footnotes'] = []
    raw['document']['texts'][2]['text'] += '*'
    table['prov'][0]['bbox'] = {'l': 0, 'r': 100, 't': 10, 'b': 100, 'coord_origin': 'TOPLEFT'}
    note = raw['document']['texts'][3]
    note['prov'] = [{'page_no': 2, 'bbox': {'l': 0, 'r': 100, 't': 102, 'b': 112, 'coord_origin': 'TOPLEFT'}}]
    raw['document']['body']['children'].append({'$ref': note['self_ref']})
    units = document_units(raw, 'a' * 64)
    parsed_table = next(u for u in units if u['kind'] == 'TABLE')
    assert parsed_table['object']['footnotes'] == [note['text']]
    assert parsed_table['source']['associated_notes'][0]['object_ref'] == note['self_ref']
    assert note['text'] in parsed_table['text']


def test_local_fallback_is_explicit_hash_checked_and_never_overwrites(tmp_path):
    from pypdf import PdfWriter
    pdf = tmp_path / 'paper.pdf'
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(pdf)
    sha = digest(pdf.read_bytes())
    result = parse_document(pdf, sha, tmp_path / 'parsed')
    assert result['status'] == 'FALLBACK_TEXT_ONLY' and result['fallback_reason'] == 'DOCLING_NOT_CONFIGURED'
    assert result['fact_review'] == 'NOT_REVIEWED'
    with pytest.raises(FileExistsError):
        parse_document(pdf, sha, tmp_path / 'parsed')
    with pytest.raises(ValueError, match='version changed'):
        parse_document(pdf, 'f' * 64, tmp_path / 'wrong')
