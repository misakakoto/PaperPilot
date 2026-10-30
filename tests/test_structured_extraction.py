import copy

import pytest

from paperpilot_agentic.docling_adapter import project_table, choose_candidate
from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion, table_observation_status


def region():
    return SourceRegion(paper_sha256='source', object_id='table1', label='Table I', page=2,
                        bbox=[10, 20, 100, 90], crop_sha256='crop')


def table_item():
    artifact = ParsedArtifact(kind='table', source=region(), parser='test', parser_version='1',
        config_sha256='config', parse_status='PARSED', validation_status='VALIDATED',
        column_headers=['F1'], table_rows=[{'cells': {'F1': '80.14'}}])
    return {'source_uri': 'paper://source/layout/table1', 'content': 'Completely different presentation',
        'provenance': {'layout_object_id': 'table1', 'layout_target_binding': {
            'document_sha256': 'source', 'object_id': 'table1', 'label': ['table', 'I'],
            'basis': 'own_caption_or_heading'}, 'structured_extraction': artifact.model_dump()}}


def test_status_uses_bound_typed_observation_not_text():
    item = table_item()
    assert table_observation_status(item, 'source', 'Table I') == 'TRANSCRIBED'
    assert item['provenance']['structured_extraction']['factual_review'] == 'NOT_REVIEWED'
    assert table_observation_status(item, 'other', 'Table I') == 'NOT_VERIFIED'
    assert table_observation_status(item, 'source', 'Table II') == 'NOT_VERIFIED'


@pytest.mark.parametrize('field,value', [('validation_status', 'CONFLICT'), ('parse_status', 'FAILED'),
                                        ('table_rows', []), ('schema_version', 'unknown')])
def test_rejects_unvalidated_or_malformed_observation(field, value):
    item = table_item()
    item['provenance']['structured_extraction'][field] = value
    assert table_observation_status(item, 'source', 'Table I') == 'NOT_VERIFIED'


def test_marker_cannot_promote_unverified_rows_and_legacy_is_read_only():
    item = table_item()
    del item['provenance']['structured_extraction']
    item['content'] = 'Table I Named-cell transcription'
    assert table_observation_status(item, 'source', 'Table I') == 'NOT_VERIFIED'
    item['provenance'].update(vision_verified=True, vision_metadata={'audited_table_rows': [{'cells': {'F1': '1'}}]})
    before = copy.deepcopy(item)
    assert table_observation_status(item, 'source', 'Table I') == 'TRANSCRIBED'
    assert item == before


def cell(r, c, text, *, endr=None, endc=None, header=False):
    return {'start_row_offset_idx': r, 'end_row_offset_idx': endr or r + 1,
            'start_col_offset_idx': c, 'end_col_offset_idx': endc or c + 1,
            'text': text, 'column_header': header}


def grouped_table():
    return {'num_rows': 4, 'num_cols': 3, 'table_cells': [
        cell(0, 0, 'Conditions', endc=2, header=True), cell(0, 2, 'F1', endr=2, header=True),
        cell(1, 0, 'A', header=True), cell(1, 1, 'B', header=True),
        cell(2, 0, '✓', endr=4), cell(2, 1, ''), cell(2, 2, '80.14'),
        cell(3, 1, '✓'), cell(3, 2, '80.26')]}


def test_spans_preserve_condition_and_metric_ownership():
    original = grouped_table()
    before = copy.deepcopy(original)
    cells, headers, paths, indicators, rows = project_table(original)
    assert headers == ['Conditions / A', 'Conditions / B', 'F1']
    assert paths[0] == ['Conditions', 'A']
    assert indicators == headers[:2]
    assert rows[0]['checked_columns'] == [headers[0]]
    assert rows[1]['checked_columns'] == headers[:2]
    assert [r['cells']['F1'] for r in rows] == ['80.14', '80.26']
    assert any(c.row_span == 2 and c.text == '✓' for c in cells)
    assert original == before


@pytest.mark.parametrize('problem', ['overlap', 'missing', 'outside'])
def test_conflicts_are_rejected_without_filling_cells(problem):
    table = grouped_table()
    if problem == 'overlap':
        table['table_cells'].append(cell(2, 0, ''))
    elif problem == 'missing':
        table['table_cells'].pop()
    else:
        table['table_cells'][0]['end_col_offset_idx'] = 4
    with pytest.raises(ValueError):
        project_table(table)


def test_empty_ocr_table_is_not_promoted_to_validated_evidence():
    table = grouped_table()
    for entry in table['table_cells']:
        entry['text'] = ''
    with pytest.raises(ValueError, match='Empty table transcription'):
        project_table(table)


def test_bottom_left_coordinates_and_wrong_caption_do_not_select_neighbor():
    source = region()
    candidate = {'prov': [{'page_no': 2, 'bbox': {'l': 15, 'r': 90, 't': 180, 'b': 120,
                                                'coord_origin': 'BOTTOMLEFT'}}], 'captions_text': ['Table I']}
    raw = {'document': {'pages': {'2': {'size': {'height': 200}}}}, 'candidates': [candidate]}
    assert choose_candidate(raw, source, 'table') == candidate
    raw['candidates'][0]['captions_text'] = ['Table II']
    with pytest.raises(ValueError):
        choose_candidate(raw, source, 'table')


@pytest.mark.parametrize('failure', [False, True])
def test_local_tool_dispatch_does_not_reserve_provider_attempts(tmp_path, monkeypatch, failure):
    import json
    import subprocess
    from paperpilot_agentic import builtin_tools
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.tooling import ToolContext
    from paperpilot_agentic.structured_extraction import file_sha
    pdf, image = tmp_path / 'paper.pdf', tmp_path / 'crop.png'
    pdf.write_bytes(b'local fixture')
    image.write_bytes(b'local crop')
    sha = file_sha(pdf)
    item = {'object_id': 'table1', 'object_type': 'table', 'page': 2, 'bbox': [10, 20, 100, 90],
            'caption': 'Table I', 'artifact_path': str(image), 'document_sha256': sha}
    def parse(pdf, output, **kwargs):
        (output / 'layout_manifest.json').write_text(json.dumps({'objects': [item]}))
        return {'objects': [item]}
    monkeypatch.setattr(builtin_tools, 'parse_pdf_layout', parse)
    monkeypatch.setattr(builtin_tools, 'search_layout_objects', lambda *a, **k: [item])
    class Local:
        local_execution = True
        def analyze(self, path, prompt, metadata):
            if failure:
                raise subprocess.TimeoutExpired('local-worker', 300)
            artifact = ParsedArtifact(kind='table', source=metadata['source_region'], parser='local',
                parser_version='1', config_sha256='config', parse_status='PARSED',
                validation_status='VALIDATED', column_headers=['F1'], table_rows=[{'cells': {'F1': '1'}}])
            return {'description': 'F1: 1', 'verified': True, 'confidence': .6, 'metadata': {
                'structured_extraction': artifact.model_dump(), 'local_parse_attempts': 1}}
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run('Read Table I', Budget(max_llm_calls=1))
    context = ToolContext(rid, 'step', 'tool', tmp_path, db, None)
    result = builtin_tools._layout_handler(Local())(context, {'pdf_path': str(pdf), 'query': 'Table I',
        'object_type': 'table', 'gap': 'numeric_results'})
    assert result.output['vision_api_calls'] == 0
    assert result.output['local_parse_attempts'] == 1
    assert db.get_run(rid)['used_llm_calls'] == 0
    assert result.evidence[0]['provenance']['vision_verified'] is (not failure)


def test_parse_and_pending_reviews_are_not_quality_passes():
    from evals.structured_pdf import quality_summary
    case = {'id': 't', 'kind': 'table', 'known_failure': True, 'source': {'crop_sha256': 'crop'}}
    assert not quality_summary([case], {'t': {'status': 'PARSED'}}, {})['promote_default']


def test_changed_source_is_rejected_before_any_local_or_cloud_dispatch(tmp_path):
    from paperpilot_agentic.docling_adapter import DoclingEvidenceAdapter, LocalExtractionError
    pdf, image = tmp_path / 'paper.pdf', tmp_path / 'crop.png'
    pdf.write_bytes(b'changed source')
    image.write_bytes(b'pixels')
    adapter = DoclingEvidenceAdapter.__new__(DoclingEvidenceAdapter)
    with pytest.raises(LocalExtractionError, match='Source changed') as caught:
        adapter.analyze(image, '', {'object_type': 'table', 'pdf_path': str(pdf),
            'source_region': region().model_dump()})
    assert caught.value.local_parse_attempts == caught.value.local_sdk_requests == 0
