import copy
import pytest
from paperpilot_agentic.reviewed_sources import ReviewedSources
from paperpilot_agentic.structured_extraction import ParsedArtifact, SourceRegion


def artifact():
    return ParsedArtifact(kind='table', source=SourceRegion(paper_sha256='p', object_id='o',
        label='Table I', page=2, bbox=[1, 2, 10, 20], crop_sha256='c'),
        parser='literal-transcription', parser_version='1', config_sha256='cfg',
        parse_status='PARSED', validation_status='VALIDATED', column_headers=['F1 (%)'],
        table_rows=[{'row_label': '', 'checked_columns': [], 'cells': {'F1 (%)': '62.30'}}])


def review():
    return {'reviewer_id': 'Codex', 'reviewer_kind': 'ai', 'reviewed_at': '2026-10-10',
            'checks': [{'dimension': d, 'result': 'PASS', 'note': 'Checked exact source crop.'}
                       for d in ('identity', 'values', 'conditions', 'headers', 'spans', 'units')]}


def test_reviewed_asset_applies_without_rewriting_old_response(tmp_path):
    registry = ReviewedSources(tmp_path)
    asset = artifact()
    identifier = registry.register(asset, review())
    old = {'uncertain_fields': ['Original uncertain row alignment'], 'verified': False}
    evidence = {'content': 'old', 'provenance': {'old_response': copy.deepcopy(old)}}
    assert registry.apply(evidence, asset.source)
    assert evidence['provenance']['old_response'] == old
    assert evidence['provenance']['reviewed_source']['review']['reviewer_kind'] == 'ai'
    assert evidence['provenance']['vision_metadata']['structured_extraction']['factual_review'] == 'NOT_REVIEWED'
    assert registry.register(asset, review()) == identifier


@pytest.mark.parametrize('field,value', [('crop_sha256', 'changed'), ('paper_sha256', 'other'),
    ('page', 3), ('bbox', [1, 2, 11, 20]), ('object_id', 'other')])
def test_any_source_identity_change_invalidates_asset(tmp_path, field, value):
    registry = ReviewedSources(tmp_path)
    item = artifact()
    registry.register(item, review())
    source = item.source.model_copy(update={field: value})
    assert registry.resolve(source) is None


def test_conflicting_transcription_or_tampering_never_reuses(tmp_path):
    registry = ReviewedSources(tmp_path)
    item = artifact()
    identifier = registry.register(item, review())
    path = tmp_path / (identifier + '.json')
    path.write_text(path.read_text().replace('62.30', '99.90'))
    assert registry.resolve(item.source) is None


def test_unreviewed_dimension_cannot_register(tmp_path):
    check = review()
    check['checks'][0]['result'] = 'UNKNOWN'
    with pytest.raises(ValueError): ReviewedSources(tmp_path).register(artifact(), check)


def test_performance_table_without_module_toggles_retains_unknown_conditions(tmp_path):
    from paperpilot_agentic.table_claims import row_catalog
    from paperpilot_agentic.structured_extraction import annotate_observation
    from paperpilot_agentic.table_projection import project_table_facts
    item = artifact()
    item.table_rows[0]['checked_columns'] = None
    item.table_rows[0]['row_label'] = 'Candidate on unseen event'
    registry = ReviewedSources(tmp_path)
    registry.register(item, review())
    evidence = {'evidence_id': 'e', 'content': 'old', 'provenance': {},
                'source_uri': 'paper://p/layout/o'}
    assert registry.apply(evidence, item.source)
    annotate_observation(evidence, region=item.source, parser='source')
    catalog = row_catalog([evidence])
    assert catalog[0]['checked_columns'] is None
    assert catalog[0]['component_columns'] == []
    draft = {'facts': [{'id': 'f1', 'kind': 'TABLE_FACT', 'operation': 'value',
        'row_refs': [{'evidence_id': 'e', 'row_id': catalog[0]['row_id'], 'metrics': ['F1 (%)']}]}],
        'direct_answer_id': 'f1', 'limitations': []}
    result, _ = project_table_facts(draft, catalog, chinese=True)
    assert '62.30' in result['facts'][0]['text']
    assert result['facts'][0]['kind'] == 'SOURCE_FACT'


def test_formal_extraction_reuses_asset_before_cloud_and_retains_row_catalog(tmp_path, monkeypatch):
    from paperpilot_agentic import builtin_tools
    from paperpilot_agentic.db import RuntimeDatabase
    from paperpilot_agentic.models import Budget
    from paperpilot_agentic.tooling import ToolContext
    from paperpilot_agentic.structured_extraction import file_sha
    from paperpilot_agentic.table_claims import row_catalog
    import json
    pdf, image = tmp_path/'p.pdf', tmp_path/'crop.png'
    pdf.write_bytes(b'pdf identity fixture'); image.write_bytes(b'crop identity fixture')
    item = artifact()
    item.source.paper_sha256 = file_sha(pdf); item.source.crop_sha256 = file_sha(image)
    registry = ReviewedSources(tmp_path/'registry'); registry.register(item, review())
    monkeypatch.setenv('PAPERPILOT_REVIEWED_SOURCE_DIR', str(registry.root))
    obj = {'object_id': 'o', 'object_type': 'table', 'page': 2, 'bbox': item.source.bbox,
           'caption': 'Table I', 'artifact_path': str(image), 'document_sha256': file_sha(pdf)}
    def parse(pdf, output, **kwargs):
        (output/'layout_manifest.json').write_text(json.dumps({'objects': [obj]}))
        return {'objects': [obj]}
    monkeypatch.setattr(builtin_tools, 'parse_pdf_layout', parse)
    monkeypatch.setattr(builtin_tools, 'search_layout_objects', lambda *a, **k: [obj])
    class Cloud:
        def analyze(self, *args, **kwargs):
            raise AssertionError('No cloud request allowed for matching registered source')
    db = RuntimeDatabase(tmp_path/'runtime.sqlite3')
    rid = db.create_run('Read Table I', Budget())
    result = builtin_tools._layout_handler(Cloud())(ToolContext(rid, 's', 't', tmp_path, db, None),
        {'pdf_path': str(pdf), 'query': 'Table I', 'object_type': 'table', 'gap': 'numeric_results'})
    assert result.output['reviewed_source_hits'] == 1 and result.output['vision_api_calls'] == 0
    source = {**result.evidence[0], 'evidence_id': 'e'}
    catalog = row_catalog([source])
    assert catalog[0]['cells']['F1 (%)'] == '62.30'
    assert catalog[0]['source_review']['review']['reviewer_kind'] == 'ai'
