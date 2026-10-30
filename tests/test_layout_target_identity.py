"""Exercise real layout selection before any paid visual extraction."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from paperpilot_agentic.builtin_tools import _layout_handler
from paperpilot_agentic.db import RuntimeDatabase
from paperpilot_agentic.models import Budget


def invoke_layout(tmp_path, objects, query, *, top_k=1):
    pdf = tmp_path / 'fixture.pdf'
    pdf.write_bytes(b'synthetic fixture; cached manifest, no PDF parsing')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    db = RuntimeDatabase(tmp_path / 'runtime.sqlite3')
    rid = db.create_run(query, Budget())
    output = tmp_path / '.paperpilot_agentic' / 'runs' / rid / 'layout' / f'fixture-{sha[:12]}'
    output.mkdir(parents=True)
    manifest = {'parser': 'synthetic', 'warnings': [], 'objects': [
        {'page': index + 1, 'bbox': [0, 0, 100, 100], 'document_sha256': sha,
         'metadata': {'rows': 0}, **row} for index, row in enumerate(objects)]}
    from PIL import Image
    for row in manifest['objects']:
        if row.get('artifact_path'):
            image = tmp_path / row['artifact_path']
            Image.new('RGB', (100, 100), 'white').save(image)
            row['artifact_path'] = str(image)
    (output / 'layout_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    calls = []

    class Vision:
        def analyze(self, path, prompt, metadata):
            calls.append(metadata['object_id'])
            return {'verified': True, 'description': 'synthetic transcription', 'metadata': {}}

    context = SimpleNamespace(workspace=tmp_path, db=db, run_id=rid)
    result = _layout_handler(Vision())(context, {
        'pdf_path': str(pdf), 'object_type': 'table', 'query': query, 'top_k': top_k})
    return result, calls


def table(identity, caption='', text=''):
    return {'object_id': identity, 'object_type': 'table', 'caption': caption,
            'text': text, 'artifact_path': f'{identity}.png'}


@pytest.mark.parametrize('row', [
    table('other', 'Table 2. Human evaluation'),
    table('placeholder', 'Table TABREF35. Human evaluation'),
    table('prose', '', 'See Table 35 for human evaluation.'),
])
def test_absent_table_is_not_replaced_by_an_unrelated_crop(tmp_path, row):
    result, calls = invoke_layout(tmp_path, [row], 'Table 35. Transcribe human evaluation.')
    assert result.evidence == [], 'An absent table must not acquire another object identity'
    assert calls == [], 'Do not spend a visual request on a different or unbound table'


def test_named_table_is_selected_before_top_k_ranking(tmp_path):
    rows = [table('decoy', 'Table 2. Ablation quantitative baseline comparison'),
            table('target', 'Table 35. Human evaluation')]
    result, calls = invoke_layout(tmp_path, rows, 'Table 35 ablation quantitative baseline comparison')
    assert calls == ['target']
    assert result.evidence[0]['source_uri'].endswith('/target')


def test_nearby_table_mention_cannot_name_an_unlabelled_crop(tmp_path):
    rows = [table('unknown'), {'object_id': 'discussion', 'object_type': 'text',
                              'text': 'See Table 35 for results.', 'page': 1}]
    result, calls = invoke_layout(tmp_path, rows, 'Table 35')
    assert result.evidence == []
    assert calls == []


def test_placeholder_only_query_cannot_select_printed_number(tmp_path):
    result, calls = invoke_layout(tmp_path, [table('printed', 'Table 35')], 'Table TABREF35')
    assert result.evidence == []
    assert calls == []
    assert result.output['layout_target_status'] == 'UNRESOLVED_REFERENCE'


def test_caption_takes_precedence_over_cross_references_in_cells(tmp_path):
    result, calls = invoke_layout(tmp_path, [table('other', 'Table 2', 'Table 35 is the baseline.')], 'Table 35')
    assert result.evidence == []
    assert calls == []


def test_ordinary_semantic_search_remains_available(tmp_path):
    result, calls = invoke_layout(tmp_path, [table('result', 'Table 2. Human evaluation')], 'Human evaluation')
    assert calls == ['result']
    assert result.output['layout_target_status'] == 'SEMANTIC_SEARCH'


def test_exact_table_records_source_owned_binding(tmp_path):
    result, calls = invoke_layout(tmp_path, [table('target', 'Table XII. Human evaluation')], 'Table XII')
    assert calls == ['target']
    binding = result.evidence[0]['provenance']['layout_target_binding']
    assert binding['label'] == ['table', 'XII']
    assert binding['basis'] == 'own_caption_or_heading'
    assert result.evidence[0]['source_uri'] == f"paper://{binding['document_sha256']}/layout/{binding['object_id']}"
