"""Failing output patterns from live acceptance, exercised without a provider."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from paperpilot_agentic.qwen_vision import (
    QwenTableRow, QwenVisualExtraction, QwenVisionEvidenceAdapter,
    _audit_indicator_pixels, _validate_table_structure,
)


def payload(**changes):
    value = {
        'description': 'A visible table', 'transcription': '',
        'visible_values': ['80.00'], 'column_headers': ['Method', 'Score'],
        'table_rows': [{'row_label': 'A', 'checked_columns': [],
                        'cells': {'Method': 'A', 'Score': '80.00'}}],
        'uncertain_fields': [], 'image_readable': True,
        'evidence_relevant': True, 'confidence': 1.0,
    }
    return {**value, **changes}


def adapter(value):
    requests = []
    def send(**kwargs):
        requests.append(kwargs)
        response_value = ({'column_headers': value.get('column_headers', []),
                           'indicator_columns': value.get('indicator_columns', []),
                           'group_headers': [], 'uncertain_fields': []}
                          if kwargs['max_tokens'] == 600 else value)
        return SimpleNamespace(
            model='fixture', choices=[SimpleNamespace(finish_reason='stop',
                message=SimpleNamespace(content=json.dumps(response_value)))],
            usage=SimpleNamespace(prompt_tokens=657, completion_tokens=777, total_tokens=1434))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
    return QwenVisionEvidenceAdapter(api_key='fixture', client=client), requests


def test_positional_cells_are_mapped_only_by_complete_unique_leaf_headers(tmp_path):
    image = tmp_path/'table.png'
    Image.new('L', (100, 100), 255).save(image)
    vision, requests = adapter(payload(table_rows=[
        {'row_label': 'A', 'checked_columns': [], 'cells': ['A', '80.00']}]))
    result = vision.analyze(str(image), 'Read this table', {'object_type': 'table'})
    assert result['verified']
    assert result['metadata']['table_rows'][0]['cells'] == {'Method': 'A', 'Score': '80.00'}
    assert result['metadata']['schema_normalizations'] == [
        {'row_index': 0, 'operation': 'positional_cells_to_named_cells'}]
    assert len(requests) == 1


@pytest.mark.parametrize('headers,cells', [
    (['Group', 'A', 'B', 'Score'], ['', '80.00']),
    (['Score', 'Score'], ['80.00', '80.00']),
    ([], ['80.00']),
    (['Method', 'Score'], ['A', 80]),
])
def test_ambiguous_arrays_remain_rejected_and_paid_usage_is_preserved(tmp_path, headers, cells):
    image = tmp_path/'table.png'
    Image.new('L', (100, 100), 255).save(image)
    vision, requests = adapter(payload(column_headers=headers, table_rows=[
        {'row_label': '', 'checked_columns': [], 'cells': cells}]))
    with pytest.raises(ValueError) as failure:
        vision.analyze(str(image), 'Read this table', {'object_type': 'table'})
    assert failure.value.vision_usage['prompt_tokens'] == 657
    assert failure.value.vision_usage['completion_tokens'] == 777
    assert failure.value.vision_schema_issues[0]['path'] == ['table_rows', 0, 'cells']
    assert '80.00' not in str(failure.value)
    assert len(requests) == 1


def test_paid_response_without_choices_keeps_its_usage(tmp_path):
    source = tmp_path/'table.png'
    Image.new('L', (100, 100), 255).save(source)
    raw = SimpleNamespace(choices=[], model='fixture',
                          usage=SimpleNamespace(prompt_tokens=657, completion_tokens=777))
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: raw)))
    vision = QwenVisionEvidenceAdapter(api_key='fixture', client=sdk)
    with pytest.raises(ValueError) as failure:
        vision.analyze(str(source), 'Read', {'object_type': 'table'})
    assert failure.value.vision_usage['prompt_tokens'] == 657
    assert failure.value.vision_usage['completion_tokens'] == 777


@pytest.mark.parametrize('changes,reason', [
    ({'table_rows': [
        {'row_label': '', 'checked_columns': [], 'cells': {'Method': '', 'Score': ''}},
        {'row_label': 'A', 'checked_columns': [], 'cells': {'Method': 'A', 'Score': '80.00'}}]},
     'empty_data_row'),
    ({'column_headers': ['Group', 'Method', 'Score']}, 'header_cell_coverage_mismatch'),
    ({'column_headers': ['Method', 'Score', 'Score']}, 'duplicate_column_headers'),
    ({'column_headers': ['Method', 'Other']}, 'header_cell_coverage_mismatch'),
])
def test_structural_audit_rejects_header_and_empty_row_patterns(changes, reason):
    extraction = QwenVisualExtraction.model_validate(payload(**changes))
    valid, actual = _validate_table_structure(extraction)
    assert not valid
    assert actual == reason


def test_declared_indicator_is_audited_even_when_model_omits_every_checkmark(tmp_path):
    image = Image.new('L', (220, 160), 255)
    draw = ImageDraw.Draw(image)
    for y in (20, 22, 60, 62, 140, 142):
        draw.line((20, y, 200, y), fill=0)
    # Two rows; the second row enables A. The model reports no marks at all.
    draw.line((60, 118, 64, 123), fill=0, width=2)
    draw.line((64, 123, 73, 112), fill=0, width=2)
    source = tmp_path/'omitted-indicator.png'
    image.save(source)
    vision, _ = adapter(payload(column_headers=['A', 'Score'], indicator_columns=['A'],
        visible_values=['80.00', '81.00'], table_rows=[
            {'row_label': '', 'checked_columns': [], 'cells': {'A': '', 'Score': score}}
            for score in ('80.00', '81.00')]))
    result = vision.analyze(str(source), 'Read the ablation table', {'object_type': 'table'})
    assert result['verified']
    assert [r['checked_columns'] for r in result['metadata']['audited_table_rows']] == [[], ['A']]
    assert result['metadata']['indicator_pixel_audit']['status'] == 'corrected'


def test_pixel_audit_does_not_reorder_nonleading_indicators_into_leftmost_columns(tmp_path):
    image = Image.new('L', (220, 160), 255)
    draw = ImageDraw.Draw(image)
    for y in (20, 22, 60, 62, 140, 142):
        draw.line((20, y, 200, y), fill=0)
    # The score is on the LEFT; its digits must not become a checkmark in
    # the empty Enabled column on the RIGHT.
    draw.text((54, 93), '80.00', fill=0)
    source = tmp_path/'nonleading-indicator.png'
    image.save(source)
    rows = [QwenTableRow(checked_columns=['Enabled'], cells={'Score': '80.00', 'Enabled': '✓'})]
    valid, reason, _, _ = _audit_indicator_pixels(source, rows,
        column_headers=['Score', 'Enabled'])
    assert not valid
    assert reason == 'unsupported_indicator_column_layout'


def test_missing_indicator_roles_cannot_hide_an_entire_empty_condition_column():
    extraction = QwenVisualExtraction.model_validate(payload(column_headers=['Flag', 'Score'],
        table_rows=[{'row_label': '', 'checked_columns': [],
                     'cells': {'Flag': '', 'Score': '80.00'}}]))
    valid, reason = _validate_table_structure(extraction)
    assert not valid
    assert reason == 'unclassified_empty_leading_column'


def test_correct_numeric_values_do_not_verify_shifted_cumulative_conditions(tmp_path):
    image = Image.new('L', (350, 200), 255)
    draw = ImageDraw.Draw(image)
    for y in (20, 22, 60, 62, 160, 162):
        draw.line((20, y, 330, y), fill=0)
    headers = ['A', 'B', 'C', 'Score']
    for index, header in enumerate(headers):
        draw.text((50 + index*77, 45), header, fill=0)
    expected = [[], [0], [1], [2], [0, 1, 2]]
    scores = ['80.00', '81.00', '82.00', '83.00', '84.00']
    for row_index, checks in enumerate(expected):
        cy = 62 + 98*(row_index+0.5)/5
        for column in checks:
            cx = 20 + 310*(column+0.5)/4
            draw.line((cx-5, cy, cx-1, cy+4), fill=0, width=2)
            draw.line((cx-1, cy+4, cx+6, cy-5), fill=0, width=2)
        draw.text((286, cy-4), scores[row_index], fill=0)
    source = tmp_path/'shifted-conditions.png'
    image.save(source)
    shifted = [['A'], ['A', 'B'], ['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C']]
    rows = [{'row_label': '', 'checked_columns': checks,
             'cells': {**{h: '✓' if h in checks else '' for h in headers[:3]}, 'Score': score}}
            for checks, score in zip(shifted, scores, strict=True)]
    vision, requests = adapter(payload(column_headers=headers, indicator_columns=headers[:3],
        visible_values=scores, table_rows=rows))
    result = vision.analyze(str(source), 'Read the ablation table', {'object_type': 'table'})
    assert not result['verified']
    assert 'indicator_pixel_model_conflict' in result['metadata']['table_structure_reason']
    assert result['metadata']['table_rows'] == rows
    assert result['metadata']['reported_confidence'] == 1.0
    assert len(requests) == 2
    assert result['metadata']['provider_attempts'] == 2
    assert result['metadata']['total_tokens'] == 2868
