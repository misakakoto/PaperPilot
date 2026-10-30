from io import BytesIO

from PIL import Image, ImageDraw

from paperpilot_agentic.table_views import row_focus_view


def fixture(tmp_path, *, internal_rule=False):
    path = tmp_path/'table.png'
    image = Image.new('RGB', (600, 260), 'white')
    draw = ImageDraw.Draw(image)
    for y in (20, 80, 230):
        draw.line((30, y, 570, y), fill='black')
    draw.text((60, 42), 'Flag A     Flag B       Score', fill='black')
    for i, y in enumerate((100, 130, 160, 190)):
        draw.text((490, y), f'{80+i}.11', fill='black')
        if i:
            draw.line((100, y+5, 109, y+13), fill='black', width=2)
    if internal_rule:
        draw.line((30, 148, 570, 148), fill='black')
    image.save(path)
    return path


def test_focus_preserves_source_row_pixels_including_blank_first_indicator(tmp_path):
    path = fixture(tmp_path)
    data, metadata = row_focus_view(path)
    assert data is not None, metadata
    assert metadata['numeric_row_count'] == 4
    assert metadata['cell_correction'] is False
    with Image.open(path).convert('RGB') as source, Image.open(BytesIO(data)) as result:
        y = 12
        for box in [metadata['header_bbox_pixels'], *metadata['row_bboxes_pixels']]:
            crop = source.crop(box)
            assert result.crop((12, y, 12+crop.width, y+crop.height)).tobytes() == crop.tobytes()
            y += crop.height+20


def test_internal_rules_do_not_produce_row_view_for_merged_groups(tmp_path):
    data, metadata = row_focus_view(fixture(tmp_path, internal_rule=True))
    assert data is None
    assert metadata['status'] == 'not_simple_three_rule_table'


def test_unreadable_region_retains_original_image(tmp_path):
    path = tmp_path/'invalid.png'
    path.write_bytes(b'not an image')
    assert row_focus_view(path)[0] is None


def test_collapsed_group_column_cannot_match_a_header_included_as_data(tmp_path):
    from paperpilot_agentic.qwen_vision import QwenTableRow, _audit_indicator_pixels
    path = fixture(tmp_path)
    rows = [QwenTableRow(checked_columns=[] if i == 0 else ['Flags'],
                        cells={'Flags': '' if i == 0 else '✓', 'Score': f'{80+i}.11'})
            for i in range(4)]
    valid, _, _, details = _audit_indicator_pixels(path, rows,
                    column_headers=['Flags', 'Score'], indicator_columns=['Flags'])
    assert not valid or details['table_bbox_pixels'][1] >= 78
