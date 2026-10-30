from paperpilot_agentic.table_context import nearby_caption


def test_caption_is_nearest_same_source_column_text():
    item = {'bbox': [150, 190, 190, 200], 'page': 3, 'document_sha256': 'source'}
    text = {'object_type': 'text', 'bbox': [50, 205, 290, 220], 'page': 3,
            'document_sha256': 'source', 'text': 'Ablation on dataset X', 'object_id': 'caption'}
    distractors = [{**text, 'page': 4}, {**text, 'document_sha256': 'other'},
                  {**text, 'bbox': [310, 201, 500, 220]},
                  {**text, 'bbox': [50, 201, 290, 220], 'object_type': 'table'}]
    assert nearby_caption({'objects': distractors}, item) is None
    assert nearby_caption({'objects': [*distractors, text]}, item)['object_id'] == 'caption'
    from paperpilot_agentic.layout import LayoutManifest, LayoutObject
    fresh = LayoutManifest('fixture', 'source', 'test', 3, [LayoutObject(**text)])
    assert nearby_caption(fresh, item)['object_id'] == 'caption'
