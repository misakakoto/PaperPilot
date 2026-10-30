from pathlib import Path

import pytest
from PIL import Image

from paperpilot_agentic.layout import parse_pdf_layout


def test_caption_table_crop_retains_native_vector_detail_at_recorded_scale(tmp_path: Path):
    fitz = pytest.importorskip('pymupdf')
    pdf = tmp_path / 'source.pdf'
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=400)
        page.insert_text((30, 60), 'TABLE I', fontsize=10)
        page.draw_line((30, 80), (270, 80))
        page.insert_text((80, 98), 'Flag   Score', fontsize=7)
        page.insert_text((120, 118), '81.11', fontsize=7)
        doc.save(pdf)
    manifest = parse_pdf_layout(pdf, tmp_path / 'layout', render_regions=True).to_dict()
    table = next(obj for obj in manifest['objects'] if obj['object_type'] == 'table'
                 and obj['metadata'].get('caption_only'))
    assert table['metadata']['render_scale'] == 4.0
    bbox = table['metadata']['render_context_bbox']
    with Image.open(table['artifact_path']) as image:
        assert abs(image.width - (bbox[2]-bbox[0])*4) <= 2
        assert abs(image.height - (bbox[3]-bbox[1])*4) <= 2
    assert table['document_sha256'] == manifest['document_sha256']
