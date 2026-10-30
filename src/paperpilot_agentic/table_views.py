"""Source-bound row crops for simple tables; no OCR or cell correction."""
from io import BytesIO
import hashlib
from pathlib import Path


def row_focus_view(path: Path) -> tuple[bytes | None, dict]:
    """Space uniform numeric rows apart, preserving every source pixel in a row.

    Only three-rule tables with equally spaced numeric bands qualify. Internal
    rules/merged groups, ambiguous bands and raster-poor regions retain the
    original view. This changes evidence presentation, never a cell value or
    the downstream pixel audit's original image.
    """
    try:
        import numpy as np
        from PIL import Image
        source = Image.open(path).convert('RGB')
    except (ImportError, OSError):
        return None, {'status': 'unavailable'}
    scale = min(1.0, 650/source.width)
    probe = source.resize((round(source.width*scale), round(source.height*scale)))
    dark = np.asarray(probe.convert('L')) < 160
    ys = np.flatnonzero(dark.sum(axis=1) >= max(40, int(probe.width*.45)))
    rules = []
    for y in ys:
        if not rules or y-rules[-1][-1] > 8:
            rules.append([int(y)])
        else:
            rules[-1].append(int(y))
    if len(rules) != 3:
        return None, {'status': 'not_simple_three_rule_table'}
    top, bottom = max(rules[1]), min(rules[2])
    rule_x = np.flatnonzero(dark[top] | dark[bottom])
    if not len(rule_x) or bottom-top < 30:
        return None, {'status': 'ambiguous_table_extent'}
    left, right = int(rule_x[0]), int(rule_x[-1])+1
    # A rightmost numeric column supplies row centers independently of the
    # model's checkmarks. Never select geometry by agreement with its output.
    metric_left = round(left+(right-left)*.75)
    ys = np.flatnonzero(dark[top+3:bottom-3, metric_left:right-2].sum(axis=1) >= 3)+top+3
    bands = []
    for y in ys:
        if not bands or y-bands[-1][-1] > 3:
            bands.append([int(y)])
        else:
            bands[-1].append(int(y))
    if not 3 <= len(bands) <= 30 or any(not 4 <= len(band) <= 20 for band in bands):
        return None, {'status': 'ambiguous_numeric_row_bands'}
    centers = [(band[0]+band[-1])/2 for band in bands]
    gaps = [b-a for a, b in zip(centers, centers[1:])]
    if min(gaps) < 8 or max(gaps) > min(gaps)*1.25:
        return None, {'status': 'nonuniform_numeric_rows'}
    edges = [top+3, *[round((a+b)/2) for a, b in zip(centers, centers[1:])], bottom-3]
    if any(band[0] < edges[i] or band[-1] >= edges[i+1] for i, band in enumerate(bands)):
        return None, {'status': 'row_band_crosses_crop'}
    def native_box(y0, y1):
        return [round(left/scale), round(y0/scale), round(right/scale), round(y1/scale)]
    header_box = native_box(max(0, min(rules[0])-3), top+2)
    row_boxes = [native_box(a, b) for a, b in zip(edges, edges[1:])]
    boxes = [header_box, *row_boxes]
    margin, gap = 12, 20
    width = boxes[0][2]-boxes[0][0]
    height = sum(box[3]-box[1] for box in boxes)+2*margin+gap*(len(boxes)-1)
    canvas = Image.new('RGB', (width+2*margin, height), 'white')
    y = margin
    for box in boxes:
        crop = source.crop(box)
        canvas.paste(crop, (margin, y))
        y += crop.height+gap
    buffer = BytesIO()
    canvas.save(buffer, format='PNG')
    data = buffer.getvalue()
    return data, {'status': 'source_row_crops', 'header_bbox_pixels': header_box,
                  'row_bboxes_pixels': row_boxes, 'numeric_row_count': len(row_boxes),
                  'derived_image_sha256': hashlib.sha256(data).hexdigest(),
                  'operation': 'crop_and_white_spacing_only', 'cell_correction': False}
