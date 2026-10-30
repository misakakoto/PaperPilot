"""Preserve nearby caption text without inferring a dataset or task."""


def nearby_caption(manifest, item):
    if hasattr(manifest, 'to_dict'):
        manifest = manifest.to_dict()
    box = item.get('bbox') or []
    if len(box) != 4:
        return None
    center = (box[0] + box[2]) / 2
    candidates = []
    for row in manifest.get('objects', []):
        bounds = row.get('bbox') or []
        if (row.get('object_type') != 'text' or row.get('page') != item.get('page')
                or row.get('document_sha256') != item.get('document_sha256')
                or len(bounds) != 4 or not row.get('text')):
            continue
        if bounds[0] <= center <= bounds[2] and 0 <= bounds[1] - box[3] <= 24:
            candidates.append(row)
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: row['bbox'][1])
    return {key: selected.get(key) for key in
            ('object_id', 'document_sha256', 'page', 'bbox', 'text')}
