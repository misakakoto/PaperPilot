"""Immutable literal captions/legends bound to an existing reviewed crop.

These are source transcriptions, not answer keys or support verdicts. The host
registers them after inspecting the same crop; clients cannot supply paths.
"""
import json
import re

from .structured_extraction import SourceRegion, config_sha

VERSION = 'reviewed-source-annotation-v1'


def _folder(root, parent):
    if not isinstance(parent, str) or not re.fullmatch(r'[0-9a-f]{64}', parent):
        raise ValueError('Expected a registered source asset identity')
    return root / 'annotations' / parent


def _validate(value, parent, region):
    if (not isinstance(value, dict) or set(value) != {
            'version', 'parent_asset_id', 'source', 'location', 'kind', 'text', 'review'}
            or value['version'] != VERSION or value['parent_asset_id'] != parent
            or SourceRegion.model_validate(value['source']) != region
            or value['location'] != 'bound_source_crop'
            or value['kind'] not in {'caption', 'legend', 'footnote'}
            or not isinstance(value['text'], str) or not value['text'].strip()
            or len(value['text']) > 16000):
        raise ValueError('Literal annotation source identity or schema mismatch')
    review = value['review']
    if (not isinstance(review, dict) or set(review) != {
            'reviewer_id', 'reviewer_kind', 'reviewed_at', 'checks'}
            or review['reviewer_kind'] not in {'ai', 'human'}
            or any(not isinstance(review[k], str) or not review[k].strip()
                   for k in ['reviewer_id', 'reviewed_at'])
            or not isinstance(review['checks'], list) or len(review['checks']) != 3
            or any(not isinstance(c, dict) or set(c) != {'dimension', 'result', 'note'}
                   or c['result'] != 'PASS' or not isinstance(c['note'], str) or not c['note'].strip()
                   for c in review['checks'])
            or {c['dimension'] for c in review['checks']} != {'identity', 'location', 'literal_text'}):
        raise ValueError('Literal annotation needs explicit source/text/location review')


def read_annotations(root, parent, region):
    result = []
    for path in sorted(_folder(root, parent).glob('*.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        if path.stem != config_sha(value):
            raise ValueError('Literal annotation identity mismatch')
        _validate(value, parent, region)
        if any(v['kind'] == value['kind'] for v in result):
            raise ValueError('Conflicting literal annotations for one crop and kind')
        result.append({'annotation_id': path.stem, **value})
    return result


def register_annotation(root, parent, region, *, kind, text, review):
    value = {'version': VERSION, 'parent_asset_id': parent,
             'source': region.model_dump(mode='json'), 'location': 'bound_source_crop',
             'kind': kind, 'text': text, 'review': review}
    _validate(value, parent, region)
    identifier = config_sha(value)
    existing = read_annotations(root, parent, region)
    if any(a['kind'] == kind and a['annotation_id'] != identifier for a in existing):
        raise ValueError('Conflicting literal annotation; preserve original records')
    folder = _folder(root, parent)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identifier + '.json')
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Literal annotation identity conflict')
    return identifier
