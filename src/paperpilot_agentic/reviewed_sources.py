"""Immutable, source-bound literal transcriptions with an explicit review sidecar.

These are source assets, not answers. Review describes who checked the source;
it does not grant any answer, inference or automatic parser a correctness verdict.
"""
import json
import os
from pathlib import Path
from .structured_extraction import ParsedArtifact, config_sha

VERSION = 'reviewed-source-v1'


class ReviewedSources:
    def __init__(self, root):
        self.root = Path(root)

    @classmethod
    def configured(cls):
        default = Path(__file__).resolve().parents[2]/'outputs/reviewed-sources'
        return cls(os.environ.get('PAPERPILOT_REVIEWED_SOURCE_DIR', str(default)))

    def identity(self):
        values = {p.stem: config_sha(json.loads(p.read_text(encoding='utf-8')))
                  for p in sorted(self.root.glob('*.json'))}
        values.update({p.relative_to(self.root).as_posix():
                       config_sha(json.loads(p.read_text(encoding='utf-8')))
                       for p in sorted((self.root / 'annotations').rglob('*.json'))})
        return config_sha(values)

    def register_annotation(self, asset_id, region, *, kind, text, review):
        from .reviewed_annotations import register_annotation
        asset = self.resolve(region)
        if asset is None or asset['asset_id'] != asset_id:
            raise ValueError('Literal annotation must bind the exact registered source identity')
        return register_annotation(self.root, asset_id, region, kind=kind, text=text, review=review)

    def register(self, artifact, review):
        artifact = ParsedArtifact.model_validate(artifact)
        if (artifact.kind != 'table' or artifact.parse_status != 'PARSED' or
                artifact.validation_status != 'VALIDATED' or not artifact.table_rows or artifact.uncertain_fields):
            raise ValueError('Register only a complete, structurally validated literal table')
        if (set(review) != {'reviewer_id', 'reviewer_kind', 'reviewed_at', 'checks'} or
                review['reviewer_kind'] not in {'ai', 'human'} or
                not all(isinstance(review[k], str) and review[k].strip()
                        for k in ('reviewer_id', 'reviewed_at')) or not isinstance(review['checks'], list)):
            raise ValueError('Explicit source reviewer and checks required')
        required = {'identity', 'values', 'conditions', 'headers', 'spans', 'units'}
        if ({c.get('dimension') for c in review['checks']} != required or
                any(set(c) != {'dimension', 'result', 'note'} or c['result'] != 'PASS' or
                    not isinstance(c['note'], str) or not c['note'].strip() for c in review['checks'])):
            raise ValueError('Every source dimension needs a recorded passing review')
        payload = {'version': VERSION, 'artifact': artifact.model_dump(mode='json'), 'review': review}
        asset_id = config_sha(payload)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (asset_id + '.json')
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != payload:
                raise ValueError('Immutable source asset conflict')
        else:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
        return asset_id

    def resolve(self, region):
        """Only the host's exact region selects assets; callers cannot pass paths."""
        if not self.root.exists():
            return None
        matches = []
        for path in self.root.glob('*.json'):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                if payload.get('version') != VERSION or path.stem != config_sha(payload):
                    continue
                artifact = ParsedArtifact.model_validate(payload['artifact'])
                if artifact.source != region:
                    continue
                # Revalidate using the same registration contract without writing.
                review = payload['review']
                if (review.get('reviewer_kind') not in {'ai', 'human'} or
                        artifact.validation_status != 'VALIDATED' or artifact.uncertain_fields):
                    continue
                matches.append({'asset_id': path.stem, **payload})
            except (ValueError, KeyError, OSError):
                continue
        if len(matches) > 1:
            raise ValueError('Conflicting reviewed assets for one source region')
        return matches[0] if matches else None

    def apply(self, evidence, region):
        asset = self.resolve(region)
        if asset is None:
            return False
        from .reviewed_annotations import read_annotations
        notes = read_annotations(self.root, asset['asset_id'], region)
        artifact = asset['artifact']
        literal = {'column_headers': artifact['column_headers'], 'rows': artifact['table_rows']}
        if notes:
            literal['literal_source_annotations'] = notes
        evidence['content'] = region.label + '\nLiteral source transcription:\n' + json.dumps(
            literal, ensure_ascii=False)
        evidence['confidence'] = 0.0  # No invented confidence probability.
        evidence['provenance'].update(analysis_mode='reviewed_source_asset',
            reviewed_source={'asset_id': asset['asset_id'], 'version': VERSION, 'review': asset['review']},
            vision_verified=True, vision_metadata={'structured_extraction': artifact,
                'audited_table_rows': artifact['table_rows'], 'column_headers': artifact['column_headers'],
                'parser_version': artifact['parser_version'], 'uncertain_fields': []})
        if notes:
            evidence['provenance']['vision_metadata']['literal_source_annotations'] = notes
        return True
