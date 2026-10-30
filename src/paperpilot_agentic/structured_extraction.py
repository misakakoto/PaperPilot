"""Versioned source observations. Parsing is never independent factual review."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def file_sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


class SourceRegion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    paper_sha256: str
    object_id: str
    label: str = ''
    page: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    coordinate_space: Literal['pdf_points_top_left'] = 'pdf_points_top_left'
    crop_sha256: str

    @model_validator(mode='after')
    def valid_box(self):
        x0, y0, x1, y1 = self.bbox
        if not all(math.isfinite(v) for v in self.bbox) or not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError('Invalid source region')
        return self


class TableCell(BaseModel):
    model_config = ConfigDict(extra='forbid')
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str
    column_header: bool = False
    row_header: bool = False
    bbox: dict[str, Any] | None = None


class ParsedArtifact(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal['structured-region-v1'] = 'structured-region-v1'
    kind: Literal['table', 'equation', 'figure']
    source: SourceRegion
    parser: str
    parser_version: str
    config_sha256: str
    weights_sha256: str | None = None
    parse_status: Literal['PARSED', 'FAILED']
    validation_status: Literal['VALIDATED', 'UNVERIFIED', 'CONFLICT']
    factual_review: Literal['NOT_REVIEWED'] = 'NOT_REVIEWED'
    column_headers: list[str] = Field(default_factory=list)
    header_paths: list[list[str]] = Field(default_factory=list)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    latex: str | None = None
    raw_artifact_path: str | None = None
    uncertain_fields: list[str] = Field(default_factory=list)
    parser_notes: list[str] = Field(default_factory=list)


def annotate_observation(item, *, region, parser):
    """Project legacy validated rows into a typed observation, without rewriting raw output."""
    provenance = item['provenance']
    meta = provenance.get('vision_metadata') or {}
    rows = meta.get('audited_table_rows') or []
    provided = meta.get('structured_extraction')
    if provided:
        observation = ParsedArtifact.model_validate(provided)
        if observation.source != region:
            raise ValueError('Parser returned a different source region')
    else:
        observation = ParsedArtifact(kind=item['object_type'], source=region,
            parser=parser, parser_version=str(meta.get('parser_version') or meta.get('model') or 'legacy'),
            config_sha256=config_sha({'model': meta.get('model'), 'schema': 'legacy-named-cells',
                                     **({'uncertainty_policy': meta['uncertainty_policy']}
                                        if meta.get('uncertainty_policy') else {})}),
            parse_status='PARSED' if meta else 'FAILED',
            validation_status='VALIDATED' if provenance.get('vision_verified') else 'UNVERIFIED',
            column_headers=meta.get('column_headers') or [], table_rows=rows,
            uncertain_fields=meta.get('uncertain_fields') or [], parser_notes=meta.get('parser_notes') or [])
    provenance['structured_extraction'] = observation.model_dump(mode='json')


def table_observation_status(item, paper_sha, label) -> str:
    """Status is source-bound data, independent of prose formatting. Legacy reads stay read-only."""
    provenance = item.get('provenance') or {}
    binding = provenance.get('layout_target_binding') or {}
    oid = provenance.get('layout_object_id')
    if (not oid or binding.get('document_sha256') != paper_sha or
            binding.get('object_id') != oid or binding.get('label') != ['table', label.split()[-1]] or
            binding.get('basis') != 'own_caption_or_heading' or
            item.get('source_uri') != f'paper://{paper_sha}/layout/{oid}'):
        return 'NOT_VERIFIED'
    payload = provenance.get('structured_extraction')
    if payload is not None:
        try:
            artifact = ParsedArtifact.model_validate(payload)
        except ValueError:
            return 'NOT_VERIFIED'
        if (artifact.kind == 'table' and artifact.source.paper_sha256 == paper_sha and
                artifact.source.object_id == oid and artifact.source.label == label and
                artifact.parse_status == 'PARSED' and artifact.validation_status == 'VALIDATED' and
                artifact.table_rows):
            return 'TRANSCRIBED'
        return 'NOT_VERIFIED'
    meta = provenance.get('vision_metadata') or {}
    return 'TRANSCRIBED' if provenance.get('vision_verified') and meta.get('audited_table_rows') else 'NOT_VERIFIED'
