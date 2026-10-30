"""Bounded Qwen-VL bridge for scholarly figures, tables and equations."""

from __future__ import annotations

import base64
import copy
import hashlib
import itertools
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from .config import read_dashscope_api_key_file
from .errors import ConfigurationError

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class QwenTableRow(BaseModel):
    """One row with explicit column associations for auditable tables."""

    model_config = ConfigDict(extra="forbid")

    row_label: str = Field(default="", max_length=1_000)
    checked_columns: list[str] = Field(default_factory=list, max_length=50)
    cells: dict[str, str] = Field(default_factory=dict)


class QwenVisualExtraction(BaseModel):
    """Strict semantic contract returned by the remote vision model."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=12_000)
    transcription: str = Field(default="", max_length=12_000)
    visible_values: list[str] = Field(default_factory=list, max_length=200)
    table_rows: list[QwenTableRow] = Field(default_factory=list, max_length=200)
    column_headers: list[str] = Field(default_factory=list, max_length=50)
    indicator_columns: list[str] = Field(default_factory=list, max_length=12)
    uncertain_fields: list[str] = Field(default_factory=list, max_length=100)
    parser_notes: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                                     max_length=1000)]] = Field(default_factory=list, max_length=100)
    image_readable: bool
    evidence_relevant: bool
    confidence: float = Field(ge=0.0, le=1.0)


class VisionExtractionError(ValueError):
    """A returned, paid response violated the contract; its usage stays attached."""


def _normalize_named_cells(value: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize lossless representation differences; never resolve uncertainty."""
    result = dict(value)
    headers = result.get('column_headers')
    changes: list[dict[str, Any]] = []
    notes = result.get('parser_notes')
    if isinstance(notes, str) and notes.strip():
        result['parser_notes'] = [notes]
        changes.append({'operation': 'scalar_note_to_array', 'original_value': notes})
    uncertain = result.get('uncertain_fields')
    if (isinstance(uncertain, list) and uncertain
            and all(isinstance(item, dict) and set(item) == {'field', 'reason'}
                    and all(isinstance(item[key], str) and item[key].strip()
                            for key in ('field', 'reason')) for item in uncertain)):
        result['uncertain_fields'] = [item['field'] + ': ' + item['reason'] for item in uncertain]
        changes.append({'operation': 'uncertainty_objects_to_strings', 'original_fields': uncertain})
    if (not isinstance(headers, list) or not headers
            or any(not isinstance(h, str) or not h.strip() for h in headers)
            or len(headers) != len(set(headers))):
        return result, changes
    rows = result.get('table_rows')
    if not isinstance(rows, list):
        return result, changes
    normalized = []
    for index, row in enumerate(rows):
        if isinstance(row, dict) and isinstance(row.get('cells'), list):
            cells = row['cells']
            if len(cells) == len(headers) and all(isinstance(cell, str) for cell in cells):
                row = {**row, 'cells': dict(zip(headers, cells, strict=True))}
                changes.append({'row_index': index, 'operation': 'positional_cells_to_named_cells'})
        normalized.append(row)
    result['table_rows'] = normalized
    return result, changes


class QwenVisionEvidenceAdapter:
    parser_version = 'paperpilot-qwen-vision-v7-retained-contract'
    """Send only a selected local image region to Qwen through DashScope.

    The adapter never logs the data URL or API key.  A result becomes
    ``verified`` only when the response is complete, schema-valid, readable,
    relevant, sufficiently confident and free of declared ambiguous fields.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-vl-flash-2025-10-15",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: float = 90.0,
        max_output_tokens: int = 4_000,
        max_image_bytes: int = 10 * 1024 * 1024,
        min_image_pixels: int = 262_144,
        min_confidence: float = 0.72,
        client: Any = None,
        fallback_models: tuple[str, ...] = (),
        table_prompt_profile: str = 'retained-v7',
    ) -> None:
        if not api_key.strip():
            raise ConfigurationError("DashScope API key is empty")
        self.model = model.strip() or "qwen3-vl-flash-2025-10-15"
        profiles = {'retained-v7': 'paperpilot-qwen-vision-v7-retained-contract',
                    'compact-v1': 'paperpilot-qwen-vision-v8-compact-table-contract',
                    'compact-system-v1': 'paperpilot-qwen-vision-v9-system-table-contract'}
        if table_prompt_profile not in profiles:
            raise ConfigurationError('Unknown table prompt profile')
        self.table_prompt_profile = table_prompt_profile
        self.parser_version = profiles[table_prompt_profile]
        self.base_url = base_url.rstrip("/")
        self.fallback_models = fallback_models
        self.request_executor = None
        self.durable_dispatch = False
        self.max_output_tokens = max(256, min(int(max_output_tokens), 8_000))
        self.max_image_bytes = max(1, min(int(max_image_bytes), 20 * 1024 * 1024))
        self.min_image_pixels = max(65_536, min(int(min_image_pixels), 4_194_304))
        self.min_confidence = max(0.0, min(float(min_confidence), 1.0))
        self.client = client or OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    @classmethod
    def from_env(cls, *, api_key_file: str | Path | None = None) -> QwenVisionEvidenceAdapter:
        from .model_policy import DAILY_MODEL
        load_dotenv(override=False)
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        configured_file = api_key_file or os.getenv("PAPERPILOT_QWEN_API_KEY_FILE")
        if not key and configured_file:
            key = read_dashscope_api_key_file(Path(configured_file))
        if not key:
            raise ConfigurationError(
                "Qwen vision key not found. Set DASHSCOPE_API_KEY or "
                "PAPERPILOT_QWEN_API_KEY_FILE."
            )
        return cls(
            api_key=key,
            model=os.getenv(
                "PAPERPILOT_QWEN_VISION_MODEL",
                os.getenv("PAPERPILOT_LLM_MODEL", DAILY_MODEL),
            ),
            base_url=os.getenv(
                "PAPERPILOT_QWEN_BASE_URL",
                os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            ),
            timeout_seconds=float(os.getenv("PAPERPILOT_QWEN_TIMEOUT_SECONDS", "90")),
            max_output_tokens=int(os.getenv("PAPERPILOT_QWEN_MAX_OUTPUT_TOKENS", "4000")),
            max_image_bytes=int(
                os.getenv("PAPERPILOT_QWEN_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))
            ),
            min_image_pixels=int(
                os.getenv("PAPERPILOT_QWEN_MIN_IMAGE_PIXELS", "262144")
            ),
            min_confidence=float(os.getenv("PAPERPILOT_QWEN_MIN_CONFIDENCE", "0.72")),
            table_prompt_profile=os.getenv('PAPERPILOT_QWEN_TABLE_PROMPT_PROFILE', 'retained-v7'),
            fallback_models=tuple(name.strip() for name in os.getenv('PAPERPILOT_LLM_FALLBACK_MODELS',
                'qwen3-vl-32b-instruct' if os.getenv('PAPERPILOT_QWEN_VISION_MODEL',
                    os.getenv('PAPERPILOT_LLM_MODEL', DAILY_MODEL)) ==
                    'qwen3-vl-30b-a3b-instruct' else '').split(',') if name.strip()),
        )

    def for_run(self, db, run_id):
        from .model_policy import FREE_MODELS
        if self.model not in FREE_MODELS:
            return self
        if (any(name not in FREE_MODELS or name == self.model for name in self.fallback_models)
                or len(self.fallback_models) > 1 or
                self.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1'):
            raise ValueError('Vision fallback requires the confirmed Beijing Qwen model pair')
        from .vision_dispatch import VisionDispatch
        adapter = copy.copy(self)
        adapter.request_executor = VisionDispatch(adapter, db, run_id)
        adapter.durable_dispatch = True
        return adapter

    def analyze(
        self,
        image_path: str,
        prompt: str,
        metadata: Mapping[str, Any] | None = None,
        *, before_attempt=None, on_dispatch=None,
    ) -> Mapping[str, Any]:
        source = Path(image_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Vision image does not exist: {source}")
        mime_type = _MIME_TYPES.get(source.suffix.casefold())
        if mime_type is None:
            raise ValueError(f"Unsupported vision image type: {source.suffix}")
        image_bytes = source.read_bytes()
        if not image_bytes:
            raise ValueError("Vision image is empty")
        if len(image_bytes) > self.max_image_bytes:
            raise ValueError(
                f"Vision image exceeds configured limit ({len(image_bytes)} > "
                f"{self.max_image_bytes} bytes)"
            )

        context = dict(metadata or {})
        object_type = str(context.get("object_type") or "scholarly region")
        transmitted_bytes = image_bytes
        row_view = {"status": "not_applicable"}
        if object_type == 'table':
            from .table_views import row_focus_view
            _, row_view = row_focus_view(source)
        data_url = f"data:{mime_type};base64,{base64.b64encode(transmitted_bytes).decode('ascii')}"
        instruction = (
            "You are extracting auditable evidence from one cropped scholarly-paper image. "
            "Use only pixels that are actually visible. Never infer missing cells, hidden labels, "
            "uncropped legends or unreadable digits. Return one JSON object with exactly these keys: "
            "description (string), transcription (string), visible_values (string array), "
            "table_rows (array of objects with row_label, checked_columns and cells), "
            "column_headers (string array of ALL leaf column headers in visible left-to-right order), "
            "indicator_columns (string array of ALL checkmark-type leaf headers, including columns "
            "where every data cell is empty), "
            "uncertain_fields (string array), parser_notes (string array), image_readable (boolean), evidence_relevant (boolean), "
            "Put unreadable digits, uncertain header/cell associations, spans, symbols or conditions in "
            "uncertain_fields: EVERY entry blocks use. Put only non-ambiguous layout explanations "
            "or absent background definitions in parser_notes. Notes are NOT source facts and cannot "
            "resolve uncertainties. Do not move an ambiguous field into notes to avoid blocking. "
            "confidence (number 0..1). Preserve visible numeric precision and row/column associations. "
            "For a table, emit every visible data row in table_rows; use checked_columns to name only "
            "indicator columns with checkmarks. cells MUST be a JSON OBJECT mapping exact leaf "
            "headers to strings, never an array. Each cells value MUST be a JSON string, never a "
            "boolean, number or null; use an empty string for an empty cell. "
            "Map every visible cell, including empty indicator cells, under its exact leaf header. "
            "column_headers must match those per-row cells keys exactly; do not list an extra "
            "group heading that has no data column. "
            "Exclude spanning group headings from column_headers; they are not extra columns. "
            "Do not emit captions, headers, blank spacer lines or separators as data rows. "
            "Read the indicator cells independently for each data row; do not accumulate marks "
            "from earlier rows or move a mark to the following numeric row. "
            "A checkmark can be vertically centered inside a merged cell that spans several adjacent "
            "numeric rows. In that case it applies to every row in that visually bounded group (until "
            "the next horizontal separator), so repeat that indicator column in checked_columns for "
            "each affected row; do not attach it only to the row nearest the glyph. "
            "Missing horizontal separators alone do NOT establish merged cells. When marks align "
            "with individual numeric rows, read each row independently: a blank indicator cell "
            "stays blank even when the preceding row had a mark in that column. Never propagate "
            "marks down to later rows unless the image explicitly shows a merged-cell group. "
            "Before returning, re-read the indicator cells row by row and ensure checked_columns "
            "equals the marked indicator cells in that SAME row's cells object. "
            "Copy row labels literally without adding interpretations in parentheses. Re-read every "
            "numeric cell before returning; if any digit or decimal remains ambiguous, name that exact "
            "cell in uncertain_fields instead of guessing. "
            "For non-table regions return an empty table_rows array. "
            'Shape example ONLY (invented labels/values, do not copy them): '
            '{"description":"Example table","transcription":"",'
            '"visible_values":["0.00"],"column_headers":["Flag A","Score"],'
            '"indicator_columns":["Flag A"],"table_rows":[{"row_label":"",'
            '"checked_columns":["Flag A"],"cells":{"Flag A":"✓","Score":"0.00"}}],'
            '"uncertain_fields":[],"parser_notes":[],"image_readable":true,"evidence_relevant":true,'
            '"confidence":0.0}. Replace ALL example content with the visible image evidence. '
            f"Region type: {object_type}. Research request: {prompt}"
        )
        subcalls = []
        attempted = 0
        def usage_totals():
            return {key: sum(row[key] for row in subcalls)
                    for key in ('prompt_tokens', 'completion_tokens')}

        def request(stage, text, max_tokens):
            nonlocal attempted
            def dispatched():
                nonlocal attempted
                attempted += 1
                if on_dispatch:
                    on_dispatch()
            try:
                kwargs = dict(
                    model=self.model, messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': text}, {'type': 'image_url',
                         'image_url': {'url': data_url}, 'min_pixels': self.min_image_pixels,
                         'max_pixels': 4_194_304}]}], temperature=0, max_tokens=max_tokens,
                    response_format={'type': 'json_object'}, extra_body={'enable_thinking': False})
                if object_type == 'table' and stage in {'rows','region'} and self.table_prompt_profile == 'compact-system-v1':
                    from .table_transcription_prompt import system_table_messages
                    kwargs['messages'] = system_table_messages(text, kwargs['messages'][0]['content'][1])
                if self.request_executor is not None:
                    result = self.request_executor.send(kwargs, stage=stage,
                        before_attempt=before_attempt, on_dispatch=dispatched)
                else:
                    if before_attempt is not None:
                        before_attempt()
                    dispatched()
                    result = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                exc.vision_usage = {**usage_totals(), 'model': self.model,
                                    'provider_attempts': attempted}
                raise
            usage = getattr(result, 'usage', None)
            subcalls.append({'stage': stage,
                             'prompt_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0),
                             'completion_tokens': int(getattr(usage, 'completion_tokens', 0) or 0)})
            return result

        header_read = None
        if row_view.get('status') == 'source_row_crops':
            from .table_header_contract import header_candidates
            header_response = request('headers',
                'Read ONLY the column headers from this table image. Return JSON with '
                'column_headers (ALL LEAF column headers in physical left-to-right order), '
                'indicator_columns (leaf headers of checkmark columns), group_headers '
                '(spanning parent headings ONLY), uncertain_fields, parser_notes. Only non-ambiguous '
                'layout explanations belong in parser_notes; every uncertainty blocks use. Each column_headers '
                'item MUST be a leaf; a spanning heading belongs only in group_headers, '
                'never column_headers. Carefully read the lower header tier: its separate '
                'labels replace its parent heading. Metric headers centered vertically in '
                'the upper tier remain individual leaves. Do not transcribe data rows or '
                'numerical results. group_headers must not repeat ANY column_headers entry. '
                'A parent groups multiple leaf columns; a vertically centered single metric '
                'label is a leaf, not a parent. uncertain_fields MUST be an array of strings, '
                'never field/reason objects.', 600)
            try:
                if (not header_response.choices
                        or header_response.choices[0].finish_reason != 'stop'):
                    raise ValueError('Incomplete header-only extraction')
                raw_headers = _parse_json_object(_message_text(header_response.choices[0].message.content))
                candidates = header_candidates(raw_headers)
                header_read = {'raw': raw_headers, 'candidates': candidates}
            except Exception as exc:
                failure = VisionExtractionError('Invalid or uncertain table header read')
                failure.vision_usage = {**usage_totals(), 'model': self.model,
                                        'provider_attempts': attempted}
                raise failure from exc
        if object_type == 'table':
            from .table_transcription_prompt import table_transcription_prompt, retained_table_hint
            if self.table_prompt_profile == 'retained-v7':
                if header_read:
                    instruction += retained_table_hint(prompt, header_read['candidates'])
            else:
                instruction = table_transcription_prompt(prompt, header_read['candidates'] if header_read else None)
        completion = request('rows' if header_read else 'region', instruction, self.max_output_tokens)
        finish_reason = 'unknown'
        try:
            if not completion.choices:
                raise ValueError("Qwen vision returned no choices")
            choice = completion.choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or "unknown")
            text = _message_text(getattr(choice.message, "content", ""))
            raw_provider_extraction = _parse_json_object(text)
            normalized, schema_normalizations = _normalize_named_cells(raw_provider_extraction)
            raw_extraction = normalized
            if header_read:
                from .table_header_contract import normalize_header_tiers
                normalized, header_changes = normalize_header_tiers(normalized, header_read['candidates'])
                schema_normalizations.extend(header_changes)
            extraction = QwenVisualExtraction.model_validate(normalized)
        except Exception as exc:
            # A paid response can fail parsing. Preserve its returned usage so
            # failed extraction is not silently accounted as zero tokens.
            failure = VisionExtractionError('Qwen vision output violates the extraction schema')
            # Pydantic's default error text includes rejected input values.
            # Persist only stable field paths and machine codes at this boundary.
            allowed_fields = set(QwenVisualExtraction.model_fields) | set(QwenTableRow.model_fields)
            failure.vision_schema_issues = [
                {'path': [part if type(part) is int or part in allowed_fields else '[key]'
                          for part in issue['loc']], 'code': issue['type']}
                for issue in (exc.errors(include_input=False, include_url=False)[:20]
                              if isinstance(exc, ValidationError) else [])
            ]
            failure.vision_usage = {
                **usage_totals(), 'provider_attempts': attempted,
                'finish_reason': finish_reason,
                'model': str(getattr(completion, 'model', '') or self.model),
            }
            raise failure from exc

        evidence_has_detail = bool(
            extraction.transcription or extraction.visible_values or extraction.table_rows
        )
        if object_type not in {"table", "equation"}:
            evidence_has_detail = bool(extraction.description)
        table_structure_valid = True
        table_structure_reason = "not_applicable"
        audited_table_rows = extraction.table_rows
        indicator_pixel_audit: dict[str, Any] = {"status": "not_applicable"}
        if object_type == "table":
            table_structure_valid, table_structure_reason = _validate_table_structure(extraction)
            if table_structure_valid:
                (
                    table_structure_valid,
                    indicator_reason,
                    audited_table_rows,
                    indicator_pixel_audit,
                ) = _audit_indicator_pixels(source, extraction.table_rows,
                    column_headers=extraction.column_headers,
                    indicator_columns=extraction.indicator_columns)
                table_structure_reason = f"{table_structure_reason};{indicator_reason}"
        relevance_valid, relevance_mode = _audited_relevance(
            prompt,
            extraction,
            context,
        )
        blocking_uncertain_fields = list(extraction.uncertain_fields)
        verified = bool(
            finish_reason not in {"length", "content_filter"}
            and extraction.image_readable
            and relevance_valid
            and evidence_has_detail
            and table_structure_valid
            and extraction.confidence >= self.min_confidence
            and not blocking_uncertain_fields
        )
        description = extraction.description.strip()
        if object_type == "table" and indicator_pixel_audit.get("status") in {
            "matched",
            "corrected",
        }:
            description = _audited_table_description(
                extraction,
                audited_table_rows,
                context,
            )
        elif extraction.transcription.strip() and extraction.transcription.strip() not in description:
            description = f"{description}\nVisible transcription: {extraction.transcription.strip()}"
        metadata_out = {
            "provider": "dashscope",
            "parser_version": self.parser_version,
            "table_prompt_profile": self.table_prompt_profile,
            "uncertainty_policy": "explicit-blocking-v1",
            "parser_notes": extraction.parser_notes,
            "parser_notes_role": "NOT_SOURCE_EVIDENCE",
            "raw_provider_extraction": raw_provider_extraction,
            "model": str(getattr(completion, "model", "") or self.model),
            "finish_reason": finish_reason,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_bytes": len(image_bytes),
            "transmitted_image_sha256": hashlib.sha256(transmitted_bytes).hexdigest(),
            "transmitted_image_bytes": len(transmitted_bytes),
            "row_focus_view": row_view,
            "row_focus_view_role": "independent_geometry_only; original_crop_transmitted",
            "min_image_pixels": self.min_image_pixels,
            "object_type": object_type,
            "schema": "paperpilot-qwen-vision-v4" if header_read else "paperpilot-qwen-vision-v3",
            "header_read": header_read,
            "raw_table_extraction": raw_extraction if header_read else None,
            "provider_attempts": attempted,
            "subcalls": subcalls,
            "schema_normalizations": schema_normalizations,
            "image_readable": extraction.image_readable,
            "evidence_relevant": relevance_valid,
            "model_evidence_relevant": extraction.evidence_relevant,
            "relevance_mode": relevance_mode,
            "visible_values": extraction.visible_values,
            "table_rows": [row.model_dump() for row in extraction.table_rows],
            "column_headers": extraction.column_headers,
            "indicator_columns": extraction.indicator_columns,
            "audited_table_rows": [row.model_dump() for row in audited_table_rows],
            "table_column_order": (
                list(extraction.table_rows[0].cells) if extraction.table_rows else []
            ),
            "table_structure_valid": table_structure_valid,
            "table_structure_reason": table_structure_reason,
            "indicator_pixel_audit": indicator_pixel_audit,
            "reported_confidence": extraction.confidence,
            "uncertain_fields": extraction.uncertain_fields,
            "blocking_uncertain_fields": blocking_uncertain_fields,
            **usage_totals(),
            "total_tokens": sum(usage_totals().values()),
        }
        return {
            "description": description,
            "confidence": extraction.confidence if verified else min(extraction.confidence, 0.49),
            "verified": verified,
            "metadata": metadata_out,
        }

    def reaudit(
        self,
        original_result: Mapping[str, Any],
        image_path: str | Path,
        *,
        prompt: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reapply current deterministic gates without another provider call."""

        return reaudit_qwen_result(
            original_result,
            image_path,
            prompt=prompt,
            context=metadata,
            min_confidence=self.min_confidence,
        )


_NUMERIC_VALUE_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?![\w.])")


def _numeric_values(values: list[str]) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for value in values:
        tokens.update(_NUMERIC_VALUE_RE.findall(str(value)))
    return tokens


def _validate_table_structure(extraction: QwenVisualExtraction) -> tuple[bool, str]:
    """Cross-check redundant table representations before trusting a model."""

    if not extraction.table_rows:
        return False, "missing_structured_rows"
    if any(not row.cells for row in extraction.table_rows):
        return False, "row_without_cells"
    if any(not row.row_label.strip() and not row.checked_columns
           and not any(cell.strip() for cell in row.cells.values()) for row in extraction.table_rows):
        return False, 'empty_data_row'
    columns = list(extraction.table_rows[0].cells)
    column_set = set(columns)
    if any(set(row.cells) != column_set for row in extraction.table_rows):
        return False, "inconsistent_row_columns"
    # Checkmark columns are represented separately from numeric cells by the
    # public schema.  Requiring them to be duplicated in ``cells`` rejected the
    # exact structure requested from Qwen and prevented the deterministic pixel
    # audit from checking them.  They may overlap with cells, but need not.
    referenced_indicators = [
        str(column).strip()
        for row in extraction.table_rows
        for column in row.checked_columns
        if str(column).strip()
    ]
    headers = extraction.column_headers
    declared_indicators = extraction.indicator_columns
    if headers and len(headers) != len(set(headers)):
        return False, 'duplicate_column_headers'
    if (len(declared_indicators) != len(set(declared_indicators))
            or (declared_indicators and not set(referenced_indicators).issubset(declared_indicators))):
        return False, 'invalid_indicator_columns'
    if headers and set(headers) != column_set | set(referenced_indicators) | set(declared_indicators):
        return False, 'header_cell_coverage_mismatch'
    if declared_indicators and not headers:
        return False, 'indicator_column_order_unknown'
    known_indicators = set(declared_indicators) | set(_indicator_columns(extraction.table_rows))
    for column in headers or columns:
        if column not in column_set:
            continue
        values = [row.cells[column].strip().casefold() for row in extraction.table_rows]
        if any(value and value not in _CHECK_MARK_VALUES for value in values):
            break
        # An omitted checkmark column must not become "no indicators" merely
        # because the same model also omitted its role. Keep ambiguous blank
        # leading cells unresolved instead of treating them as a verified table.
        if not any(values) and column not in known_indicators:
            return False, 'unclassified_empty_leading_column'
    if len(set(referenced_indicators)) > 12:
        return False, "too_many_indicator_columns"
    if any(len(row.checked_columns) != len(set(row.checked_columns)) for row in extraction.table_rows):
        return False, "duplicate_checked_columns"

    row_values: list[str] = []
    for row in extraction.table_rows:
        row_values.extend(str(value) for value in row.cells.values())
    row_numbers = _numeric_values(row_values)
    visible_numbers = _numeric_values(extraction.visible_values)
    transcription_numbers = _numeric_values([extraction.transcription])

    if visible_numbers and visible_numbers - row_numbers:
        return False, "visible_values_missing_from_rows"
    if transcription_numbers and transcription_numbers != row_numbers:
        return False, "transcription_row_numeric_mismatch"
    return True, "redundant_numeric_views_consistent"


_CHECK_MARK_VALUES = {"✓", "√", "✔", "check", "checked", "yes", "true"}

_TABLE_LABEL_RE = re.compile(r"\btable\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
_QUERY_INTENTS = {
    "ablation": ("ablation", "消融"),
    "comparison": ("comparison", "compare", "对比", "比较"),
    "baseline": ("baseline", "基线"),
    "offset": ("offset", "alignment", "光流", "对齐", "偏移"),
}
_BLOCKING_UNCERTAINTY_RE = re.compile(
    r"unreadable|illegible|unclear|ambiguous|uncertain|truncat(?:ed|ion)|"
    r"cut\s*off|cropp(?:ed|ing)|missing|not\s+visible|digit|decimal|"
    r"header|cell|row|column|check\s*mark|checkbox|"
    r"无法|模糊|歧义|不确定|截断|裁剪掉|缺失|不可见|数字|小数|表头|单元格|行|列|勾选",
    re.IGNORECASE,
)


def _audited_relevance(
    prompt: str,
    extraction: QwenVisualExtraction,
    context: Mapping[str, Any],
) -> tuple[bool, str]:
    """Audit a model relevance flag using visible labels and task intent.

    Qwen sometimes marked the exact ablation table irrelevant even while
    transcribing its title and every requested value.  We only override that
    flag for an exact table-number match or a specific research intent present
    in both the query and visible/caption text; generic token overlap is not
    enough.
    """

    if extraction.evidence_relevant:
        return True, "model"
    query = str(prompt or "").casefold()
    visible = " ".join(
        [
            extraction.description,
            extraction.transcription,
            str(context.get("caption") or ""),
            str(context.get("section_path") or ""),
        ]
    ).casefold()
    query_labels = {match.casefold() for match in _TABLE_LABEL_RE.findall(query)}
    visible_labels = {match.casefold() for match in _TABLE_LABEL_RE.findall(visible)}
    if query_labels and query_labels.intersection(visible_labels):
        return True, "audited_exact_table_label"
    for intent, markers in _QUERY_INTENTS.items():
        if any(marker in query for marker in markers) and any(
            marker in visible for marker in markers
        ):
            return True, f"audited_shared_{intent}_intent"
    return False, "model_rejected_no_specific_local_match"


def _is_blocking_visual_uncertainty(value: str) -> bool:
    """Separate unreadable cells from harmless missing background definitions."""

    return bool(_BLOCKING_UNCERTAINTY_RE.search(str(value or "")))


def _indicator_columns(rows: list[QwenTableRow]) -> list[str]:
    if not rows:
        return []
    referenced_in_model_order = list(
        dict.fromkeys(
            str(column).strip()
            for row in rows
            for column in row.checked_columns
            if str(column).strip()
        )
    )
    referenced = set(referenced_in_model_order)
    # Preserve the table's declared left-to-right column order whenever the
    # structured cells include indicator columns. Ordering by the first
    # non-empty checkmark is unsafe: in FlowMamba Table VII the first MFA mark
    # spans several rows, while CDA first appears as a conventional single-row
    # mark; Qwen can therefore mention CDA first and swap the physical columns.
    candidates: list[str] = []
    for column in rows[0].cells:
        values = [str(row.cells.get(column, "")).strip().casefold() for row in rows]
        if all(not value or value in _CHECK_MARK_VALUES for value in values) and (
            any(value in _CHECK_MARK_VALUES for value in values) or column in referenced
        ) and column not in candidates:
            candidates.append(column)
    for column in referenced_in_model_order:
        if column not in candidates:
            candidates.append(column)
    return candidates


def _audit_indicator_pixels(
    image_path: Path,
    rows: list[QwenTableRow],
    *, column_headers: list[str] | None = None,
    indicator_columns: list[str] | None = None,
) -> tuple[bool, str, list[QwenTableRow], dict[str, Any]]:
    """Audit checkmark-style cells with deterministic image projections."""

    indicators = list(dict.fromkeys([*(indicator_columns or []), *_indicator_columns(rows)]))
    if not indicators:
        return True, "not_applicable", rows, {"status": "not_applicable"}
    declared = list(column_headers or [])
    # Dict key order may have been alphabetically sorted by persistence. Neither
    # cells order nor first-mentioned checkmarks establish physical header order.
    if (not declared or len(declared) != len(set(declared))
            or not set(indicators).issubset(declared)):
        return False, 'indicator_column_order_unknown', rows, {'status': 'ambiguous'}
    indicators = [column for column in declared if column in indicators]
    # The projection algorithm only supports leading checkmark columns. Never
    # rearrange a score/label column behind an indicator and then audit the
    # wrong physical cells as if their geometry were correct.
    if declared[:len(indicators)] != indicators:
        return False, 'unsupported_indicator_column_layout', rows, {'status': 'ambiguous'}
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False, "indicator_pixel_audit_unavailable", rows, {"status": "unavailable"}

    pixels = np.asarray(Image.open(image_path).convert("L"))
    dark = pixels < 160
    _, width = dark.shape
    from .table_views import row_focus_view
    _, independent_view = row_focus_view(image_path)
    independently_located_top = None
    if independent_view.get('status') == 'source_row_crops':
        if independent_view['numeric_row_count'] != len(rows):
            return False, 'numeric_row_count_mismatch', rows, {'status': 'ambiguous'}
        independently_located_top = independent_view['row_bboxes_pixels'][0][1]
    long_rows = np.flatnonzero(dark.sum(axis=1) >= max(40, int(width * 0.45))).tolist()
    groups: list[list[int]] = []
    for y in long_rows:
        if not groups or y - groups[-1][-1] > 8:
            groups.append([y])
        else:
            groups[-1].append(y)
    if len(groups) < 2:
        return False, "table_rules_not_detected", rows, {"status": "ambiguous"}

    data_bottom = min(groups[-1])
    # The crop contains indicator columns before numeric metric columns even
    # when Qwen follows the schema and keeps checkmarks out of ``cells``.
    all_columns = declared
    geometry_candidates: list[dict[str, Any]] = []
    indicator_set = set(indicators)
    for top_group in groups[:-1]:
        data_top = max(top_group)
        # Do not choose a header-inclusive geometry because it happens to agree
        # with a model that collapsed several leaf flags into their group label.
        # Numeric row crops locate the data start independently of model cells.
        if (independently_located_top is not None
                and not 0 <= independently_located_top-data_top <= max(8, width/650*4)):
            continue
        if data_bottom - data_top < len(rows) * 8:
            continue
        rule_pixels = np.flatnonzero(dark[data_top] | dark[data_bottom]).tolist()
        if not rule_pixels:
            continue
        table_left = min(rule_pixels)
        table_right = max(rule_pixels) + 1
        x_edges = np.linspace(table_left, table_right, len(all_columns) + 1)
        table_width = max(1, table_right - table_left)

        def header_indicator_bounds() -> tuple[list[tuple[int, int]], list[float]] | None:
            # The first table columns are often much narrower than metric
            # columns and many papers omit vertical rules. Equal-width bins
            # then put the offset text inside the CDA bin. Recover the first
            # indicator headers plus the following non-indicator header from
            # the lower header band and use their midpoints as boundaries.
            needed = len(indicators) + 1
            for band_height in (24, 32, 42):
                y0 = max(0, data_top - band_height)
                y1 = max(y0 + 1, data_top - 3)
                projection = dark[y0:y1, table_left:table_right].sum(axis=0)
                active = np.flatnonzero(projection > 0).tolist()
                tokens: list[list[int]] = []
                for value in active:
                    if not tokens or value - tokens[-1][-1] > 6:
                        tokens.append([value])
                    else:
                        tokens[-1].append(value)
                tokens = [token for token in tokens if token[-1] - token[0] + 1 >= 3]
                if len(tokens) < needed:
                    continue
                centers = [table_left + (token[0] + token[-1]) / 2 for token in tokens[:needed]]
                # Leading indicator columns must start near the left edge.
                # An empty-indicator data row starts at offset/metric columns;
                # treating those numbers as headers silently shifts all bins.
                if centers[0] > table_left + 1.5 * table_width / len(all_columns):
                    continue
                if centers[-1] > table_left + table_width * 0.68:
                    continue
                bounds: list[tuple[int, int]] = []
                for index in range(len(indicators)):
                    left = (
                        table_left
                        if index == 0
                        else round((centers[index - 1] + centers[index]) / 2)
                    )
                    right = round((centers[index] + centers[index + 1]) / 2)
                    if right - left < 8:
                        break
                    bounds.append((left, right))
                if len(bounds) == len(indicators):
                    return bounds, centers
            return None

        header_bounds = header_indicator_bounds()
        if header_bounds is None:
            indicator_bounds = []
            for column in indicators:
                column_index = all_columns.index(column)
                indicator_bounds.append(
                    (round(x_edges[column_index]), round(x_edges[column_index + 1]))
                )
            indicator_x_mode = "equal_width_fallback"
            indicator_header_centers: list[float] = []
        else:
            indicator_bounds, indicator_header_centers = header_bounds
            indicator_x_mode = "header_projection"

        # Some scholarly tables use one vertically merged checkmark cell for a
        # block of several numeric rows (FlowMamba Table VII is a real example).
        # Treating every numeric row as an independent indicator cell moves that
        # single checkmark onto one arbitrary row and falsely clears the rest.
        full_rule_y: list[int] = []
        for group in groups:
            center = round(sum(group) / len(group))
            if center < data_top - 2 or center > data_bottom + 2:
                continue
            representative = max(
                group,
                key=lambda value: int(
                    dark[min(max(value, 0), dark.shape[0] - 1), table_left:table_right].sum()
                ),
            )
            row = dark[
                min(max(representative, 0), dark.shape[0] - 1),
                table_left:table_right,
            ]
            positions = np.flatnonzero(row).tolist()
            if positions and (max(positions) - min(positions) + 1) >= int(table_width * 0.88):
                full_rule_y.append(center)
        section_bounds = sorted(
            set([data_top, data_bottom, *[value for value in full_rule_y if data_top < value < data_bottom]])
        )

        def allocate_section_rows(bounds: list[int], row_count: int) -> list[int]:
            spans = [max(1, bounds[index + 1] - bounds[index]) for index in range(len(bounds) - 1)]
            if len(spans) <= 1 or len(spans) > row_count:
                return [row_count]
            total = sum(spans)
            raw = [span * row_count / total for span in spans]
            counts = [max(1, int(value)) for value in raw]
            while sum(counts) > row_count:
                candidates = [index for index, value in enumerate(counts) if value > 1]
                if not candidates:
                    return [row_count]
                index = min(candidates, key=lambda item: raw[item] - counts[item])
                counts[index] -= 1
            while sum(counts) < row_count:
                index = max(range(len(counts)), key=lambda item: raw[item] - counts[item])
                counts[index] += 1
            return counts

        section_row_counts = allocate_section_rows(section_bounds, len(rows))
        use_merged_sections = len(section_row_counts) > 1
        if use_merged_sections:
            cursor = 0
            for count in section_row_counts:
                section_rows = rows[cursor : cursor + count]
                # A vision model can correctly recognize a row-spanning glyph
                # on most rows while omitting the first/last row of the group.
                # Accept only a near-unanimous one-row miss in a sufficiently
                # large section; a lone/spurious mark or a two/three-row split
                # remains ambiguous and falls back to row-wise fail-closed
                # auditing.
                votes = {
                    column: sum(column in row.checked_columns for row in section_rows)
                    for column in indicators
                }
                section_consistent = all(value in {0, count} for value in votes.values())
                near_unanimous_single_miss = count >= 4 and all(
                    value in {0, count - 1, count} for value in votes.values()
                )
                if not (section_consistent or near_unanimous_single_miss):
                    use_merged_sections = False
                    break
                cursor += count

        ink_counts: list[dict[str, int]] = []
        detected_by_row: list[list[str]] = []
        ambiguous = False
        audit_bands: list[tuple[int, int, list[int]]] = []
        if use_merged_sections:
            cursor = 0
            for section_index, count in enumerate(section_row_counts):
                audit_bands.append(
                    (
                        section_bounds[section_index],
                        section_bounds[section_index + 1],
                        list(range(cursor, cursor + count)),
                    )
                )
                cursor += count
        else:
            y_edges = np.linspace(data_top, data_bottom, len(rows) + 1)
            audit_bands = [
                (round(y_edges[index]), round(y_edges[index + 1]), [index])
                for index in range(len(rows))
            ]

        row_counts_by_index: dict[int, dict[str, int]] = {}
        detected_by_index: dict[int, list[str]] = {}
        for band_top, band_bottom, row_indexes in audit_bands:
            counts: dict[str, int] = {}
            detected: list[str] = []
            y0 = round(band_top) + 3
            y1 = round(band_bottom) - 3
            for column, (cell_left, cell_right) in zip(
                indicators,
                indicator_bounds,
                strict=True,
            ):
                cell_width = cell_right - cell_left
                margin = max(2, round(cell_width * 0.10))
                x0 = round(cell_left) + margin
                x1 = round(cell_right) - margin
                count = int(dark[y0:y1, x0:x1].sum())
                counts[column] = count
                if 4 <= count < 10:
                    ambiguous = True
                if count >= 10:
                    detected.append(column)
            for row_index in row_indexes:
                row_counts_by_index[row_index] = dict(counts)
                detected_by_index[row_index] = list(detected)
        ink_counts = [row_counts_by_index[index] for index in range(len(rows))]
        detected_by_row = [detected_by_index[index] for index in range(len(rows))]
        if ambiguous:
            continue
        agreement = 0
        for row, detected in zip(rows, detected_by_row, strict=True):
            expected = set(row.checked_columns) & indicator_set
            observed = set(detected)
            agreement += sum(
                (column in expected) == (column in observed)
                for column in indicators
            )
        geometry_candidates.append(
            {
                "agreement": agreement,
                "span": data_bottom - data_top,
                "data_top": data_top,
                "table_left": table_left,
                "table_right": table_right,
                "ink_counts": ink_counts,
                "detected_by_row": detected_by_row,
                "row_partition_mode": "merged_sections" if use_merged_sections else "uniform_rows",
                "section_row_counts": section_row_counts if use_merged_sections else [len(rows)],
                "indicator_x_mode": indicator_x_mode,
                "indicator_header_centers": indicator_header_centers,
            }
        )

    if not geometry_candidates:
        return False, "table_row_geometry_invalid", rows, {"status": "ambiguous"}
    geometry = max(
        geometry_candidates,
        # A partition whose full-width rules align with constant model
        # indicators is stronger evidence than an accidental row-wise pixel
        # match across a vertically merged cell.
        key=lambda item: (
            int(item.get("indicator_x_mode") == "header_projection"),
            (
                int(item.get("row_partition_mode") == "merged_sections")
                if item.get("indicator_x_mode") == "header_projection"
                else int(item["agreement"])
            ),
            (
                int(item["agreement"])
                if item.get("indicator_x_mode") == "header_projection"
                else int(item.get("row_partition_mode") == "merged_sections")
            ),
            int(item["span"]),
        ),
    )
    data_top = int(geometry["data_top"])
    table_left = int(geometry["table_left"])
    table_right = int(geometry["table_right"])
    ink_counts = list(geometry["ink_counts"])
    detected_by_row = list(geometry["detected_by_row"])

    audited_rows: list[QwenTableRow] = []
    corrections: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        audited = row.model_copy(deep=True)
        detected = detected_by_row[row_index]
        model_checked = [column for column in indicators if column in row.checked_columns]
        if model_checked != detected:
            corrections.append(
                {
                    "row_index": row_index,
                    "model_checked_columns": model_checked,
                    "pixel_checked_columns": detected,
                }
            )
        audited.checked_columns = detected
        for column in indicators:
            audited.cells[column] = "✓" if column in detected else ""
        audited_rows.append(audited)

    # Pixel projection is a cross-check, not an oracle.  A large disagreement
    # commonly means row-spanning cells or ambiguous geometry; automatically
    # rewriting many model rows produced a false-positive verification on
    # FlowMamba Table VII.  Fail closed and retain the original transcription.
    max_safe_corrections = max(1, len(rows) // 4)
    label_permutation: dict[str, str] | None = None
    label_permutation_score: int | None = None
    if (
        len(corrections) > max_safe_corrections
        and 2 <= len(indicators) <= 4
        and geometry.get("indicator_x_mode") == "header_projection"
    ):
        total = len(rows) * len(indicators)
        best_score = -1
        best_mappings: list[dict[str, str]] = []
        for permutation in itertools.permutations(indicators):
            mapping = dict(zip(indicators, permutation, strict=True))
            if all(mapping[column] == column for column in indicators):
                continue
            score = 0
            for row, detected in zip(rows, detected_by_row, strict=True):
                model = set(row.checked_columns)
                observed = set(detected)
                score += sum(
                    (mapping[column] in model) == (column in observed)
                    for column in indicators
                )
            if score > best_score:
                best_score = score
                best_mappings = [mapping]
            elif score == best_score:
                best_mappings.append(mapping)
        # Permit one residual cell mistake after a globally consistent header
        # permutation. This matches the observed Qwen output where the two
        # indicator names were swapped and one conventional row carried one
        # extra mark; pixels remain the final authority for every corrected
        # cell.
        if best_score >= total - 1 and len(best_mappings) == 1:
            label_permutation = best_mappings[0]
            label_permutation_score = best_score
    if len(corrections) > max_safe_corrections and label_permutation is None:
        return (
            False,
            "indicator_pixel_model_conflict",
            rows,
            {
                "status": "conflict",
                "indicator_columns": indicators,
                "ink_counts": ink_counts,
                "detected_checked_columns": detected_by_row,
                "corrections": corrections,
                "max_safe_corrections": max_safe_corrections,
                "table_bbox_pixels": [table_left, data_top, table_right, data_bottom],
                "geometry_candidate_count": len(geometry_candidates),
                "geometry_indicator_agreement": int(geometry["agreement"]),
                "row_partition_mode": geometry.get("row_partition_mode"),
                "section_row_counts": geometry.get("section_row_counts"),
                "indicator_x_mode": geometry.get("indicator_x_mode"),
                "indicator_header_centers": geometry.get("indicator_header_centers"),
                "label_permutation_detected": None,
                "label_permutation_score": label_permutation_score,
            },
        )

    status = "corrected" if corrections else "matched"
    return (
        True,
        f"indicator_pixels_{status}",
        audited_rows,
        {
            "status": status,
            "indicator_columns": indicators,
            "ink_counts": ink_counts,
            "detected_checked_columns": detected_by_row,
            "corrections": corrections,
            "table_bbox_pixels": [table_left, data_top, table_right, data_bottom],
            "geometry_candidate_count": len(geometry_candidates),
            "geometry_indicator_agreement": int(geometry["agreement"]),
            "row_partition_mode": geometry.get("row_partition_mode"),
            "section_row_counts": geometry.get("section_row_counts"),
            "indicator_x_mode": geometry.get("indicator_x_mode"),
            "indicator_header_centers": geometry.get("indicator_header_centers"),
            "label_permutation_detected": label_permutation,
            "label_permutation_score": label_permutation_score,
        },
    )


def _render_table_rows(rows: list[QwenTableRow]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].cells)
    lines = ["row_label | " + " | ".join(columns)]
    for row in rows:
        values = [row.row_label, *(str(row.cells.get(column, "")) for column in columns)]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def _description_column_order(
    description: str,
    available: set[str],
) -> list[str]:
    """Recover the provider's visible table order after JSON storage sorted keys."""

    for line in str(description or "").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if not parts or parts[0].casefold() != "row_label":
            continue
        ordered = [part for part in parts[1:] if part in available]
        if ordered:
            return ordered
    return []


def _restore_table_column_order(
    rows: list[QwenTableRow],
    metadata: Mapping[str, Any],
    description: str,
) -> list[QwenTableRow]:
    if not rows:
        return rows
    available = set(rows[0].cells)
    requested = [
        str(column)
        for column in list(metadata.get("table_column_order") or [])
        if str(column) in available
    ]
    if not requested:
        requested = _description_column_order(description, available)
    if not requested:
        return rows
    order = [*requested, *(column for column in rows[0].cells if column not in requested)]
    restored: list[QwenTableRow] = []
    for row in rows:
        copy_row = row.model_copy(deep=True)
        copy_row.cells = {column: str(row.cells.get(column, "")) for column in order}
        restored.append(copy_row)
    return restored


def _audited_table_description(
    extraction: QwenVisualExtraction,
    rows: list[QwenTableRow],
    context: Mapping[str, Any],
) -> str:
    """Publish audited cells, not unconstrained model prose about acronyms."""

    visible = " ".join(
        [
            str(context.get("caption") or ""),
            extraction.transcription,
            extraction.description,
        ]
    )
    match = re.search(r"\btable\s+(?:[ivxlcdm]+|\d+)\b", visible, re.IGNORECASE)
    heading = match.group(0).upper() if match else "AUDITED SCHOLARLY TABLE"
    return f"{heading}\nAudited structured transcription:\n{_render_table_rows(rows)}"


def reaudit_qwen_result(
    original_result: Mapping[str, Any],
    image_path: str | Path,
    *,
    prompt: str,
    context: Mapping[str, Any] | None = None,
    min_confidence: float = 0.72,
) -> dict[str, Any]:
    """Re-evaluate a persisted Qwen table response under the current gates.

    This function performs no network access. It binds the persisted response to
    the exact crop SHA before rebuilding the strict schema, relevance,
    uncertainty and deterministic indicator-pixel checks. It is used by paid
    evidence replay so old adapter bugs can be fixed without changing the
    original provider response or spending another request.
    """

    source = Path(image_path).expanduser().resolve()
    image_bytes = source.read_bytes()
    metadata = dict(original_result.get("metadata") or {})
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    if image_sha != str(metadata.get("image_sha256") or ""):
        raise ValueError("Image SHA does not match the persisted Qwen result")

    description = str(original_result.get("description") or "").strip()
    transcription = ""
    summary = description
    for marker in ("\nVisible transcription: ", "\nAudited structured transcription:\n"):
        if marker in description:
            summary, transcription = description.split(marker, 1)
            break
    extraction = QwenVisualExtraction(
        description=summary or "Persisted scholarly table extraction",
        transcription=transcription,
        visible_values=metadata.get("visible_values") or [],
        table_rows=metadata.get("table_rows") or [],
        column_headers=metadata.get("column_headers") or [],
        indicator_columns=metadata.get("indicator_columns") or [],
        uncertain_fields=metadata.get("uncertain_fields") or [],
        parser_notes=metadata.get("parser_notes") or [],
        image_readable=bool(metadata.get("image_readable")),
        evidence_relevant=bool(
            metadata.get("model_evidence_relevant", metadata.get("evidence_relevant"))
        ),
        confidence=float(
            metadata.get("reported_confidence")
            or original_result.get("confidence")
            or 0.0
        ),
    )
    extraction.table_rows = _restore_table_column_order(
        extraction.table_rows,
        metadata,
        description,
    )
    structure_valid, structure_reason = _validate_table_structure(extraction)
    audited_rows = extraction.table_rows
    pixel_details: dict[str, Any] = {"status": "not_run"}
    if structure_valid:
        structure_valid, pixel_reason, audited_rows, pixel_details = _audit_indicator_pixels(
            source,
            extraction.table_rows,
            column_headers=extraction.column_headers,
            indicator_columns=extraction.indicator_columns,
        )
        structure_reason = f"{structure_reason};{pixel_reason}"
    relevance_valid, relevance_mode = _audited_relevance(
        prompt,
        extraction,
        dict(context or {}),
    )
    blocking_uncertain_fields = [
        value
        for value in extraction.uncertain_fields
        if metadata.get('uncertainty_policy') == 'explicit-blocking-v1' or _is_blocking_visual_uncertainty(value)
    ]
    finish_reason = str(metadata.get("finish_reason") or "unknown")
    evidence_has_detail = bool(
        extraction.transcription or extraction.visible_values or extraction.table_rows
    )
    verified = bool(
        finish_reason not in {"length", "content_filter"}
        and extraction.image_readable
        and relevance_valid
        and evidence_has_detail
        and structure_valid
        and extraction.confidence >= float(min_confidence)
        and not blocking_uncertain_fields
    )
    audited_description = summary.strip()
    if audited_rows:
        audited_description = _audited_table_description(
            extraction,
            audited_rows,
            dict(context or {}),
        )
    metadata.update(
        {
            "evidence_relevant": relevance_valid,
            "model_evidence_relevant": extraction.evidence_relevant,
            "relevance_mode": relevance_mode,
            "audited_table_rows": [row.model_dump() for row in audited_rows],
            "table_column_order": (
                list(extraction.table_rows[0].cells) if extraction.table_rows else []
            ),
            "table_structure_valid": structure_valid,
            "table_structure_reason": structure_reason,
            "indicator_pixel_audit": pixel_details,
            "blocking_uncertain_fields": blocking_uncertain_fields,
            "offline_reaudit": True,
            "offline_reaudit_policy": "paperpilot-qwen-current-gates-v2",
        }
    )
    return {
        "description": audited_description,
        "confidence": extraction.confidence if verified else min(extraction.confidence, 0.49),
        "verified": verified,
        "metadata": metadata,
    }


def _parse_json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Qwen vision returned invalid JSON") from None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen vision returned invalid JSON") from exc
    # Some OpenAI-compatible multimodal gateways honor JSON mode but wrap the
    # single requested object in a one-element array (or a JSON-encoded string).
    # Unwrap only an unambiguous single object; multiple candidates remain a
    # contract error rather than silently choosing one.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        value = value[0]
    if not isinstance(value, Mapping):
        raise TypeError("Qwen vision response must be a JSON object")
    return value


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")
