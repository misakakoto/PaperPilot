from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from paperpilot_agentic.adapters import make_vision_adapter
from paperpilot_agentic.qwen_vision import (
    QwenTableRow,
    QwenVisualExtraction,
    QwenVisionEvidenceAdapter,
    _audit_indicator_pixels,
    _indicator_columns,
    _parse_json_object,
    _validate_table_structure,
    reaudit_qwen_result,
)


class _FakeCompletions:
    def __init__(self, payload: dict, *, finish_reason: str = "stop") -> None:
        self.payload = payload
        self.finish_reason = finish_reason
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model="qwen3-vl-flash-test",
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=json.dumps(self.payload)),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165),
        )


class _FakeClient:
    def __init__(self, payload: dict, *, finish_reason: str = "stop") -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(payload, finish_reason=finish_reason)
        )


def _payload(**overrides):
    value = {
        "description": "Table XII reports 81.2 for the baseline.",
        "transcription": "Model | F1\nBaseline | 81.2",
        "visible_values": ["81.2"],
        "table_rows": [
            {
                "row_label": "Baseline",
                "checked_columns": [],
                "cells": {"Score": "81.2"},
            }
        ],
        "uncertain_fields": [],
        "image_readable": True,
        "evidence_relevant": True,
        "confidence": 0.91,
    }
    value.update(overrides)
    return value


def test_qwen_json_parser_accepts_unambiguous_single_object_wrappers() -> None:
    expected = _payload()
    assert _parse_json_object(json.dumps([expected])) == expected
    assert _parse_json_object(json.dumps(json.dumps(expected))) == expected
    with pytest.raises(TypeError, match="JSON object"):
        _parse_json_object(json.dumps([expected, expected]))


def test_uncertainty_object_keeps_its_reason_and_cannot_turn_into_verified_fact(tmp_path):
    image = tmp_path/'table.png'
    image.write_bytes(b'fixture')
    payload = _payload(uncertain_fields=[{'field': 'Score', 'reason': 'second decimal digit unreadable'}])
    client = _FakeClient(payload)
    result = QwenVisionEvidenceAdapter(api_key='offline', client=client).analyze(
        str(image), 'Read the table.', {'object_type': 'table'})
    assert result['verified'] is False
    assert result['metadata']['uncertain_fields'] == ['Score: second decimal digit unreadable']
    assert result['metadata']['schema_normalizations'][0]['original_fields'] == payload['uncertain_fields']
    assert result['metadata']['prompt_tokens'] == 120


def test_qwen_adapter_sends_local_crop_and_returns_audited_metadata(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"\x89PNG\r\nfixture")
    client = _FakeClient(_payload())
    adapter = QwenVisionEvidenceAdapter(api_key="sk-test", client=client)

    result = adapter.analyze(
        str(image),
        "Read the ablation value.",
        {"object_type": "table", "page": 16},
    )

    assert result["verified"] is True
    assert result["metadata"]["image_sha256"]
    assert result["metadata"]["prompt_tokens"] == 120
    assert result["metadata"]["schema"] == "paperpilot-qwen-vision-v3"
    assert result["metadata"]["table_structure_valid"] is True
    assert result["metadata"]["reported_confidence"] == 0.91
    request = client.chat.completions.kwargs
    assert request is not None
    image_part = request["messages"][0]["content"][1]
    instruction = request["messages"][0]["content"][0]["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part["min_pixels"] == 262_144
    assert "merged cell that spans several adjacent numeric rows" in instruction
    assert result["metadata"]["min_image_pixels"] == 262_144
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {"enable_thinking": False}
    assert request["max_tokens"] == 4_000


def test_invalid_visual_schema_preserves_paid_response_usage(tmp_path: Path) -> None:
    image = tmp_path / 'table.png'
    image.write_bytes(b'\x89PNG\r\nfixture')
    client = _FakeClient(_payload(table_rows=[{
        'row_label': 'row', 'checked_columns': [], 'cells': {'indicator': False}}]))
    adapter = QwenVisionEvidenceAdapter(api_key='sk-test', client=client)
    with pytest.raises(ValueError) as failure:
        adapter.analyze(str(image), 'Transcribe this table', {'object_type': 'table'})
    assert failure.value.vision_usage['prompt_tokens'] == 120
    assert failure.value.vision_usage['completion_tokens'] == 45


def test_qwen_adapter_does_not_verify_ambiguous_or_truncated_output(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"\x89PNG\r\nfixture")
    client = _FakeClient(
        _payload(uncertain_fields=["second decimal digit"]),
        finish_reason="length",
    )
    adapter = QwenVisionEvidenceAdapter(api_key="sk-test", client=client)

    result = adapter.analyze(str(image), "Read the value.", {"object_type": "table"})

    assert result["verified"] is False
    assert result["confidence"] <= 0.49
    assert result["metadata"]["uncertain_fields"] == ["second decimal digit"]


def test_qwen_adapter_audits_specific_relevance_and_keeps_context_only_notes(
    tmp_path: Path,
) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"\x89PNG\r\nfixture")
    client = _FakeClient(
        _payload(
            description="TABLE XII. ABLATION STUDY OF THE PROPOSED COMPONENTS.",
            transcription="TABLE XII\nBaseline | 81.2",
            evidence_relevant=False,
            parser_notes=["The meaning of F1 is not defined within the crop."],
        )
    )
    adapter = QwenVisionEvidenceAdapter(api_key="sk-test", client=client)

    result = adapter.analyze(
        str(image),
        "Read the Table XII ablation values.",
        {"object_type": "table", "caption": "TABLE XII ABLATION STUDY"},
    )

    assert result["verified"] is True
    assert result["metadata"]["model_evidence_relevant"] is False
    assert result["metadata"]["evidence_relevant"] is True
    assert result["metadata"]["relevance_mode"] == "audited_exact_table_label"
    assert result["metadata"]["blocking_uncertain_fields"] == []


def test_offline_reaudit_binds_crop_and_reapplies_current_relevance_gates(
    tmp_path: Path,
) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"persisted-crop-fixture")
    original = {
        "description": "TABLE XII reports one ablation row.\nVisible transcription: Baseline | 81.2",
        "confidence": 0.49,
        "verified": False,
        "metadata": {
            "finish_reason": "stop",
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "image_readable": True,
            "evidence_relevant": False,
            "reported_confidence": 0.91,
            "visible_values": ["81.2"],
            "table_rows": [
                {
                    "row_label": "Baseline",
                    "checked_columns": [],
                    "cells": {"Score": "81.2"},
                }
            ],
            "uncertain_fields": ["The metric definition is not provided in this crop."],
        },
    }

    audited = reaudit_qwen_result(
        original,
        image,
        prompt="Read the Table XII ablation values.",
        context={"caption": "TABLE XII ABLATION STUDY"},
    )

    assert audited["verified"] is True
    assert audited["metadata"]["relevance_mode"] == "audited_exact_table_label"
    assert audited["metadata"]["blocking_uncertain_fields"] == []
    assert audited["metadata"]["offline_reaudit"] is True

    other = tmp_path / "other.png"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="Image SHA"):
        reaudit_qwen_result(original, other, prompt="Table XII")


def test_qwen_adapter_rejects_self_confident_incomplete_table_rows(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    image.write_bytes(b"\x89PNG\r\nfixture")
    client = _FakeClient(
        _payload(
            transcription="Only one row | 81.2",
            visible_values=["81.2", "92.16"],
            table_rows=[
                {
                    "row_label": "Only one row",
                    "checked_columns": [],
                    "cells": {"Score": "81.2"},
                }
            ],
            confidence=1.0,
        )
    )
    adapter = QwenVisionEvidenceAdapter(api_key="sk-test", client=client)

    result = adapter.analyze(str(image), "Read every row.", {"object_type": "table"})

    assert result["verified"] is False
    assert result["confidence"] == 0.49
    assert result["metadata"]["table_structure_reason"] == "visible_values_missing_from_rows"


def test_pixel_audit_corrects_a_missed_checkmark(tmp_path: Path) -> None:
    image_path = tmp_path / "ablation.png"
    image = Image.new("L", (350, 200), color=255)
    draw = ImageDraw.Draw(image)
    # Internal section separators reproduce FlowMamba Table VII.  The audit
    # must not mistake the last two separators for the full data region.
    for y in (20, 22, 60, 62, 82, 84, 120, 122, 140, 142, 160, 162):
        draw.line((20, y, 330, y), fill=0, width=1)
    x_edges = [20 + (310 * index / 7) for index in range(8)]
    y_edges = [62 + (98 * index / 5) for index in range(6)]
    expected = [[], [0], [1], [2], [0, 1, 2]]
    for row_index, checked_columns in enumerate(expected):
        cy = (y_edges[row_index] + y_edges[row_index + 1]) / 2
        for column_index in checked_columns:
            cx = (x_edges[column_index] + x_edges[column_index + 1]) / 2
            draw.line((cx - 5, cy, cx - 1, cy + 4), fill=0, width=2)
            draw.line((cx - 1, cy + 4, cx + 6, cy - 5), fill=0, width=2)
    image.save(image_path)

    columns = ["Sequential", "Cross", "Parallel", "OA", "F1", "IoU", "KC"]
    rows: list[QwenTableRow] = []
    for index in range(5):
        checked = [columns[item] for item in expected[index]]
        if index == 4:
            checked = ["Cross", "Parallel"]  # Simulate the real Qwen miss.
        # Follow the remote schema literally: checkmarks live in
        # checked_columns and numeric cells do not duplicate them.
        cells = {"OA": str(90 + index), "F1": str(80 + index), "IoU": "70", "KC": "75"}
        rows.append(
            QwenTableRow(
                row_label=f"Row {index + 1}",
                checked_columns=checked,
                cells=cells,
            )
        )

    extraction = QwenVisualExtraction(
        description="Visible ablation table",
        transcription="",
        visible_values=[],
        table_rows=rows,
        uncertain_fields=[],
        image_readable=True,
        evidence_relevant=True,
        confidence=0.95,
    )
    structure_valid, structure_reason = _validate_table_structure(extraction)
    valid, reason, audited, details = _audit_indicator_pixels(image_path, rows, column_headers=columns)

    assert structure_valid is True
    assert structure_reason == "redundant_numeric_views_consistent"
    assert valid is True
    assert reason == "indicator_pixels_corrected"
    assert audited[-1].checked_columns == ["Sequential", "Cross", "Parallel"]
    assert details["corrections"][0]["row_index"] == 4


def test_pixel_audit_preserves_row_spanning_indicator_cells(tmp_path: Path) -> None:
    image_path = tmp_path / "merged-indicators.png"
    image = Image.new("L", (420, 280), color=255)
    draw = ImageDraw.Draw(image)
    # Header/data top, baseline separator, five-row MFA block separator,
    # one-row CDA separator and final bottom rule.
    for y in (30, 32, 70, 72, 105, 107, 220, 222, 245, 247, 270, 272):
        draw.line((20, y, 400, y), fill=0, width=1)
    x_edges = [20 + (380 * index / 10) for index in range(11)]
    # Real scholarly crops expose the indicator headers immediately above the
    # data rule; this lets the audit reject a larger header-inclusive geometry.
    draw.text((24, 52), "MFA", fill=0)
    draw.text((64, 52), "CDA", fill=0)
    draw.text((104, 52), "offset", fill=0)

    def check(column: int, top: int, bottom: int) -> None:
        cx = (x_edges[column] + x_edges[column + 1]) / 2
        cy = (top + bottom) / 2
        draw.line((cx - 5, cy, cx - 1, cy + 4), fill=0, width=2)
        draw.line((cx - 1, cy + 4, cx + 6, cy - 5), fill=0, width=2)

    check(0, 107, 220)  # One merged MFA cell represents five numeric rows.
    check(1, 222, 245)
    check(0, 247, 270)
    check(1, 247, 270)
    image.save(image_path)

    checked = [[], ["MFA"], ["MFA"], ["MFA"], ["MFA"], ["MFA"], ["CDA"], ["MFA", "CDA"]]
    rows = [
        QwenTableRow(
            row_label=f"Row {index}",
            checked_columns=value,
            cells={f"metric_{column}": str(index + column) for column in range(8)},
        )
        for index, value in enumerate(checked)
    ]

    valid, reason, audited, details = _audit_indicator_pixels(image_path, rows,
        column_headers=['MFA', 'CDA', *rows[0].cells])

    assert valid is True
    assert details["row_partition_mode"] == "merged_sections", details
    assert details["section_row_counts"] == [1, 5, 1, 1], details
    assert reason == "indicator_pixels_matched", details
    assert [row.checked_columns for row in audited] == checked


def test_indicator_columns_follow_declared_table_order_not_first_mark() -> None:
    rows = [
        QwenTableRow(
            row_label="baseline",
            checked_columns=[],
            cells={"MFA": "", "CDA": "", "score": "80.14"},
        ),
        QwenTableRow(
            row_label="first visible mark",
            checked_columns=["CDA"],
            cells={"MFA": "", "CDA": "✓", "score": "80.26"},
        ),
        QwenTableRow(
            row_label="later mark",
            checked_columns=["MFA"],
            cells={"MFA": "✓", "CDA": "", "score": "80.43"},
        ),
    ]

    assert _indicator_columns(rows) == ["MFA", "CDA"]


def test_pixel_audit_repairs_one_missed_row_in_merged_indicator_group(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "merged-indicators-one-miss.png"
    image = Image.new("L", (420, 280), color=255)
    draw = ImageDraw.Draw(image)
    for y in (30, 32, 70, 72, 105, 107, 220, 222, 245, 247, 270, 272):
        draw.line((20, y, 400, y), fill=0, width=1)
    x_edges = [20 + (380 * index / 10) for index in range(11)]
    draw.text((24, 52), "MFA", fill=0)
    draw.text((64, 52), "CDA", fill=0)
    draw.text((104, 52), "offset", fill=0)

    def check(column: int, top: int, bottom: int) -> None:
        cx = (x_edges[column] + x_edges[column + 1]) / 2
        cy = (top + bottom) / 2
        draw.line((cx - 5, cy, cx - 1, cy + 4), fill=0, width=2)
        draw.line((cx - 1, cy + 4, cx + 6, cy - 5), fill=0, width=2)

    check(0, 107, 220)
    check(1, 222, 245)
    check(0, 247, 270)
    check(1, 247, 270)
    image.save(image_path)

    # Qwen read the merged MFA group correctly on four of its five numeric
    # rows, but omitted the first row (the observed FlowMamba failure mode).
    checked = [[], [], ["MFA"], ["MFA"], ["MFA"], ["MFA"], ["CDA"], ["MFA", "CDA"]]
    rows = [
        QwenTableRow(
            row_label=f"Row {index}",
            checked_columns=value,
            cells={f"metric_{column}": str(index + column) for column in range(8)},
        )
        for index, value in enumerate(checked)
    ]

    valid, reason, audited, details = _audit_indicator_pixels(image_path, rows,
        column_headers=['MFA', 'CDA', *rows[0].cells])

    assert valid is True, details
    assert reason == "indicator_pixels_corrected", details
    assert details["row_partition_mode"] == "merged_sections", details
    assert details["section_row_counts"] == [1, 5, 1, 1], details
    assert audited[1].checked_columns == ["MFA"]
    assert [row.checked_columns for row in audited[1:6]] == [["MFA"]] * 5
    assert len(details["corrections"]) == 1


def test_pixel_audit_repairs_indicator_label_swap_with_one_residual_error(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "merged-indicators-swapped-labels.png"
    image = Image.new("L", (420, 280), color=255)
    draw = ImageDraw.Draw(image)
    for y in (30, 32, 70, 72, 105, 107, 220, 222, 245, 247, 270, 272):
        draw.line((20, y, 400, y), fill=0, width=1)
    draw.text((24, 52), "MFA", fill=0)
    draw.text((64, 52), "CDA", fill=0)
    draw.text((104, 52), "offset", fill=0)
    x_edges = [20 + (380 * index / 10) for index in range(11)]

    def check(column: int, top: int, bottom: int) -> None:
        cx = (x_edges[column] + x_edges[column + 1]) / 2
        cy = (top + bottom) / 2
        draw.line((cx - 5, cy, cx - 1, cy + 4), fill=0, width=2)
        draw.line((cx - 1, cy + 4, cx + 6, cy - 5), fill=0, width=2)

    check(0, 107, 220)
    check(1, 222, 245)
    check(0, 247, 270)
    check(1, 247, 270)
    image.save(image_path)

    # MFA/CDA names are globally swapped and the conventional CDA row carries
    # one extra mark, matching the observed provider response.
    swapped = [[], ["CDA"], ["CDA"], ["CDA"], ["CDA"], ["CDA"], ["MFA", "CDA"], ["MFA", "CDA"]]
    rows = [
        QwenTableRow(
            row_label=f"Row {index}",
            checked_columns=value,
            cells={
                "MFA": "✓" if "MFA" in value else "",
                "CDA": "✓" if "CDA" in value else "",
                **{f"metric_{column}": str(index + column) for column in range(8)},
            },
        )
        for index, value in enumerate(swapped)
    ]

    valid, reason, audited, details = _audit_indicator_pixels(image_path, rows,
        column_headers=list(rows[0].cells))

    assert valid is True, details
    assert reason == "indicator_pixels_corrected", details
    assert details["label_permutation_detected"] == {"MFA": "CDA", "CDA": "MFA"}
    assert details["label_permutation_score"] == 15
    assert [row.checked_columns for row in audited] == [
        [],
        ["MFA"],
        ["MFA"],
        ["MFA"],
        ["MFA"],
        ["MFA"],
        ["CDA"],
        ["MFA", "CDA"],
    ]


def test_pixel_audit_fails_closed_on_large_indicator_conflict(tmp_path: Path) -> None:
    image_path = tmp_path / "indicator-conflict.png"
    image = Image.new("L", (420, 280), color=255)
    draw = ImageDraw.Draw(image)
    for y in (30, 32, 70, 72, 270, 272):
        draw.line((20, y, 400, y), fill=0, width=1)
    image.save(image_path)

    # The model claims an indicator in every row, while the crop contains no
    # corresponding ink.  The pixel audit must not rewrite and verify a large
    # conflict as if its geometry were unquestionably correct.
    inconsistent = [["MFA"] for _ in range(8)]
    rows = [
        QwenTableRow(
            row_label=f"Row {index}",
            checked_columns=value,
            cells={f"metric_{column}": str(index + column) for column in range(8)},
        )
        for index, value in enumerate(inconsistent)
    ]

    valid, reason, audited, details = _audit_indicator_pixels(image_path, rows,
        column_headers=['MFA', *rows[0].cells])

    assert valid is False
    assert reason == "indicator_pixel_model_conflict", details
    assert details["status"] == "conflict"
    assert len(details["corrections"]) > details["max_safe_corrections"]
    # Fail closed: do not publish pixel-rewritten rows when geometry and the
    # model disagree substantially.
    assert [row.checked_columns for row in audited] == inconsistent


@pytest.mark.parametrize('headers', [None, ['MFA'], ['CDA', 'MFA', 'MFA']])
def test_indicator_order_must_be_explicit_complete_and_unique(tmp_path: Path, headers) -> None:
    rows = [QwenTableRow(checked_columns=['CDA', 'MFA'],
                        cells={'CDA': '✓', 'MFA': '✓', 'score': '80.0'})]
    valid, reason, _, _ = _audit_indicator_pixels(tmp_path/'not-read.png', rows,
                                                  column_headers=headers)
    assert not valid
    assert reason == 'indicator_column_order_unknown'


def test_factory_reads_qwen_from_second_raw_bundle_line(tmp_path: Path) -> None:
    bundle = tmp_path / "learn-agent1.txt"
    bundle.write_text("test-deepseek-key\ntest-qwen-key\n", encoding="utf-8")
    with patch.dict(
        "os.environ",
        {
            "DASHSCOPE_API_KEY": "",
            "PAPERPILOT_QWEN_API_KEY_FILE": "",
            "PAPERPILOT_VISION_ADAPTER": "",
        },
        clear=False,
    ):
        adapter = make_vision_adapter(api_key_file=bundle)
    assert isinstance(adapter, QwenVisionEvidenceAdapter)


def test_factory_accepts_paperpilot_qwen_endpoint_alias(tmp_path: Path) -> None:
    bundle = tmp_path / "learn-agent1.txt"
    bundle.write_text("test-deepseek-key\nqwen: test-qwen-key\n", encoding="utf-8")
    endpoint = "https://workspace.example/compatible-mode/v1"
    with patch.dict(
        "os.environ",
        {
            "DASHSCOPE_API_KEY": "",
            "PAPERPILOT_QWEN_API_KEY_FILE": str(bundle),
            "PAPERPILOT_QWEN_BASE_URL": endpoint,
            "PAPERPILOT_QWEN_VISION_MODEL": "qwen-vision-test",
        },
        clear=False,
    ):
        adapter = QwenVisionEvidenceAdapter.from_env()

    assert adapter.base_url == endpoint
    assert adapter.model == "qwen-vision-test"
