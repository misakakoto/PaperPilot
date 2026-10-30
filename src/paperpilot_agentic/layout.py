from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .models import stable_hash, utc_now

try:  # Optional, preferred parser.  The modern import keeps CLI stdout JSON-only.
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover - compatibility with older PyMuPDF releases
    try:
        import fitz  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        fitz = None

try:  # Text-only fallback.
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


_SECTION_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+|[IVX]+[.)]\s+)?[A-Z][^.!?]{1,100}$")
_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?\.?|table)\s*[A-Z0-9IVX.-]+", re.I)
_MATH_RE = re.compile(r"[=+*/−∑∏∫√±≤≥≈≠∞∂∇α-ωΑ-Ω_^{}]|\\(?:frac|sum|prod|math|begin)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_SPACED_SMALL_CAP_RE = re.compile(r"\b([A-Z])\s+([A-Z]{2,})\b")


@dataclass(slots=True)
class LayoutObject:
    object_id: str
    document_sha256: str
    page: int
    object_type: str
    bbox: list[float]
    text: str = ""
    caption: str | None = None
    section_path: list[str] = field(default_factory=list)
    artifact_path: str | None = None
    parser: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LayoutManifest:
    source_path: str
    document_sha256: str
    parser: str
    page_count: int
    objects: list[LayoutObject]
    created_at: str = field(default_factory=utc_now)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "document_sha256": self.document_sha256,
            "parser": self.parser,
            "page_count": self.page_count,
            "objects": [item.to_dict() for item in self.objects],
            "created_at": self.created_at,
            "warnings": list(self.warnings),
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(value: Iterable[float]) -> list[float]:
    return [round(float(x), 3) for x in value]


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    normalized = re.sub(
        r"(?<![A-Za-z])表\s*([0-9IVXLCDM]+)",
        r"Table \1",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?<![A-Za-z])图\s*([0-9IVXLCDM]+)",
        r"Figure \1",
        normalized,
        flags=re.IGNORECASE,
    )
    # Some IEEE embedded fonts expose small caps as ``A BLATION`` and
    # ``S PATIOTEMPORAL``. Repair those search-breaking splits without joining
    # ordinary initials or metric forms such as ``F 1``.
    previous = None
    while normalized != previous:
        previous = normalized
        normalized = _SPACED_SMALL_CAP_RE.sub(r"\1\2", normalized)
    return re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", normalized)


def _object_id(document_hash: str, page: int, object_type: str, bbox: list[float], text: str) -> str:
    return "obj_" + stable_hash(document_hash, page, object_type, bbox, text).split(":", 1)[1][:24]


def _caption_for_bbox(blocks: list[tuple[list[float], str]], bbox: list[float], kind: str) -> str | None:
    x0, y0, x1, y1 = bbox
    candidates: list[tuple[float, str]] = []
    for block_bbox, text in blocks:
        normalized = _normalize_text(text)
        if not normalized or not _CAPTION_RE.match(normalized):
            continue
        bx0, by0, bx1, by1 = block_bbox
        horizontal_overlap = max(0.0, min(x1, bx1) - max(x0, bx0))
        if horizontal_overlap <= 0:
            continue
        distance = min(abs(by0 - y1), abs(y0 - by1))
        expected = normalized.lower().startswith("table") if kind == "table" else normalized.lower().startswith(("fig", "figure"))
        penalty = 0.0 if expected else 1000.0
        candidates.append((distance + penalty, normalized))
    return min(candidates, default=(math.inf, None), key=lambda item: item[0])[1]


def _render_clip(page: Any, bbox: list[float], destination: Path, *, scale: float = 2.0) -> str | None:
    if fitz is None:
        return None
    rect = fitz.Rect(*bbox)
    page_rect = page.rect
    rect = rect & page_rect
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    pix.save(str(destination))
    return str(destination.resolve())


def _caption_context_bbox(
    page: Any,
    caption_bbox: list[float],
    simple_blocks: list[tuple[list[float], str]],
    kind: str,
) -> list[float]:
    """Bound caption-only crops to their column on likely two-column pages."""

    width = float(page.rect.width)
    height = float(page.rect.height)
    midpoint = width / 2.0
    left_blocks = sum(1 for bbox, _ in simple_blocks if float(bbox[2]) <= width * 0.56)
    right_blocks = sum(1 for bbox, _ in simple_blocks if float(bbox[0]) >= width * 0.44)
    likely_two_column = left_blocks >= 3 and right_blocks >= 3

    x0 = 0.0
    x1 = width
    y1_padding = 180.0
    caption_center = (float(caption_bbox[0]) + float(caption_bbox[2])) / 2.0
    # A caption centred on the page usually belongs to a table spanning both
    # columns.  Treating the exact midpoint as the right column used to crop
    # away the left-hand ablation indicators while retaining only the metric
    # values (observed on FlowMamba Table VII).
    centered_full_width = abs(caption_center - midpoint) <= width * 0.08
    if likely_two_column and kind == "table" and not centered_full_width:
        # A small gutter overlap retains rules and cells near the column edge.
        if caption_center < midpoint:
            x1 = width * 0.53
        else:
            x0 = width * 0.47
        y1_padding = 140.0
    return [
        x0,
        max(0.0, float(caption_bbox[1]) - 12.0),
        x1,
        min(height, float(caption_bbox[3]) + y1_padding),
    ]


def _section_heading(block: Mapping[str, Any], body_font: float) -> str | None:
    lines = block.get("lines") or []
    spans = [span for line in lines for span in (line.get("spans") or [])]
    if not spans:
        return None
    text = _normalize_text(" ".join(str(span.get("text", "")) for span in spans))
    if not text or len(text) > 120:
        return None
    max_size = max(float(span.get("size", body_font)) for span in spans)
    flags = max(int(span.get("flags", 0)) for span in spans)
    likely_bold = bool(flags & 16)
    if max_size >= body_font * 1.18 or likely_bold:
        if _SECTION_RE.match(text) or text.isupper():
            return text
    return None


def parse_pdf_layout(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    render_regions: bool = True,
    max_objects_per_page: int = 250,
) -> LayoutManifest:
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    document_hash = sha256_file(source)
    if fitz is not None:
        manifest = _parse_with_fitz(
            source,
            output,
            document_hash,
            render_regions=render_regions,
            max_objects_per_page=max_objects_per_page,
        )
    elif PdfReader is not None:
        manifest = _parse_with_pypdf(source, output, document_hash)
    else:
        raise RuntimeError("PDF parsing requires PyMuPDF (preferred) or pypdf")
    manifest_path = output / "layout_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _parse_with_fitz(
    source: Path,
    output: Path,
    document_hash: str,
    *,
    render_regions: bool,
    max_objects_per_page: int,
) -> LayoutManifest:
    doc = fitz.open(str(source))
    page_count = int(doc.page_count)
    objects: list[LayoutObject] = []
    warnings: list[str] = []
    section_stack: list[str] = []
    try:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            raw = page.get_text("dict", sort=True)
            text_blocks = [block for block in raw.get("blocks", []) if block.get("type") == 0]
            font_sizes = [
                float(span.get("size", 10.0))
                for block in text_blocks
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if _normalize_text(str(span.get("text", "")))
            ]
            body_font = median(font_sizes) if font_sizes else 10.0
            normalized_blocks: list[tuple[Mapping[str, Any], str, list[float]]] = []
            simple_blocks: list[tuple[list[float], str]] = []
            for block in text_blocks:
                text = _normalize_text(
                    " ".join(
                        str(span.get("text", ""))
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    )
                )
                if not text:
                    continue
                block_bbox = _bbox(block.get("bbox", [0, 0, 0, 0]))
                normalized_blocks.append((block, text, block_bbox))
                simple_blocks.append((block_bbox, text))

            # Object creation is deliberately a second pass: caption crops need
            # the complete page-wide block distribution to detect two columns.
            for block, text, block_bbox in normalized_blocks:
                heading = _section_heading(block, body_font)
                if heading:
                    # PDF hierarchy is often unavailable; keep a stable latest-heading path.
                    level_match = re.match(r"^(\d+(?:\.\d+)*)", heading)
                    if level_match:
                        level = level_match.group(1).count(".") + 1
                        section_stack = section_stack[: max(0, level - 1)] + [heading]
                    else:
                        section_stack = [heading]
                math_hits = len(_MATH_RE.findall(text))
                math_ratio = math_hits / max(1, len(text))
                caption_match = _CAPTION_RE.match(text)
                if caption_match and text.lower().startswith("table"):
                    object_type = "table"
                elif caption_match:
                    object_type = "figure"
                else:
                    object_type = "equation" if (math_hits >= 2 and (math_ratio > 0.025 or len(text) < 180)) else "text"
                artifact_path = None
                context_bbox = None
                if object_type == "equation" and render_regions:
                    artifact_path = _render_clip(
                        page,
                        block_bbox,
                        output / "objects" / f"p{page_number:04d}_equation_{len(objects):05d}.png",
                    )
                elif caption_match and render_regions:
                    # A caption-only table/figure is a locator, not the visual
                    # content itself.  Preserve a bounded page region so a
                    # configured vision/OCR adapter can inspect the actual
                    # object instead of being handed only the words "TABLE XII".
                    context_bbox = _caption_context_bbox(
                        page,
                        block_bbox,
                        simple_blocks,
                        object_type,
                    )
                    artifact_path = _render_clip(
                        page,
                        context_bbox,
                        output / "objects" / f"p{page_number:04d}_{object_type}_{len(objects):05d}.png",
                        scale=4.0 if object_type == "table" else 2.0,
                    )
                objects.append(
                    LayoutObject(
                        object_id=_object_id(document_hash, page_number, object_type, block_bbox, text),
                        document_sha256=document_hash,
                        page=page_number,
                        object_type=object_type,
                        bbox=block_bbox,
                        text=text,
                        caption=text if _CAPTION_RE.match(text) else None,
                        section_path=list(section_stack),
                        artifact_path=artifact_path,
                        parser="pymupdf",
                        metadata={
                            "body_font": body_font,
                            "caption_only": bool(caption_match),
                            "vision_verified": False,
                            "render_context_bbox": context_bbox if caption_match else None,
                            "render_scale": (4.0 if object_type == "table" else 2.0) if caption_match else None,
                        },
                    )
                )
                if len([x for x in objects if x.page == page_number]) >= max_objects_per_page:
                    warnings.append(f"Page {page_number}: object limit reached")
                    break

            # Native table extraction, when supported by the installed PyMuPDF version.
            try:
                finder = page.find_tables() if hasattr(page, "find_tables") else None
                tables = list(getattr(finder, "tables", []) or [])
            except Exception as exc:
                tables = []
                warnings.append(f"Page {page_number}: table extraction failed: {type(exc).__name__}: {exc}")
            for table_index, table in enumerate(tables):
                table_bbox = _bbox(table.bbox)
                try:
                    cells = table.extract()
                except Exception:
                    cells = []
                text = "\n".join(" | ".join(_normalize_text(str(cell or "")) for cell in row) for row in cells)
                caption = _caption_for_bbox(simple_blocks, table_bbox, "table")
                artifact_path = None
                if render_regions:
                    artifact_path = _render_clip(
                        page,
                        table_bbox,
                        output / "objects" / f"p{page_number:04d}_table_{table_index:03d}.png",
                    )
                objects.append(
                    LayoutObject(
                        object_id=_object_id(document_hash, page_number, "table", table_bbox, text or caption or ""),
                        document_sha256=document_hash,
                        page=page_number,
                        object_type="table",
                        bbox=table_bbox,
                        text=text,
                        caption=caption,
                        section_path=list(section_stack),
                        artifact_path=artifact_path,
                        parser="pymupdf",
                        metadata={"rows": len(cells), "columns": max((len(row) for row in cells), default=0)},
                    )
                )

            # Raster image regions. Vector figures are represented by their caption text blocks.
            seen_rects: set[tuple[float, ...]] = set()
            try:
                images = page.get_images(full=True)
            except Exception:
                images = []
            figure_index = 0
            for image in images:
                xref = image[0]
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                for rect in rects:
                    figure_bbox = _bbox(rect)
                    key = tuple(figure_bbox)
                    if key in seen_rects or (figure_bbox[2] - figure_bbox[0]) < 30 or (figure_bbox[3] - figure_bbox[1]) < 30:
                        continue
                    seen_rects.add(key)
                    caption = _caption_for_bbox(simple_blocks, figure_bbox, "figure")
                    artifact_path = None
                    if render_regions:
                        artifact_path = _render_clip(
                            page,
                            figure_bbox,
                            output / "objects" / f"p{page_number:04d}_figure_{figure_index:03d}.png",
                        )
                    objects.append(
                        LayoutObject(
                            object_id=_object_id(document_hash, page_number, "figure", figure_bbox, caption or f"image xref {xref}"),
                            document_sha256=document_hash,
                            page=page_number,
                            object_type="figure",
                            bbox=figure_bbox,
                            text=caption or "",
                            caption=caption,
                            section_path=list(section_stack),
                            artifact_path=artifact_path,
                            parser="pymupdf",
                            metadata={
                                "xref": int(xref),
                                "analysis_mode": "caption_only",
                                "vision_verified": False,
                            },
                        )
                    )
                    figure_index += 1

            # Captions without detected raster images still carry useful, explicitly text-only evidence.
            existing_captions = {item.caption for item in objects if item.page == page_number and item.caption}
            for caption_bbox, caption_text in simple_blocks:
                if caption_text in existing_captions or not _CAPTION_RE.match(caption_text):
                    continue
                kind = "table" if caption_text.lower().startswith("table") else "figure"
                objects.append(
                    LayoutObject(
                        object_id=_object_id(document_hash, page_number, kind, caption_bbox, caption_text),
                        document_sha256=document_hash,
                        page=page_number,
                        object_type=kind,
                        bbox=caption_bbox,
                        text=caption_text,
                        caption=caption_text,
                        section_path=list(section_stack),
                        parser="pymupdf",
                        metadata={"caption_only": True, "vision_verified": False},
                    )
                )
    finally:
        doc.close()
    return LayoutManifest(
        source_path=str(source),
        document_sha256=document_hash,
        parser="pymupdf",
        page_count=page_count,
        objects=objects,
        warnings=warnings,
    )


def _parse_with_pypdf(source: Path, output: Path, document_hash: str) -> LayoutManifest:
    reader = PdfReader(str(source))
    objects: list[LayoutObject] = []
    warnings = ["PyMuPDF is unavailable: layout coordinates and rendered regions are not available"]
    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        text = page.extract_text() or ""
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]
        for index, paragraph in enumerate(paragraphs):
            normalized = _normalize_text(paragraph)
            object_type = "equation" if len(_MATH_RE.findall(normalized)) >= 2 else "text"
            synthetic_bbox = [0.0, float(index), 1.0, float(index + 1)]
            objects.append(
                LayoutObject(
                    object_id=_object_id(document_hash, page_number, object_type, synthetic_bbox, normalized),
                    document_sha256=document_hash,
                    page=page_number,
                    object_type=object_type,
                    bbox=synthetic_bbox,
                    text=normalized,
                    caption=normalized if _CAPTION_RE.match(normalized) else None,
                    parser="pypdf-text-only",
                    metadata={"synthetic_bbox": True},
                )
            )
    return LayoutManifest(
        source_path=str(source),
        document_sha256=document_hash,
        parser="pypdf-text-only",
        page_count=len(reader.pages),
        objects=objects,
        warnings=warnings,
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def search_layout_objects(
    manifest: LayoutManifest | Mapping[str, Any] | str | Path,
    *,
    query: str,
    object_type: str | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        data = load_manifest(manifest)
    elif isinstance(manifest, LayoutManifest):
        data = manifest.to_dict()
    else:
        data = dict(manifest)
    objects = [dict(raw) for raw in data.get("objects", [])]
    objects_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in objects:
        objects_by_page.setdefault(int(item.get("page") or 0), []).append(item)
    normalized_query = _normalize_text(query)
    query_tokens = {token.lower() for token in _TOKEN_RE.findall(normalized_query)}
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in objects:
        if object_type and item.get("object_type") != object_type:
            continue
        page = int(item.get("page") or 0)
        item_bbox = item.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        context: list[dict[str, Any]] = []
        for candidate in objects_by_page.get(page, []):
            if candidate.get("object_id") == item.get("object_id"):
                continue
            candidate_bbox = candidate.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            vertical_gap = max(
                0.0,
                float(candidate_bbox[1]) - float(item_bbox[3]),
                float(item_bbox[1]) - float(candidate_bbox[3]),
            )
            if vertical_gap <= 90.0:
                context.append(candidate)

        # Captions and their discussion are frequently split across columns or
        # page boundaries.  Bind explicit references such as "In Table XII"
        # from the adjacent page without merging unrelated page text.
        label_match = re.search(
            r"\b(table|fig(?:ure)?)\s*([A-Z0-9IVXLCDM.-]+)",
            str(item.get("caption") or item.get("text") or ""),
            re.IGNORECASE,
        )
        if label_match:
            label_pattern = re.compile(
                rf"\b{re.escape(label_match.group(1))}\s*{re.escape(label_match.group(2))}\b",
                re.IGNORECASE,
            )
            for nearby_page in (page - 1, page, page + 1):
                for candidate in objects_by_page.get(nearby_page, []):
                    if candidate.get("object_id") == item.get("object_id"):
                        continue
                    if label_pattern.search(str(candidate.get("text") or "")):
                        context.append(candidate)

        unique_context: list[dict[str, Any]] = []
        seen_context: set[str] = set()
        for candidate in context:
            identity = str(candidate.get("object_id") or stable_hash(candidate))
            if identity in seen_context:
                continue
            seen_context.add(identity)
            unique_context.append(candidate)
        unique_context = unique_context[:8]
        context_text = _normalize_text(
            " ".join(
                str(candidate.get("text") or candidate.get("caption") or "").strip()
                for candidate in unique_context
                if str(candidate.get("text") or candidate.get("caption") or "").strip()
            )
        )[:4_000]
        haystack = " ".join(
            [
                str(item.get("text") or ""),
                str(item.get("caption") or ""),
                " ".join(item.get("section_path") or []),
                context_text,
            ]
        )
        normalized_haystack = _normalize_text(haystack)
        tokens = {token.lower() for token in _TOKEN_RE.findall(normalized_haystack)}
        overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
        phrase_bonus = (
            0.3
            if normalized_query and normalized_query.casefold() in normalized_haystack.casefold()
            else 0.0
        )
        query_folded = normalized_query.casefold()
        haystack_folded = normalized_haystack.casefold()
        # Task-intent terms carry more routing information than ubiquitous
        # scholarly words such as "table", "model" or "results". Without
        # this weighting, a generic performance table can outrank the actual
        # ablation table merely by sharing more common tokens.
        intent_bonus = 0.0
        intent_markers = (
            (("ablation", "消融"), 0.8),
            (("quantitative", "numeric", "数值", "定量"), 0.25),
            (("qualitative", "visualization", "可视化", "定性"), 0.35),
            (("formula", "equation", "公式", "方程"), 0.45),
        )
        for markers, weight in intent_markers:
            if any(marker in query_folded for marker in markers) and any(
                marker in haystack_folded for marker in markers
            ):
                intent_bonus += weight
        explicit_label_bonus = 0.0
        if label_match:
            normalized_label = _normalize_text(
                f"{label_match.group(1)} {label_match.group(2)}"
            ).casefold()
            if re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_label)}(?![a-z0-9])",
                normalized_query.casefold(),
            ):
                explicit_label_bonus = 0.75
        type_bonus = 0.2 if object_type and item.get("object_type") == object_type else 0.0
        caption_bonus = 0.1 if item.get("caption") else 0.0
        score = (
            overlap
            + phrase_bonus
            + intent_bonus
            + explicit_label_bonus
            + type_bonus
            + caption_bonus
        )
        item["retrieval_score"] = round(score, 6)
        item["retrieval_context"] = context_text
        item["retrieval_context_object_ids"] = [
            candidate.get("object_id") for candidate in unique_context if candidate.get("object_id")
        ]
        item["retrieval_context_bboxes"] = [
            candidate.get("bbox") for candidate in unique_context if candidate.get("bbox")
        ]
        scored.append((score, item))
    scored.sort(key=lambda pair: (pair[0], -int(pair[1].get("page", 0))), reverse=True)
    return [item for _, item in scored[: max(1, top_k)]]


def inspect_layout_object(manifest_path: str | Path, object_id: str) -> dict[str, Any]:
    data = load_manifest(manifest_path)
    for item in data.get("objects", []):
        if item.get("object_id") == object_id:
            result = dict(item)
            result["source_path"] = data.get("source_path")
            result["document_sha256"] = data.get("document_sha256")
            if result.get("object_type") == "figure" and not (result.get("metadata") or {}).get("vision_verified"):
                result["inspection_warning"] = (
                    "No vision model was configured; this object is represented by its caption and location only."
                )
            return result
    raise KeyError(f"Unknown layout object: {object_id}")
