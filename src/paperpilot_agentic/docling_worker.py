"""Local-only Docling worker. stdin/stdout JSON; no credentials or question sent to a service."""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path


def retain_ocr_cells(cells):
    """Return source-cell indices and a record of duplicate physical observations."""
    retained, removed, groups = [], [], {}
    for position, cell in enumerate(cells):
        box = _ocr_box(cell)
        key = (cell.get('text'), cell.get('orig'), cell.get('text_direction'))
        duplicate = None
        if box is not None:
            for earlier, other in groups.get(key, []):
                intersection = (max(0, min(box[2], other[2]) - max(box[0], other[0])) *
                                max(0, min(box[3], other[3]) - max(box[1], other[1])))
                area = (box[2] - box[0]) * (box[3] - box[1])
                other_area = (other[2] - other[0]) * (other[3] - other[1])
                iom = intersection / min(area, other_area)
                iou = intersection / (area + other_area - intersection)
                near_center = all(abs((box[i] + box[i + 2]) - (other[i] + other[i + 2])) / 2
                    <= .15 * min(box[i + 2] - box[i], other[i + 2] - other[i]) for i in (0, 1))
                if iom >= .9 and iou >= .65 and near_center:
                    duplicate = {'retained_position': earlier, 'removed_position': position,
                                 'removed_cell': cell, 'intersection_over_min': iom,
                                 'intersection_over_union': iou}
                    break
        if duplicate is not None:
            removed.append(duplicate)
        else:
            retained.append(position)
            if box is not None:
                groups.setdefault(key, []).append((position, box))
    return retained, removed


def _ocr_box(cell):
    if cell.get('from_ocr') is not True or not cell.get('text'):
        return None
    rect = cell.get('rect') or {}
    if rect.get('coord_origin') != 'TOPLEFT':
        return None
    try:
        xs = [float(rect[f'r_x{i}']) for i in range(4)]
        ys = [float(rect[f'r_y{i}']) for i in range(4)]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in xs + ys):
        return None
    if not (xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3]):
        return None  # Rotated envelopes do not prove the same physical observation.
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    return (left, top, right, bottom) if right > left and bottom > top else None


class SourceOcrStage:
    def __init__(self, delegate, diagnostics):
        self.delegate = delegate
        self.diagnostics = diagnostics

    def __call__(self, conv_res, pages):
        for page in self.delegate(conv_res, pages):
            cells = list(page.cells)
            retained, removed = retain_ocr_cells([c.model_dump(mode='json') for c in cells])
            if removed:
                page.parsed_page.textline_cells = [cells[i] for i in retained]
            self.diagnostics.append({'page': page.page_no, 'input_cells': len(cells),
                                     'retained_cells': len(retained), 'removed': removed})
            yield page


class SourceFormulaStage:
    elements_batch_size = 1

    def __init__(self, delegate, source, diagnostics):
        self.delegate = delegate
        self.source = source
        self.diagnostics = diagnostics
        self.enriched_refs = set()

    def is_processable(self, doc, element):
        if not self.delegate.is_processable(doc, element):
            return False
        for provenance in getattr(element, 'prov', []):
            if provenance.page_no != self.source['page']:
                continue
            try:
                box = provenance.bbox.model_dump(mode='json')
                left, top, right, bottom = (float(box[k]) for k in ('l', 't', 'r', 'b'))
                if box.get('coord_origin') == 'BOTTOMLEFT':
                    height = doc.pages[provenance.page_no].size.height
                    top, bottom = height - top, height - bottom
                elif box.get('coord_origin') != 'TOPLEFT':
                    continue
                if not all(math.isfinite(v) for v in (left, top, right, bottom)):
                    continue
                if right <= left or bottom <= top:
                    continue
                a, b, c, d = self.source['bbox']
                overlap = max(0, min(c, right) - max(a, left)) * max(0, min(d, bottom) - max(b, top))
                if overlap / ((right - left) * (bottom - top)) >= .75:
                    return True
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return False

    def prepare_element(self, conv_res, element):
        if not self.is_processable(conv_res.document, element):
            return None
        return self.delegate.prepare_element(conv_res, element)

    def __call__(self, doc, element_batch):
        batch = list(element_batch)
        self.diagnostics.append({'item_refs': [e.item.self_ref for e in batch]})
        for element in self.delegate(doc, batch):
            self.enriched_refs.add(element.self_ref)
            yield element


def convert(payload):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions, AcceleratorDevice, PdfPipelineOptions, RapidOcrOptions)
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel

    ocr_diagnostics = []
    formula_diagnostics, formula_stages = [], []
    class SourceBoundPdfPipeline(StandardPdfPipeline):
        def __init__(self, pipeline_options):
            super().__init__(pipeline_options)
            self.ocr_model = SourceOcrStage(self.ocr_model, ocr_diagnostics)
            if payload['kind'] == 'equation':
                stage = SourceFormulaStage(self.enrichment_pipe[0], payload['source_region'],
                                           formula_diagnostics)
                if not isinstance(stage.delegate, CodeFormulaVlmModel):
                    raise ValueError('Unexpected formula stage; local SDK contract changed')
                self.enrichment_pipe[0] = stage
                formula_stages.append(stage)

    path = Path(payload['pdf_path']).resolve(strict=True)
    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CPU)
    options.do_ocr = True  # The frozen PDFs contain rasterized table bodies.
    options.ocr_options = RapidOcrOptions(backend='torch', lang=['en'], force_full_page_ocr=False)
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = payload.get('cell_matching', True)
    options.do_formula_enrichment = payload['kind'] == 'equation'
    options.enable_remote_services = False
    options.artifacts_path = Path(payload['models_path']).resolve(strict=True)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
        pipeline_options=options, backend=PyPdfiumDocumentBackend,
        pipeline_cls=SourceBoundPdfPipeline)})
    if payload['kind'] == 'document':
        result = converter.convert(path)
    else:
        page = int(payload['page'])
        result = converter.convert(path, page_range=(page, page))
    doc = result.document
    raw = doc.export_to_dict()
    candidates = []
    enriched = set().union(*(stage.enriched_refs for stage in formula_stages))
    for item, _ in doc.iterate_items():
        label = str(getattr(item, 'label', '')).split('.')[-1].lower()
        if (payload['kind'] == 'table' and label != 'table') or (
                payload['kind'] == 'equation' and label != 'formula'):
            continue
        if payload['kind'] == 'equation' and item.self_ref not in enriched:
            continue  # A layout/OCR formula is not a successful formula-model output.
        entry = item.model_dump(mode='json')
        entry['captions_text'] = [ref.resolve(doc).text for ref in getattr(item, 'captions', [])]
        candidates.append(entry)
    return {'document': raw, 'candidates': candidates, 'ocr_source_cells': ocr_diagnostics,
            'formula_enrichment': formula_diagnostics,
            'parser_version': importlib.metadata.version('docling'), 'status': str(result.status),
            'errors': [error.model_dump(mode='json') for error in result.errors],
            'table_stage': [{'page': p.page_no,
                'prediction': p.predictions.tablestructure.model_dump(mode='json')
                    if p.predictions.tablestructure else None} for p in result.pages]}


def main():
    payload = json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = convert(payload)
        json.dump(result, sys.stdout, ensure_ascii=False)
    except Exception as exc:
        json.dump({'error': f'{type(exc).__name__}: {exc}'}, sys.stdout, ensure_ascii=False)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
