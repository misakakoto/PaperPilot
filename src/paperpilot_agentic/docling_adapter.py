"""Subprocess adapter for frozen local Docling models, without a cloud fallback."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from .structured_extraction import ParsedArtifact, SourceRegion, TableCell, config_sha, file_sha

_LOCAL_LOCK = threading.Lock()
CHECKED = {'✓', '✔', '√', '☑'}
EMPTY = {'', '-', '—', '–', '✗', '×', '✘', '☐'}


class LocalExtractionError(ValueError):
    """A local failure retains dispatch/cache metrics without becoming source evidence."""
    def __init__(self, message, *, attempts=0, cache_hit=False, raw_path=None, sdk_requests=0):
        super().__init__(message)
        self.local_parse_attempts = attempts
        self.local_cache_hit = cache_hit
        self.raw_artifact_path = raw_path
        self.local_sdk_requests = sdk_requests


def project_table(data):
    """Expand explicit cell spans only; preserve original cells and hierarchical headers."""
    cells = [TableCell(row=int(c['start_row_offset_idx']), column=int(c['start_col_offset_idx']),
        row_span=int(c['end_row_offset_idx']) - int(c['start_row_offset_idx']),
        column_span=int(c['end_col_offset_idx']) - int(c['start_col_offset_idx']),
        text=str(c.get('text') or ''), column_header=bool(c.get('column_header')),
        row_header=bool(c.get('row_header')), bbox=c.get('bbox')) for c in data['table_cells']]
    nrows, ncols = int(data['num_rows']), int(data['num_cols'])
    if not 0 < nrows <= 500 or not 0 < ncols <= 100:
        raise ValueError('Invalid table dimensions')
    grid = {}
    for cell in cells:
        if cell.row + cell.row_span > nrows or cell.column + cell.column_span > ncols:
            raise ValueError('Cell span outside table')
        for r in range(cell.row, cell.row + cell.row_span):
            for c in range(cell.column, cell.column + cell.column_span):
                if (r, c) in grid:
                    raise ValueError('Overlapping table cells')
                grid[r, c] = cell
    hrows = sorted({r for (r, _), cell in grid.items() if cell.column_header})
    if not hrows or hrows != list(range(max(hrows) + 1)):
        raise ValueError('Missing or noncontiguous column headers')
    start = max(hrows) + 1
    if start >= nrows:
        raise ValueError('Table has no data rows')
    if not any(cell.text.strip() for cell in cells if cell.row >= start):
        raise ValueError('Empty table transcription; structure alone is not evidence')
    paths = []
    for c in range(ncols):
        paths.append(list(dict.fromkeys(grid[r, c].text.strip() for r in hrows
            if (r, c) in grid and grid[r, c].text.strip())))
    headers = [' / '.join(path) or f'[unlabeled column {c}]' for c, path in enumerate(paths)]
    if len(set(headers)) != ncols:
        raise ValueError('Ambiguous duplicate leaf headers')
    indicators = []
    for c, header in enumerate(headers):
        values = {grid[r, c].text.strip() if (r, c) in grid else '' for r in range(start, nrows)}
        if values & CHECKED and values <= CHECKED | EMPTY:
            indicators.append(header)
    rows = []
    for r in range(start, nrows):
        if any((r, c) not in grid for c in range(ncols)):
            raise ValueError('Missing table cells')
        values = {headers[c]: grid[r, c].text.strip() for c in range(ncols)}
        rows.append({'row_label': '', 'checked_columns': [h for h in indicators if values[h] in CHECKED],
                     'cells': {h: v for h, v in values.items() if h not in indicators}})
    return cells, headers, paths, indicators, rows


def candidate_box(candidate, document, page):
    for provenance in candidate.get('prov') or []:
        if provenance.get('page_no') != page:
            continue
        box = provenance['bbox']
        if str(box.get('coord_origin', '')).upper().endswith('BOTTOMLEFT'):
            height = document['pages'][str(page)]['size']['height']
            return [box['l'], height - box['t'], box['r'], height - box['b']]
        return [box['l'], box['t'], box['r'], box['b']]
    return None


def choose_candidate(raw, source, kind):
    choices = []
    for candidate in raw.get('candidates') or []:
        box = candidate_box(candidate, raw['document'], source.page)
        if not box:
            continue
        a, b, c, d = source.bbox
        x, y, z, w = box
        overlap = max(0, min(c, z) - max(a, x)) * max(0, min(d, w) - max(b, y))
        if overlap / max(1, (z - x) * (w - y)) < .75:
            continue
        captions = ' '.join(candidate.get('captions_text') or [])
        if kind == 'table' and source.label and captions:
            from .layout_identity import _explicit_layout_label
            if _explicit_layout_label(captions, 'table') != _explicit_layout_label(source.label, 'table'):
                continue
        choices.append(candidate)
    if len(choices) != 1:
        raise ValueError(f'Expected one bound {kind}, found {len(choices)}')
    return choices[0]


class DoclingEvidenceAdapter:
    local_execution = True
    model = 'docling-local-cpu'

    def __init__(self, python, models_path, cache_path, *, timeout=300):
        self.python = Path(python).resolve()
        self.models = Path(models_path).resolve()
        self.cache = Path(cache_path).resolve()
        self.timeout = min(300, max(1, int(timeout)))
        manifest = self.models / 'weights-manifest.json'
        if not self.python.is_file() or not manifest.is_file():
            raise ValueError('Docling environment or frozen weights missing; run setup_docling.py')
        self.manifest = json.loads(manifest.read_text(encoding='utf-8'))
        installed = subprocess.run([str(self.python), '-I', '-c',
            'import importlib.metadata,json,sys; '
            'print(json.dumps({n:importlib.metadata.version(n) for n in json.loads(sys.argv[1])}))',
            json.dumps(list(self.manifest['packages']))], capture_output=True, text=True,
            timeout=15, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if json.loads(installed.stdout) != self.manifest['packages']:
            raise ValueError('Docling dependency identity changed; freeze the environment again')
        for item in self.manifest['files']:
            weight = (self.models / item['path']).resolve()
            weight.relative_to(self.models)
            if not weight.is_file() or file_sha(weight) != item['sha256']:
                raise ValueError('Docling weight identity changed')
        self.weights_sha = file_sha(manifest)
        self.identity = {'parser': 'docling', 'version': self.manifest['docling_version'],
            'worker_sha256': file_sha(Path(__file__).with_name('docling_worker.py')),
            'weights_sha256': self.weights_sha, 'device': 'cpu', 'threads': 2,
            'ocr': 'rapidocr-torch-en-bitmaps', 'pdf_backend': 'pypdfium2',
            'cell_matching': True, 'formula_for_equations': True,
            'formula_scope': 'requested-source-region-v1',
            'remote': False, 'timeout': self.timeout}

    @classmethod
    def from_env(cls):
        repo = Path(__file__).resolve().parents[2]
        return cls(os.getenv('PAPERPILOT_DOCLING_PYTHON', str(repo / 'outputs/docling-env/Scripts/python.exe')),
            os.getenv('PAPERPILOT_DOCLING_MODELS', str(repo / 'outputs/docling-models')),
            os.getenv('PAPERPILOT_DOCLING_CACHE', str(repo / 'outputs/docling-cache')))

    def analyze(self, image_path, prompt, metadata=None, *, before_attempt=None):
        metadata = dict(metadata or {})
        kind = metadata['object_type']
        if kind not in {'table', 'equation'}:
            raise ValueError('Local Docling adapter supports tables and equations only')
        try:
            region = SourceRegion.model_validate(metadata['source_region'])
            pdf = Path(metadata['pdf_path']).resolve(strict=True)
            if file_sha(pdf) != region.paper_sha256 or file_sha(image_path) != region.crop_sha256:
                raise ValueError('Source changed before local parsing')
        except (ValueError, OSError) as exc:
            raise LocalExtractionError(str(exc)) from exc
        identity = {**self.identity, 'source': region.model_dump(), 'kind': kind}
        cache = self.cache / config_sha(identity)
        payload = {'pdf_path': str(pdf), 'models_path': str(self.models), 'page': region.page,
                   'kind': kind, 'source_region': region.model_dump(mode='json')}
        started = time.perf_counter()
        with _LOCAL_LOCK:
            cached = (cache / 'raw.json').is_file()
            if cached:
                try:
                    raw = json.loads((cache / 'raw.json').read_text(encoding='utf-8'))
                except (ValueError, OSError) as exc:
                    raise LocalExtractionError('Unreadable local raw cache', cache_hit=True,
                        raw_path=str(cache / 'raw.json')) from exc
            else:
                env = {k: v for k, v in os.environ.items() if k.upper() in {
                    'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'COMSPEC', 'NUMBER_OF_PROCESSORS',
                    'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'HOMEDRIVE', 'HOMEPATH'}}
                env.update(PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
                try:
                    process = subprocess.run([str(self.python), str(Path(__file__).with_name('docling_worker.py'))],
                        input=json.dumps(payload), encoding='utf-8', capture_output=True,
                        timeout=self.timeout, env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except subprocess.TimeoutExpired as exc:
                    raise LocalExtractionError('Local Docling timeout', attempts=1, sdk_requests=None) from exc
                raw = json.loads(process.stdout or '{}')
                if process.returncode or raw.get('error'):
                    raise LocalExtractionError(raw.get('error') or process.stderr[-1000:], attempts=1, sdk_requests=None)
                if raw.get('parser_version') != self.manifest['docling_version']:
                    raise ValueError('Docling version changed')
                cache.mkdir(parents=True, exist_ok=True)
                temporary = cache / f'raw.{os.getpid()}.tmp'
                temporary.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
                temporary.replace(cache / 'raw.json')
                (cache / 'worker.stderr.txt').write_text(process.stderr, encoding='utf-8')
        try:
            if raw.get('parser_version') != self.manifest['docling_version']:
                raise ValueError('Docling version changed')
            if raw.get('errors') or raw.get('status') not in {'success', 'ConversionStatus.SUCCESS'}:
                raise ValueError('Incomplete Docling conversion: ' + json.dumps(raw.get('errors') or raw.get('status')))
            candidate = choose_candidate(raw, region, kind)
            fields = {}
            indicators = []
            if kind == 'table':
                cells, headers, paths, indicators, rows = project_table(candidate['data'])
                fields.update(cells=cells, column_headers=headers, header_paths=paths, table_rows=rows)
            else:
                fields['latex'] = str(candidate.get('text') or '')
                if not fields['latex'].strip():
                    raise ValueError('Empty formula transcription')
        except (ValueError, KeyError, TypeError) as exc:
            raise LocalExtractionError(str(exc), attempts=int(not cached), cache_hit=cached,
                                       raw_path=str(cache / 'raw.json'), sdk_requests=int(not cached)) from exc
        artifact = ParsedArtifact(kind=kind, source=region, parser='docling',
            parser_version=raw['parser_version'], config_sha256=config_sha(self.identity),
            weights_sha256=self.weights_sha, parse_status='PARSED', validation_status='VALIDATED',
            raw_artifact_path=str(cache / 'raw.json'), **fields)
        # VALIDATED is source/structure, never an independent factual verdict.
        result_meta = {'model': self.model, 'provider_attempts': 0, 'local_parse_attempts': int(not cached),
            'local_sdk_requests': int(not cached),
            'local_cache_hit': cached, 'local_duration_ms': (time.perf_counter() - started) * 1000,
            'structured_extraction': artifact.model_dump(mode='json'), 'column_headers': artifact.column_headers,
            'audited_table_rows': artifact.table_rows, 'indicator_columns': indicators,
            'evidence_relevant': True, 'factual_review': 'NOT_REVIEWED'}
        return {'description': artifact.latex if kind == 'equation' else json.dumps(artifact.table_rows, ensure_ascii=False),
                'verified': kind == 'table', 'confidence': .6, 'metadata': result_meta}
