"""Local structured ingestion with explicit text-only fallback and source locations.

No experiment facts are guessed from numeric cells. Reports are admitted later
only through the source-checked report contract in paper_evidence_index.
"""
import json
import os
from pathlib import Path
import subprocess

from .paper_catalog import identity, frontmatter_observations
from .paper_structure import digest, save_json


def _location(item, sha):
    spans = [{'page': p['page_no'], 'bbox': p.get('bbox'), 'charspan': p.get('charspan')}
             for p in item.get('prov', []) if type(p.get('page_no')) is int and p['page_no'] > 0]
    if not spans:
        raise ValueError('Structured item has no original PDF provenance')
    return {'paper_sha256': sha, 'page_range': [min(p['page'] for p in spans), max(p['page'] for p in spans)],
            'object_ref': item['self_ref'], 'spans': spans, 'quality': 'STRUCTURED_PARSE_NOT_FACT_VERIFIED'}


def document_units(raw, sha, *, table_rows=24):
    """Traverse Docling body references in reading order, preserving heading paths."""
    if type(table_rows) is not int or not 1 <= table_rows <= 64:
        raise ValueError('Table groups require 1..64 rows')
    doc = raw['document']
    lookup = {}
    for name in ('texts', 'tables', 'pictures', 'groups'):
        lookup.update({i['self_ref']: i for i in doc.get(name, [])})
    sections, visited, units = [], set(), []

    def resolve_refs(refs):
        return [lookup[r['$ref']].get('text', '') for r in refs if r.get('$ref') in lookup]

    def walk(refs):
        for ref in refs:
            name = ref.get('$ref')
            if name in visited:
                raise ValueError('Repeated/cyclic document reference')
            visited.add(name)
            if name not in lookup:
                raise ValueError('Document child reference is missing')
            item = lookup[name]
            label = item.get('label', '')
            text = item.get('text', '')
            if label in {'section_header', 'title'}:
                level = max(1, int(item.get('level', 1)))
                del sections[level-1:]
                sections.append(text)
            elif label not in {'page_header', 'page_footer'}:
                kind = {'table': 'TABLE', 'formula': 'FORMULA', 'picture': 'FIGURE'}.get(label, 'TEXT')
                if text.strip() and not item.get('prov'):
                    raise ValueError('Text item has no original PDF provenance')
                if item.get('prov') and (text.strip() or kind in {'TABLE', 'FIGURE'}):
                    source = _location(item, sha)
                    if kind == 'TABLE':
                        data = item.get('data') or {}
                        cells = data.get('table_cells', [])
                        headers = [c for c in cells if c.get('column_header')]
                        from .paper_tables import table_structure
                        validation = table_structure(data)
                        # Preserve cells exactly, including spans and empty cells.
                        normalized = [{'row': c['start_row_offset_idx'], 'column': c['start_col_offset_idx'],
                            'row_end': c['end_row_offset_idx'], 'column_end': c['end_col_offset_idx'],
                            'text': c['text'], 'column_header': bool(c.get('column_header')),
                            'row_header': bool(c.get('row_header')), 'bbox': c.get('bbox')} for c in cells]
                        captions = resolve_refs(item.get('captions', []))
                        notes = resolve_refs(item.get('footnotes', []))
                        row_ids = sorted({c['row'] for c in normalized if not c['column_header']})
                        if not row_ids:
                            raise ValueError('Table has no transcribed data rows')
                        for start in range(0, len(row_ids), table_rows):
                            group = set(row_ids[start:start+table_rows])
                            selected = [c for c in normalized if c['column_header'] or
                                        any(c['row'] <= r < c['row_end'] for r in group)]
                            table = {'source': source, 'caption': captions, 'footnotes': notes,
                                'num_rows': data.get('num_rows'), 'num_cols': data.get('num_cols'),
                                'column_headers': [c['text'] for c in headers], 'cells': selected,
                                'row_group': sorted(group), 'header_repeated': True,
                                'parse_status': 'PARSED' if not validation['structure_errors'] else 'INVALID_STRUCTURE',
                                **validation,
                                'fact_review': 'NOT_REVIEWED'}
                            body = '\n'.join(captions + [f"r{c['row']} c{c['column']}: {c['text']}" for c in selected] + notes)
                            units.append({'id': identity([sha, name, sorted(group)]), 'paper_sha256': sha,
                                'kind': kind, 'section': ' / '.join(sections), 'text': body,
                                'source': source, 'object': table})
                    else:
                        if kind == 'FIGURE':
                            text = '\n'.join(resolve_refs(item.get('captions', [])))
                        if text.strip():
                            units.append({'id': identity([sha, name]), 'paper_sha256': sha, 'kind': kind,
                                'section': ' / '.join(sections), 'text': text, 'source': source,
                                'object': {'label': label, 'raw_ref': name} if kind != 'TEXT' else None})
            walk(item.get('children', []))
    walk(doc.get('body', {}).get('children', []))
    # Some Docling PDFs leave starred table notes as ordinary neighbouring text.
    # Associate only an explicit shared marker and an unambiguous geometric match.
    for table in [u for u in units if u['kind'] == 'TABLE']:
        captions = table['object']['caption']
        marker = next((c.rstrip()[-1] for c in captions if c.rstrip().endswith(('*', '†', '‡'))), None)
        if not marker or table['object']['footnotes']:
            continue
        bound = table['source']['spans'][-1]
        a = bound.get('bbox') or {}
        matches = []
        for candidate in units:
            if candidate['kind'] != 'TEXT' or not candidate['text'].lstrip().startswith(marker):
                continue
            for span in candidate['source']['spans']:
                b = span.get('bbox') or {}
                if span['page'] != bound['page'] or b.get('coord_origin') != a.get('coord_origin'):
                    continue
                try:
                    gap = a['b']-b['t'] if a['coord_origin'] == 'BOTTOMLEFT' else b['t']-a['b']
                    width = min(a['r'], b['r'])-max(a['l'], b['l'])
                    if 0 <= gap <= 18 and width >= .5*min(a['r']-a['l'], b['r']-b['l']):
                        matches.append(candidate)
                except (KeyError, TypeError):
                    continue
        if len(matches) == 1:
            note = matches[0]
            table['object']['footnotes'] = [note['text']]
            table['object']['footnote_sources'] = [note['source']]
            table['object']['footnote_link_policy'] = 'SHARED_MARKER_AND_ADJACENT_PDF_REGION_NOT_ENTAILMENT'
            table['text'] += '\n' + note['text']
            table['source']['associated_notes'] = [note['source']]
    return units


def parse_document(pdf, expected_sha, output, *, docling=None, table_rows=24, cell_matching=True):
    """Registered caller supplies a hash-checked path; outputs never overwrite."""
    pdf, output = Path(pdf).resolve(strict=True), Path(output).resolve()
    if type(cell_matching) is not bool:
        raise ValueError('Document cell matching must be explicitly boolean')
    if digest(pdf.read_bytes()) != expected_sha:
        raise ValueError('PDF version changed before ingestion')
    output.mkdir(parents=True, exist_ok=False)
    raw, failure = None, 'DOCLING_NOT_CONFIGURED'
    if docling is not None:
        worker = Path(__file__).with_name('docling_worker.py')
        if digest(worker.read_bytes()) != docling.identity['worker_sha256']:
            raise ValueError('Docling worker changed after preflight')
        env = {k: v for k, v in os.environ.items() if k.upper() in {
            'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'COMSPEC', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA'}}
        env.update(PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', PYTHONDONTWRITEBYTECODE='1')
        try:
            result = subprocess.run([str(docling.python), str(worker)],
                input=json.dumps({'kind': 'document', 'pdf_path': str(pdf), 'models_path': str(docling.models), 'cell_matching': cell_matching}),
                encoding='utf-8', capture_output=True, timeout=docling.timeout, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (output / 'worker.stderr.txt').write_text(result.stderr, encoding='utf-8')
            raw = json.loads(result.stdout or '{}')
            save_json(output / 'raw.json', raw)
            if result.returncode or raw.get('errors') or raw.get('status') not in {'success', 'ConversionStatus.SUCCESS'}:
                raise ValueError(raw.get('error') or 'Incomplete structured conversion')
            if raw.get('parser_version') != docling.manifest['docling_version']:
                raise ValueError('Parser version changed')
            units = document_units(raw, expected_sha, table_rows=table_rows)
            if not units:
                raise ValueError('Structured parser returned no located units')
            failure = None
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError) as exc:
            failure = f'{type(exc).__name__}: {exc}'
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    first = reader.pages[0].extract_text() or ''
    if failure:
        units = []
        for page, obj in enumerate(reader.pages, 1):
            text = obj.extract_text() or ''
            if text.strip():
                units.append({'id': identity([expected_sha, 'pypdf-page', page, text]), 'paper_sha256': expected_sha,
                    'kind': 'TEXT', 'section': '', 'text': text,
                    'source': {'page_range': [page, page], 'quality': 'PAGE_TEXT_FALLBACK_READING_ORDER_UNVERIFIED'}})
    if digest(pdf.read_bytes()) != expected_sha:
        raise ValueError('PDF changed while parsing')
    raw_hash = digest((output / 'raw.json').read_bytes()) if raw is not None else None
    for u in units:
        u['source']['parser'] = ({'identity': {**docling.identity, 'cell_matching': cell_matching,
                                            'scope': 'FULL_DOCUMENT', 'formula_enrichment': False}, 'raw_sha256': raw_hash,
                                 'raw_receipt': str(output / 'raw.json')} if not failure else
                                {'name': 'pypdf', 'fallback_reason': failure})
        if u['kind'] == 'TABLE':
            from .paper_tables import literal_reports
            u['reports'] = literal_reports(u)
    receipt = {'paper_sha256': expected_sha, 'units': units,
        'metadata_observations': frontmatter_observations(first, expected_sha),
        'parser': docling.identity if docling else None, 'status': 'FALLBACK_TEXT_ONLY' if failure else 'STRUCTURED',
        'fallback_reason': failure, 'raw_sha256': raw_hash,
        'fact_review': 'NOT_REVIEWED'}
    save_json(output / 'parsed.json', receipt)
    return receipt
