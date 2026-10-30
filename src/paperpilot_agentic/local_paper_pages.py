"""Bounded page reading from SHA-verified PDFs in configured corpus roots."""
import hashlib
from pathlib import Path

from pypdf import PdfReader

from .models import ToolResult

POLICY = 'registered-local-pages-v1'


def read_pages(adapter, args):
    if (set(args) != {'paper_sha256', 'pages'} or not isinstance(args['paper_sha256'], str)
            or len(args['paper_sha256']) != 64 or any(c not in '0123456789abcdef' for c in args['paper_sha256'])
            or not isinstance(args['pages'], list) or not 1 <= len(args['pages']) <= 3
            or any(type(p) is not int or p < 1 for p in args['pages'])
            or len(set(args['pages'])) != len(args['pages'])):
        raise ValueError('Select a registered paper SHA and 1..3 distinct positive page numbers')
    resolver = getattr(adapter, 'resolve_pdf_paths', None)
    if not callable(resolver):
        raise ValueError('Configured corpus cannot resolve registered local PDF pages')
    sha = args['paper_sha256']
    roots = [Path(r).resolve() for r in getattr(adapter, 'paper_roots', ())]
    paths = [Path(p).resolve() for p in resolver({sha})]
    if len(paths) != 1 or not any(paths[0] == r or r.is_dir() and paths[0].is_relative_to(r) for r in roots):
        raise ValueError('Registered paper must resolve uniquely inside configured corpus roots')
    path = paths[0]
    if path.suffix.casefold() != '.pdf' or path.stat().st_size > 100*1024*1024:
        raise ValueError('Registered PDF exceeds the 100 MiB reading bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Registered PDF identity changed')
    from io import BytesIO
    reader = PdfReader(BytesIO(raw))
    if max(args['pages']) > len(reader.pages):
        raise ValueError('Requested page exceeds the registered document')
    evidence = []
    for number in args['pages']:
        text = reader.pages[number-1].extract_text() or ''
        if not text.strip():
            raise ValueError('Requested page has no extractable text; do not invent OCR evidence')
        evidence.append({'evidence_type': 'TEXT', 'content': text[:20000], 'page': number,
            'source_uri': f'paper://{sha}/page/{number}/{POLICY}',
            'provenance': {'paper_sha256': sha, 'document_sha256': sha, 'extraction_policy': POLICY,
                'complete_page': len(text) <= 20000, 'original_characters': len(text),
                'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'reading_order': 'PDF_EXTRACTED_TEXT_ORDER_NOT_GUARANTEED'}})
    return ToolResult(ok=True, output={'paper_sha256': sha, 'requested_pages': args['pages'],
                                     'page_count': len(reader.pages)}, evidence=evidence)
