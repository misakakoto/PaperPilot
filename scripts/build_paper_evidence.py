"""Build a separate, explicitly selected paper_evidence_v2 snapshot from registered PDFs.

No active index writes, automatic dependency installation, API, or training.
Default ingestion preserves the current text units; --structured changes parsing
in a separate experiment. Bibliographic enrichment accepts frozen observations.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.paper_catalog import frontmatter_observations
from paperpilot_agentic.paper_evidence_index import build_index
from paperpilot_agentic.paper_structure import digest, save_json
from paperpilot_agentic.token_evidence import TokenBudget, encoded
from paperpilot_agentic.adapters import PaperPilotKnowledgeSearchAdapter
from paperpilot_retrieval import FastEmbedder


class UncachedEmbedder(FastEmbedder):
    def embed_documents(self, texts):
        return self._embed(texts)


def frozen_input(path, namespace, scope=None):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as c:
        c.row_factory = sqlite3.Row
        papers = [dict(r) for r in c.execute('SELECT * FROM papers WHERE namespace=? AND active=1', (namespace,))]
        if scope is not None:
            papers = [p for p in papers if p['sha256'] in scope]
            if {p['sha256'] for p in papers} != set(scope):
                raise ValueError('Requested papers are not registered in the source namespace')
        selected = {p['sha256'] for p in papers}
        units = [dict(r) for r in c.execute('SELECT c.*,p.sha256 FROM chunks c JOIN papers p ON p.row_id=c.paper_row_id WHERE c.namespace=? AND p.active=1 ORDER BY c.row_id', (namespace,)) if r['sha256'] in selected]
    documents = [{'sha256': p['sha256'], 'file_name': p['file_name'], 'title': p['title'],
                  'authors': json.loads(p['authors_json'])} for p in papers]
    blocks = [{'id': u['stable_id'], 'paper_sha256': u['sha256'], 'text': u['text'], 'section': u['section'] or '',
        'source': {'source_uri': f"paper://{u['sha256']}/{u['stable_id']}",
            'page_range': [u['page_start'], u['page_end']], 'quality': 'IMPORTED_TEXT_WITH_PAGE_LOCATION'}} for u in units]
    return documents, blocks


def local_models(settings, *, lexical_only=False):
    cache = Path(os.environ.get('PAPERPILOT_EMBEDDING_CACHE', settings.workspace / 'model-cache/embedding'))
    paths = list(cache.rglob('tokenizer.json'))
    if len(paths) != 1:
        raise ValueError('Configured embedding cache must contain exactly one tokenizer snapshot')
    budget = TokenBudget.from_directory(paths[0].parent)
    embedder = None if lexical_only else UncachedEmbedder(cache_dir=cache, allow_download=False)
    return budget, embedder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-db', required=True, type=Path)
    p.add_argument('--namespace', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--paper-id', action='append')
    p.add_argument('--paper-root', action='append', type=Path, default=[])
    p.add_argument('--observations', type=Path, help='Frozen SHA -> source-bearing metadata observations')
    p.add_argument('--frontmatter', action='store_true', help='Read hash-checked first pages for literal identifiers')
    p.add_argument('--structured', action='store_true')
    p.add_argument('--docling-python', type=Path)
    p.add_argument('--docling-models', type=Path)
    p.add_argument('--no-cell-matching', action='store_true', help='Separate document-parser experiment; never changes the existing region parser')
    p.add_argument('--lexical-only', action='store_true')
    args = p.parse_args()
    settings = RuntimeSettings.from_env()
    documents, units = frozen_input(args.source_db, args.namespace, args.paper_id)
    extra = json.loads(args.observations.read_bytes()) if args.observations else {}
    input_hash = digest(encoded([documents, units, extra]))
    budget, embedder = local_models(settings, lexical_only=args.lexical_only)
    if args.frontmatter or args.structured:
        # Use the existing SHA-checking resolver, but never open its source DB for writing.
        store = type('Catalog', (), {'paper_metadata': lambda self, scope: [
            {'sha256': d['sha256'], 'file_name': d['file_name'], 'title': d['title']} for d in documents if d['sha256'] in scope]})()
        resolver = PaperPilotKnowledgeSearchAdapter(store, paper_roots=args.paper_root)
        paths = resolver.resolve_pdf_paths({d['sha256'] for d in documents})
        docling = None
        if args.structured and args.docling_python and args.docling_models:
            from paperpilot_agentic.docling_adapter import DoclingEvidenceAdapter
            docling = DoclingEvidenceAdapter(args.docling_python, args.docling_models, args.output.parent / (args.output.name + '-parse-cache'))
        parsed = []
        statuses = []
        for d in documents:
            path = paths.get(d['sha256'])
            if not path:
                statuses.append({'paper_id': d['sha256'], 'status': 'PDF_UNRESOLVED_OR_HASH_MISMATCH'})
                if args.structured:
                    raise ValueError('Structured ingestion requires every selected registered PDF')
                continue
            if args.structured:
                from paperpilot_agentic.paper_document import parse_document
                result = parse_document(path, d['sha256'], args.output.parent / (args.output.name + '-parsed') / d['sha256'],
                                        docling=docling, cell_matching=not args.no_cell_matching)
                parsed += result['units']
                extra.setdefault(d['sha256'], []).extend(result['metadata_observations'])
                statuses.append({'paper_id': d['sha256'], 'status': result['status'], 'fallback_reason': result['fallback_reason']})
            else:
                from pypdf import PdfReader
                text = PdfReader(path).pages[0].extract_text() or ''
                extra.setdefault(d['sha256'], []).extend(frontmatter_observations(text, d['sha256']))
                statuses.append({'paper_id': d['sha256'], 'status': 'FIRST_PAGE_IDENTIFIERS_ONLY'})
        if args.structured:
            units = parsed
    else:
        statuses = []
    source = {'namespace': args.namespace, 'input_snapshot_sha256': input_hash,
        'units_sha256': digest(encoded(units)), 'metadata_sha256': digest(encoded(extra)),
        'parse_statuses': statuses, 'builder_sha256': digest(Path(__file__).read_bytes())}
    manifest = build_index(documents, units, args.output, budget, embedder=embedder,
                          extra_observations=extra, source_identity=source)
    save_json(args.output / 'source-freeze.json', {str(f.relative_to(Path(__file__).resolve().parents[1])): digest(f.read_bytes())
        for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')})
    print(json.dumps({k: manifest[k] for k in ('version', 'papers', 'units', 'windows', 'reports', 'default_promoted')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
