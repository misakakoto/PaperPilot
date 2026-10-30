"""Second, separately frozen experiment: Docling originals and experiment objects.

Reuses a verified saved raw conversion when explicitly supplied; never rewrites
its receipts. Cell matching is disabled only for this document experiment.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paperpilot_agentic.config import RuntimeSettings
from paperpilot_agentic.docling_adapter import DoclingEvidenceAdapter
from paperpilot_agentic.paper_catalog import frontmatter_observations
from paperpilot_agentic.paper_document import parse_document, document_units
from paperpilot_agentic.paper_evidence_index import build_index
from paperpilot_agentic.paper_tables import literal_reports
from paperpilot_agentic.paper_structure import digest, save_json
from build_paper_evidence import local_models, frozen_input


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation', required=True, type=Path)
    p.add_argument('--source-db', required=True, type=Path)
    p.add_argument('--namespace', default='all-papers-v1')
    p.add_argument('--paper-root', required=True, type=Path)
    p.add_argument('--docling-python', required=True, type=Path)
    p.add_argument('--docling-models', required=True, type=Path)
    p.add_argument('--reuse-raw', type=Path, help='An existing structured parsed.json plus its frozen raw.json')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    freeze = {str(f): digest(f.read_bytes()) for f in Path('src').rglob('*.py')}
    save_json(args.output / 'source-freeze.json', freeze)
    sources = json.loads((args.preparation / 'sources.json').read_bytes())
    observations = json.loads((args.preparation / 'observations-frozen.json').read_bytes())
    papers, _ = frozen_input(args.source_db, args.namespace, [p['sha256'] for p in sources])
    adapter = DoclingEvidenceAdapter(args.docling_python, args.docling_models, args.output / 'cache')
    reused = json.loads(args.reuse_raw.read_bytes()) if args.reuse_raw else None
    units, receipts = [], []
    for paper in sources:
        sha = paper['sha256']
        pdf = next((f for f in args.paper_root.rglob(paper['file_name']) if digest(f.read_bytes()) == sha), None)
        if pdf is None:
            raise ValueError('Cannot resolve registered PDF version')
        if reused is not None and reused['paper_sha256'] == sha:
            raw_path = args.reuse_raw.with_name('raw.json')
            raw = json.loads(raw_path.read_bytes())
            old_identity = reused['units'][0]['source']['parser']['identity']
            if (digest(raw_path.read_bytes()) != reused['raw_sha256'] or old_identity['cell_matching'] is not False
                    or old_identity['weights_sha256'] != adapter.weights_sha
                    or old_identity['worker_sha256'] != adapter.identity['worker_sha256']):
                raise ValueError('Raw conversion identity changed')
            parsed = document_units(raw, sha)
            for u in parsed:
                u['source']['parser'] = {'identity': old_identity, 'raw_sha256': reused['raw_sha256'], 'raw_receipt': str(raw_path.resolve())}
                if u['kind'] == 'TABLE':
                    u['reports'] = literal_reports(u)
            receipt = {'status': 'REPROJECTED_VERIFIED_RAW', 'raw_sha256': reused['raw_sha256'], 'units': parsed}
            folder = args.output / 'parsed' / sha
            folder.mkdir(parents=True)
            save_json(folder / 'parsed.json', receipt)
        else:
            receipt = parse_document(pdf, sha, args.output / 'parsed' / sha, docling=adapter, cell_matching=False)
        # Metadata comes from the pre-frozen original first-page text, independent of this parser experiment.
        first_page = json.loads((args.preparation / (sha + '.json')).read_bytes())[0]['text']
        observations.setdefault(sha, []).extend(frontmatter_observations(first_page, sha))
        units.extend(receipt['units'])
        summary = {'paper_id': sha, 'status': receipt['status'], 'fallback_reason': receipt.get('fallback_reason'),
            'units': len(receipt['units']), 'tables': [{'status':u['object']['structure_status'],
                'errors':u['object']['structure_errors'], 'reports':len(u.get('reports',[]))}
                for u in receipt['units'] if u['kind'] == 'TABLE']}
        receipts.append(summary)
        save_json(args.output / 'parse-progress.json', receipts)
        print(json.dumps(summary), flush=True)
    budget, embedder = local_models(RuntimeSettings.from_env())
    manifest = build_index(papers, units, args.output / 'paper-evidence-v2', budget, embedder=embedder,
        extra_observations=observations, source_identity={'parse_receipts': receipts, 'cell_matching': False,
            'purpose': 'separate structure/numerical experiment; default corpus unchanged'})
    assert freeze == {str(f): digest(f.read_bytes()) for f in Path('src').rglob('*.py')}
    print(json.dumps({k:manifest[k] for k in ['papers','units','windows','reports']}), flush=True)


if __name__ == '__main__':
    main()
