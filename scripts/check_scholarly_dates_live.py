"""One bounded public scholarly search probe, no model calls or candidate execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = Path(__file__).resolve().parents[1]/'src/paperpilot_agentic/scholarly.py'
    frozen = out/'scholarly.py'
    frozen.write_bytes(original.read_bytes())
    request = {'query': 'building change detection', 'top_k': 3, 'published_from': '2025-01-01',
               'published_to': '2026-10-24', 'sort': 'newest', 'require_full_text': True}
    (out/'scope.json').write_text(json.dumps({'request': request, 'execute': args.execute,
        'source_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'scope': 'Public retrieval-tool probe; no model query selection or exhaustive literature review',
        'model_calls': 0, 'candidate_execution': False}, indent=2), encoding='utf-8')
    if not args.execute:
        return
    spec = importlib.util.spec_from_file_location('dated_scholarly_probe', frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.ScholarlySearchAdapter(out/'cache').search(**request)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    accepted = result['items']
    assertions = {'nonempty': bool(accepted), 'dates_within_scope': all(
        request['published_from'] <= r['publication_date'] <= request['published_to'] for r in accepted),
        'supported_pdf_urls': all(r.get('full_text_url', '').startswith('https://arxiv.org/pdf/') for r in accepted),
        'topic_relevance': 'NOT_AUTOMATICALLY_VERIFIED', 'full_text_download': 'NOT_RUN',
        'providers': result['providers'], 'errors': result['errors'],
        'dates': [r.get('publication_date') for r in accepted]}
    (out/'host-check.json').write_text(json.dumps(assertions, indent=2), encoding='utf-8')
    print(json.dumps(assertions), flush=True)


if __name__ == '__main__':
    main()
