"""Resolve only explicitly observed DOIs through Crossref into a frozen observations file.

No fuzzy title matching or overwrite of source records. Exact DOI responses,
including failures, persist before derived observations are written.
"""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from paperpilot_agentic.paper_catalog import crossref_observations, normalize_doi
from paperpilot_agentic.paper_structure import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(args.observations.read_bytes())
    import httpx
    with httpx.Client(timeout=20, trust_env=False) as client:
        for sha, observations in rows.items():
            dois = {normalize_doi(o['value']) for o in observations if o['field'] == 'doi'} - {None}
            if len(dois) != 1:
                save_json(args.output / (sha + '.json'), {'status': 'SKIPPED_MISSING_OR_CONFLICTING_DOI'})
                continue
            doi = next(iter(dois))
            try:
                response = client.get('https://api.crossref.org/works/' + quote(doi, safe=''))
                receipt = {'doi': doi, 'status_code': response.status_code, 'response': response.json()}
                save_json(args.output / (sha + '.json'), receipt)
                response.raise_for_status()
                observations.extend(crossref_observations(receipt['response']['message'], doi))
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                # Public DOI data only. A failure is retained and not retried.
                save_json(args.output / (sha + '-failure.json'), {'doi': doi, 'error': str(exc), 'retries': 0})
    save_json(args.output / 'observations.json', rows)


if __name__ == '__main__':
    main()
