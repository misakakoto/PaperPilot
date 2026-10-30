"""Zero-call regression of the retained evidence-selection/projection failures."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from paperpilot_agentic.read_packet import build_read_packet
from paperpilot_agentic.table_projection import project_table_facts_locally
from evals.interview.budget import save


def replay(root, output):
    old = root/'outputs/mcp-comparison-20261017'
    report = {'api_calls': 0, 'old_results_modified': False, 'source_precision': [], 'table_projection': []}
    for name in ('followup_alignment_inference', 'unseen-alignment_loss',
                 'unseen-deployment_limits', 'unseen-robustness_conditions'):
        result = json.loads((old/'round2/mcp'/(name+'-result.json')).read_bytes())
        rows = deepcopy(result['evidence'])
        db = sqlite3.connect((old/'round2/mcp/corpus.sqlite3').as_uri()+'?mode=ro', uri=True)
        for row in rows:
            uri = row['source_uri']
            if not uri.startswith('paper://') or '/layout/' in uri:
                continue
            cid = uri.rsplit('/', 1)[-1]
            location = db.execute('SELECT page_start,page_end FROM chunks WHERE stable_id=?', (cid,)).fetchone()
            if location:
                row['page'] = location[0] if location[0] == location[1] else None
                row.setdefault('provenance', {}).update(page_precision='EXACT' if row['page'] else 'RANGE',
                                                       page_range=list(location))
        db.close()
        packet = build_read_packet(result['run']['question'], rows)
        before = result['run']['metadata']['reader_workset']
        report['source_precision'].append({'case': name, 'before_selected': before['selected_count'],
            'after_selected': packet.metadata['selected_count'],
            'unrepresented_sources': packet.metadata['unrepresented_sources'],
            'source_text_and_uris_unchanged': all(a['content'] == b['content'] and a['source_uri'] == b['source_uri']
                                               for a, b in zip(rows, result['evidence']))})
    for name in ('unseen-backbone_tradeoff', 'unseen-disaster_metric'):
        result = json.loads((old/'round2/local'/(name+'-result.json')).read_bytes())
        r = result['run']['metadata']['reader_report']
        projected, traces, errors = project_table_facts_locally(r['raw_draft'], r['table_rows'], chinese=True)
        report['table_projection'].append({'case': name,
            'previous_errors': r.get('table_projection_errors'), 'current_errors': errors,
            'retained_facts': projected['facts'], 'traces': traces})
    output.mkdir(parents=True, exist_ok=True)
    save(output/'zero-call-replay.json', report)
    print(json.dumps({'api_calls': 0, 'source_precision': report['source_precision'],
        'projection_errors': {r['case']: r['current_errors'] for r in report['table_projection']}}, ensure_ascii=True))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(Path(__file__).resolve().parents[2], args.output.resolve())
