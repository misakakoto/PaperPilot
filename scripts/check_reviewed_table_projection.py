"""Project selected cells from an existing reviewed asset, without an LLM call.

This is a provenance/contract check, not a new parser score or real QA acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.reviewed_sources import ReviewedSources
from paperpilot_agentic.structured_extraction import ParsedArtifact
from paperpilot_agentic.table_claims import row_catalog
from paperpilot_agentic.table_projection import project_table_facts


def check(asset_path, row_index, metrics):
    payload = json.loads(asset_path.read_text(encoding='utf-8'))
    artifact = ParsedArtifact.model_validate(payload['artifact'])
    registry = ReviewedSources(asset_path.parent)
    source = {'evidence_id': 'owned_table',
              'source_uri': f'paper://{artifact.source.paper_sha256}/layout/{artifact.source.object_id}',
              'content': 'Pending registry read', 'provenance': {}}
    assert registry.apply(source, artifact.source), 'Asset identity or review failed current registry checks'
    source['provenance']['structured_extraction'] = artifact.model_dump(mode='json')
    catalog = row_catalog([source])
    row = next(row for row in catalog if row['row_index'] == row_index)
    selection = {'id': 'selected', 'kind': 'TABLE_FACT', 'operation': 'value',
                 'row_refs': [{'evidence_id': row['evidence_id'], 'row_id': row['row_id'], 'metrics': metrics}]}
    projected, trace = project_table_facts({'facts': [selection]}, catalog, chinese=True)
    for metric in metrics:
        assert row['cells'][metric] in projected['facts'][0]['text']
    forbidden = dict(selection, text='Unowned model value')
    try:
        project_table_facts({'facts': [forbidden]}, catalog, chinese=True)
    except ValueError:
        unowned_value_rejected = True
    else:
        raise AssertionError('Extra model value was accepted')
    return {'mode': 'OFFLINE_REVIEWED_SOURCE_PROJECTION', 'actual_provider_calls': 0,
            'reviewed_asset_sha256': hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            'source': artifact.source.model_dump(mode='json'), 'parser': artifact.parser,
            'parser_version': artifact.parser_version, 'config_sha256': artifact.config_sha256,
            'existing_reviewer_kind': payload['review']['reviewer_kind'],
            'selection': selection, 'source_row': row, 'projection_trace': trace,
            'unowned_value_rejected': unowned_value_rejected, 'qa_acceptance': 'NOT_RUN',
            'limitations': ['Uses an existing reviewed transcription, not new automatic recognition.',
                            'No model-generated answer or independent new original-image review.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-asset', type=Path, required=True)
    parser.add_argument('--row-index', type=int, required=True)
    parser.add_argument('--metric', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.reviewed_asset.resolve(), args.row_index, args.metric)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous report; use a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(output), 'actual_provider_calls': 0,
                      'unowned_value_rejected': report['unowned_value_rejected'],
                      'answer': report['projection_trace'][0]['projected_fact']['text']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
