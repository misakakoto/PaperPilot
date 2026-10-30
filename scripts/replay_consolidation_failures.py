"""Replay a saved crop and answer through offline gates; never regenerate an answer."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT))
from evals.structured_pdf import save_new  # noqa: E402
from paperpilot_agentic.answer_audit import validate_draft  # noqa: E402
from paperpilot_agentic.experimental_scope import apply_experimental_scope, source_roles  # noqa: E402
from paperpilot_agentic.qwen_vision import reaudit_qwen_result  # noqa: E402
from paperpilot_agentic.structured_extraction import SourceRegion, annotate_observation, file_sha  # noqa: E402


def replay(path):
    before = file_sha(path)
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    observations, rows = [], []
    for old in snapshot['evidence']:
        provenance = old.get('provenance') or {}
        metadata = provenance.get('vision_metadata') or {}
        if not metadata:
            continue
        image = Path(old['artifact_path'])
        manifest = json.loads((image.parent.parent / 'layout_manifest.json').read_text(encoding='utf-8'))
        objects = [o for o in manifest['objects'] if o['object_id'] == provenance['layout_object_id']]
        if len(objects) != 1:
            raise ValueError('Historical source does not bind to one layout object')
        obj = objects[0]
        visual = reaudit_qwen_result({'description': old['content'], 'metadata': metadata,
            'verified': provenance['vision_verified'], 'confidence': old['confidence']}, image,
            prompt=snapshot['run']['question'], context={'object_type': obj['object_type'],
                'caption': obj.get('caption') or obj.get('text')})
        label = provenance['layout_target_binding']['label']
        source = SourceRegion(paper_sha256=provenance['document_sha256'], object_id=obj['object_id'],
            label=f'Table {label[-1]}', page=obj['page'],
            bbox=obj['metadata']['render_context_bbox'], crop_sha256=file_sha(image))
        current = copy.deepcopy(old)
        current['provenance'].update(vision_metadata=visual['metadata'], vision_verified=visual['verified'])
        annotate_observation(current, region=source, parser='QwenVisionEvidenceAdapter')
        observations.append(current)
        rows.append({'paper_sha256': source.paper_sha256, 'label': source.label, 'page': source.page,
            'crop_sha256': source.crop_sha256, 'source_and_structure_verified': visual['verified'],
            'factual_review': 'NOT_REVIEWED', 'provider_attempts_added': 0,
            'table_rows': visual['metadata'].get('audited_table_rows')})
    saved = snapshot['run']['metadata']['answer_first_checkpoint']['report']
    try:
        validate_draft(copy.deepcopy(saved['draft']), snapshot['evidence'], question=snapshot['run']['question'])
        contract = {'status': 'ACCEPTED_BY_CONTRACT_NOT_FACTUAL_REVIEW'}
    except (ValueError, TypeError) as exc:
        contract = {'status': 'REJECTED', 'reason': str(exc)}
    _, scope = apply_experimental_scope(saved['draft'], saved['model_audit'], source_roles(observations))
    assert before == file_sha(path)
    return {'kind': 'OFFLINE_GATE_REPLAY_NOT_NEW_ANSWER', 'snapshot_sha256': before,
        'historical_snapshot_unchanged': True, 'provider_attempts_added': 0, 'saved_crop_reaudits': rows,
        'current_writer_contract': contract, 'experimental_scope': scope,
        'paired_baseline': 'Different crop from the new 20-region set; qualitative same-source replay only',
        'new_live_demo_passes': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'outputs/interview-acceptance-20261001/revisions/0008/single_changemamba_detail-result.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/consolidation-saved-qwen-replay.json')
    args = parser.parse_args()
    result = replay(args.snapshot)
    save_new(args.output, result)
    print(json.dumps({'output': str(args.output), 'crop_count': len(result['saved_crop_reaudits']),
        'writer_contract': result['current_writer_contract'], 'scope': result['experimental_scope'],
        'provider_attempts_added': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
