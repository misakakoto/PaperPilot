"""Offline geometry diagnostic on unchanged persisted tool output; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from paperpilot_agentic.qwen_vision import QwenTableRow, _audit_indicator_pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding='utf-8'))
    for call in result.get('tool_calls', []):
        for item in (call.get('result') or {}).get('evidence', []):
            metadata = (item.get('provenance') or {}).get('vision_metadata') or {}
            if not metadata.get('table_rows'):
                continue
            path = Path(item['artifact_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['image_sha256']:
                raise ValueError('Saved crop hash mismatch')
            rows = [QwenTableRow.model_validate(row) for row in metadata['table_rows']]
            valid, reason, _, details = _audit_indicator_pixels(path, rows,
                column_headers=metadata.get('column_headers'))
            print(json.dumps({'source_uri': item['source_uri'], 'geometry_valid': valid,
                              'reason': reason, 'details': details}, ensure_ascii=False))


if __name__ == '__main__':
    main()
