"""Read-only SYSU-CD train/validation preflight. Never opens the test split."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def inspect_dataset(root):
    root = Path(root).resolve(strict=True)
    result = {'dataset': 'SYSU-CD', 'root': str(root), 'splits': {},
              'test_split_opened': False,
              'fingerprint_scope': 'Split lists and local paths only; not image contents'}
    seen = set()
    for split in ('train', 'val'):
        list_path = root / f'{split}.txt'
        if not list_path.resolve().is_relative_to(root):
            raise ValueError('Split list escapes dataset root')
        raw = list_path.read_bytes()
        records = []
        local_seen = set()
        roles = ('time1', 'time2', 'label')
        inventories = {}
        for role in roles:
            folder = root / split / role
            if not folder.resolve().is_relative_to(root):
                raise ValueError('Image directory escapes dataset root')
            # Enumerate each directory once; avoid random filesystem calls per list row.
            with os.scandir(folder) as entries:
                inventories[role] = {entry.name for entry in entries
                                     if not entry.is_symlink() and entry.is_file(follow_symlinks=False)}
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            paths = [PurePosixPath(value.replace('\\', '/')) for value in line.split()]
            if len(paths) != 3 or len({p.name for p in paths}) != 1:
                raise ValueError(f'{split}: expected three matching image names')
            name = paths[0].name
            if name in local_seen or name in seen:
                raise ValueError(f'Duplicate or overlapping sample ID: {name}')
            record = {'sample_id': name}
            for path, role in zip(paths, roles):
                if ('..' in path.parts or len(path.parts) < 3 or
                        path.parts[-3:-1] != (split, role)):
                    raise ValueError(f'{split}: wrong split or image role')
                local = root / split / role / name
                if name not in inventories[role]:
                    raise ValueError(f'Missing file or path outside dataset: {split}/{role}/{name}')
                record[role] = str(local)
            local_seen.add(name)
            records.append(record)
        if not records:
            raise ValueError(f'Empty split: {split}')
        seen.update(local_seen)
        result['splits'][split] = {'count': len(records),
                                   'list_sha256': hashlib.sha256(raw).hexdigest(),
                                   'samples': records}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON manifest outside the original dataset')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error('Output must not modify the original dataset')
    result = inspect_dataset(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'counts': {key: value['count'] for key, value in result['splits'].items()},
                      'test_split_opened': False, 'manifest': str(args.output)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
