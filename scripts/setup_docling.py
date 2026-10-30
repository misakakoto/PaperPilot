"""Explicit isolated installation/model provisioning; never uploads documents."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def freeze_models(folder):
    files = []
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or '.cache' in path.parts or path.name == 'weights-manifest.json':
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        files.append({'path': path.relative_to(folder).as_posix(), 'bytes': path.stat().st_size,
                      'sha256': digest.hexdigest()})
    manifest = {'docling_version': importlib.metadata.version('docling'), 'files': files,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['docling', 'torch', 'torchvision', 'transformers', 'docling-ibm-models', 'rapidocr']}}
    (folder / 'weights-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(files), 'downloaded_bytes': sum(f['bytes'] for f in files)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--download-ocr', action='store_true')
    args = parser.parse_args()
    env = ROOT / 'outputs/docling-env'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if args.install:
        subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.14.1+cpu',
                        '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'docling==2.132.0'], check=True)
    if args.download_models or args.download_ocr:
        from docling.utils.model_downloader import download_models
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        folder = ROOT / 'outputs/docling-models'
        download_models(folder, with_picture_classifier=False, with_rapidocr=True,
                        rapidocr_models=['torch:en'], with_tableformer=args.download_models,
                        with_layout=args.download_models, with_code_formula=args.download_models)
        freeze_models(folder)
    if python.is_file():
        frozen = subprocess.run([str(python), '-m', 'pip', 'freeze'], check=True,
                                capture_output=True, text=True).stdout
        (ROOT / 'outputs/docling-dependencies.lock.txt').write_text(frozen, encoding='utf-8')
    print(json.dumps({'python': str(python), 'ready': (ROOT / 'outputs/docling-models/weights-manifest.json').is_file()}))


if __name__ == '__main__':
    main()
