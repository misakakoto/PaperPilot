"""Loaded-code identity, separate from the possibly newer files on disk."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source_identity():
    digest = hashlib.sha256()
    for folder in (ROOT / 'src/paperpilot_agentic', ROOT / 'src/paperpilot_retrieval'):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.js', '.css', '.html'}:
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT, stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = 'unversioned'
    return {'revision': revision, 'source_digest': digest.hexdigest()[:16]}


LOADED = source_identity()


def build_status():
    current = source_identity()
    return {'loaded': LOADED, 'disk': current,
            'restart_required': current['source_digest'] != LOADED['source_digest']}
