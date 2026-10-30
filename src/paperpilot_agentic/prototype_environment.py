"""Read-only host preflight and immutable CPU execution identity."""
import os
import re
import subprocess

from .models import SandboxPolicy

IMAGE = 'paperpilot-prototype:cpu'
_IMAGE_ID = re.compile(r'sha256:[a-f0-9]{64}')


def preflight_environment():
    try:
        result = subprocess.run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}|{{.Os}}|{{.Architecture}}'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except (OSError, subprocess.TimeoutExpired):
        return {'available': False, 'reason': 'DOCKER_UNAVAILABLE', 'image_reference': IMAGE}
    if result.returncode:
        return {'available': False, 'reason': 'DOCKER_OR_IMAGE_UNAVAILABLE', 'image_reference': IMAGE}
    parts = result.stdout.strip().split('|')
    if (len(parts) != 3 or not _IMAGE_ID.fullmatch(parts[0]) or parts[1] != 'linux'
            or parts[2] not in {'amd64', 'arm64'}):
        return {'available': False, 'reason': 'INVALID_CPU_IMAGE_IDENTITY', 'image_reference': IMAGE}
    return {'available': True, 'image_id': parts[0], 'platform': parts[1]+'/'+parts[2], 'image_reference': IMAGE}


def freeze_environment(report, *, filesystem='readonly-files-v1'):
    if (report.get('available') is not True or not _IMAGE_ID.fullmatch(report.get('image_id', ''))
            or report.get('platform') not in {'linux/amd64', 'linux/arm64'}):
        raise ValueError('A verified immutable Docker image is required')
    if filesystem not in {'readonly-files-v1', 'legacy-writable-v1'}:
        raise ValueError('Unknown prototype filesystem policy')
    policy = SandboxPolicy(backend='docker', docker_image=report['image_id'], timeout_seconds=180,
                           cpu_seconds=360, memory_mb=4096, network='none', allow_local_fallback=False,
                           read_only_files=('verify.py', 'contract.json', 'config.json', 'model.py')
                           if filesystem == 'readonly-files-v1' else ())
    result = {'backend': 'docker', 'image_id': report['image_id'], 'platform': report['platform'],
            'network': 'none', 'cpu_cores': 2, 'memory_mb': 4096, 'timeout_seconds': 180,
            'policy': policy.to_dict(), 'command': ['python', 'verify.py'], 'metric_files': ['metrics.json']}
    if filesystem == 'readonly-files-v1':
        result['filesystem'] = filesystem
    return result
