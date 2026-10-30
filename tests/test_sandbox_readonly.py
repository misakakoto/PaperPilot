"""Protected verifier files must be OS-read-only while metrics remain writable."""
import os
from pathlib import Path

import pytest

from paperpilot_agentic.models import ExperimentProtocol, SandboxPolicy
from paperpilot_agentic.sandbox import SandboxExperimentRunner, _validate_protocol
from paperpilot_agentic.prototype_environment import freeze_environment, preflight_environment


def protocol(tmp_path, files=('verify.py',), **kwargs):
    return ExperimentProtocol(name='readonly-fixture', command=['python', 'probe.py'],
        source_dir=str(tmp_path/'source'), output_dir=str(tmp_path/'output'),
        policy=SandboxPolicy(backend='docker', read_only_files=files, **kwargs))


def test_readonly_policy_roundtrip_and_legacy_fingerprints(tmp_path):
    legacy = SandboxPolicy().to_dict()
    assert 'read_only_files' not in legacy
    assert SandboxPolicy.from_dict(legacy).to_dict() == legacy
    new = protocol(tmp_path).to_dict()
    assert new['policy']['read_only_files'] == ['verify.py']
    assert ExperimentProtocol.from_dict(new).to_dict() == new


@pytest.mark.parametrize('files', [('../host.py',), ('/tmp/verify.py',), ('C:/key',),
    ('folder\\verify.py',), ('./verify.py',), ('folder/../verify.py',), ('a,b.py',),
    ('verify.py', 'verify.py'), (True,), 'verify.py'])
def test_invalid_protection_paths_rejected_before_launch(tmp_path, files):
    with pytest.raises(ValueError, match='read.only'):
        _validate_protocol(protocol(tmp_path, files))


def test_readonly_protection_cannot_use_local_fallback(tmp_path):
    value = protocol(tmp_path)
    value.policy.backend = 'auto'
    with pytest.raises(ValueError, match='Docker'):
        _validate_protocol(value)


def test_missing_protected_file_does_not_start_container(tmp_path, monkeypatch):
    value = protocol(tmp_path)
    Path(value.source_dir).mkdir()
    monkeypatch.setattr('paperpilot_agentic.sandbox._docker_image_available', lambda _: True)
    monkeypatch.setattr('paperpilot_agentic.sandbox.subprocess.Popen',
                        lambda *a, **kw: pytest.fail('Missing verifier must not launch'))
    with pytest.raises(ValueError, match='read.only'):
        SandboxExperimentRunner().run(value)


def test_new_environment_freezes_readonly_files_and_legacy_is_explicit():
    report = {'available': True, 'image_id': 'sha256:'+'a'*64, 'platform': 'linux/amd64'}
    new = freeze_environment(report)
    assert new['filesystem'] == 'readonly-files-v1'
    assert set(new['policy']['read_only_files']) == {'model.py', 'verify.py', 'contract.json', 'config.json'}
    old = freeze_environment(report, filesystem='legacy-writable-v1')
    assert 'filesystem' not in old and 'read_only_files' not in old['policy']
    with pytest.raises(ValueError):
        freeze_environment(report, filesystem='unrecognized')


@pytest.mark.skipif(os.environ.get('PAPERPILOT_TEST_CONTAINER') != '1', reason='Explicit Docker opt-in required')
def test_real_docker_blocks_overwrite_and_unlink_but_allows_metrics(tmp_path):
    protected = ('verify.py', 'model.py', 'config.json', 'contract.json')
    source = tmp_path/'source'
    source.mkdir()
    for name in protected:
        (source/name).write_text('# maintained fixture\n', encoding='utf-8')
    (source/'probe.py').write_text('''from pathlib import Path
import json
results = {}
for name in ('verify.py','model.py','config.json','contract.json'):
    path = Path(name)
    original = path.read_bytes()
    attempts = []
    for operation in (lambda: path.write_text('tampered'), lambda: path.unlink()):
        try:
            operation()
        except OSError as exc:
            attempts.append(exc.errno)
        else:
            attempts.append(None)
    results[name] = {'blocked_errno': attempts, 'unchanged': path.read_bytes() == original}
Path('metrics.json').write_text(json.dumps({'protection': results, 'writable_metrics': True}))
''', encoding='utf-8')
    env = freeze_environment(preflight_environment())
    value = protocol(tmp_path, protected)
    value.policy = SandboxPolicy.from_dict(env['policy'])
    result = SandboxExperimentRunner().run(value)
    assert result.status == 'COMPLETED', result.to_dict()
    assert result.network_isolated and result.source_unchanged
    assert result.metrics['writable_metrics']
    assert all(row['unchanged'] and all(e in (16,30) for e in row['blocked_errno'])
               for row in result.metrics['protection'].values())
