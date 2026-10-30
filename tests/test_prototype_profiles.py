from types import SimpleNamespace

import pytest

from paperpilot_agentic import prototype_profiles as profiles
from paperpilot_agentic.reference_methods import sha
from paperpilot_agentic.prototype_workspace import plan, complete
from test_research_harness import SOURCE, FUSION
from test_prototype_workspace import waiting, tasks


def register(runtime, monkeypatch):
    monkeypatch.setattr(profiles, 'SOURCE_SHA', sha(SOURCE.encode()))
    monkeypatch.setattr(profiles, 'LICENSE_SHA', sha(b'test license'))
    return profiles.register_snunet_profile(runtime, SOURCE.encode(), b'test license')


def test_profile_freezes_source_license_and_splices_only_allowed_region(tmp_path, monkeypatch):
    runtime = SimpleNamespace(workspace=tmp_path)
    identity = register(runtime, monkeypatch)
    manifest, folder = profiles.load_profile(runtime, identity)
    result = profiles.render_snunet(runtime, identity, {'fusion_body': FUSION})
    assert result['source'].endswith('        return (out,)\n')
    assert result['config'] == {} and result['baseline_sha256'] == sha(SOURCE.encode())
    assert (folder/'baseline.py').read_text() == SOURCE
    assert manifest['output_contract'] == 'tuple-one/Bx2xHxW/CE'
    (folder/'baseline.py').write_text(SOURCE+'# changed')
    with pytest.raises(ValueError, match='changed'):
        profiles.load_profile(runtime, identity)


def test_profile_rejects_other_sources_and_arbitrary_replacement(tmp_path, monkeypatch):
    runtime = SimpleNamespace(workspace=tmp_path)
    identity = register(runtime, monkeypatch)
    with pytest.raises(ValueError, match='pinned'):
        profiles.register_snunet_profile(runtime, b'other code', b'test license')
    with pytest.raises(ValueError, match='fusion_body'):
        profiles.render_snunet(runtime, identity, {'source': SOURCE})


def test_bound_task_renders_copy_and_binds_profile_to_approval(tmp_path, monkeypatch):
    runtime, harness, _ = waiting(tmp_path, monkeypatch)
    identity = register(runtime, monkeypatch)
    rid = harness.start('SNUNet local change', 'snunet-bound-task', 'parent', ['source'],
                        profile_id=identity)['run']['run_id']
    state = harness._load(rid)
    plan(state['work'], tasks())
    complete(state['work'], 'read', 'read', {'source': 'owned'})
    complete(state['work'], 'design', 'design', {'hypothesis': 'fixture'})
    harness._apply(rid, state, {'action': 'work', 'reason': 'fixture', 'arguments': {
        'task_id': 'code', 'content': {'fusion_body': FUSION}}}, 1)
    artifact = state['work']['artifacts'][state['work']['tasks']['code']['output']]
    assert artifact['content']['source'].startswith('class SNUNet_ECAM:')
    assert artifact['content']['profile_id'] == identity
    harness._apply(rid, state, {'action': 'work', 'reason': 'fixture', 'arguments': {
        'task_id': 'check', 'content': None}}, 2)
    approval = runtime.db.get_approval(state['pending']['approval'])
    assert approval['request']['arguments']['profile_id'] == identity
    assert approval['request']['context']['content']['diff']
