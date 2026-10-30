"""Host registration of pinned baselines for bounded prototype editing."""
import json

from .reference_methods import SOURCE_SHA, LICENSE_SHA, PIN, sha, encoded
from .research_assets import owned_folder, immutable, build_fusion_patch


def register_snunet_profile(runtime, source, license_text):
    if sha(source) != SOURCE_SHA or sha(license_text) != LICENSE_SHA:
        raise ValueError('SNUNet source/license must match the inspected pinned revision')
    manifest = {'version': 'prototype-profile-v1', 'kind': 'snunet_ecam',
                'source_sha256': SOURCE_SHA, 'license_sha256': LICENSE_SHA,
                'source_url': PIN+'models/Models.py', 'license_url': PIN+'LICENSE',
                'output_contract': 'tuple-one/Bx2xHxW/CE',
                'editable_region': 'SNUNet_ECAM.forward attention/fusion tail only'}
    identity = sha(encoded(manifest))
    folder = owned_folder(runtime, 'prototype-profiles', identity)
    immutable(folder, 'baseline.py', source)
    immutable(folder, 'LICENSE', license_text)
    immutable(folder, 'profile.json', encoded(manifest))
    return identity


def load_profile(runtime, identity):
    folder = owned_folder(runtime, 'prototype-profiles', identity)
    manifest = json.loads((folder/'profile.json').read_bytes())
    if (sha(encoded(manifest)) != identity or manifest.get('version') != 'prototype-profile-v1'
            or manifest.get('kind') != 'snunet_ecam'
            or sha((folder/'baseline.py').read_bytes()) != manifest['source_sha256']
            or sha((folder/'LICENSE').read_bytes()) != manifest['license_sha256']
            or manifest['source_sha256'] != SOURCE_SHA or manifest['license_sha256'] != LICENSE_SHA):
        raise ValueError('Prototype baseline profile changed')
    return manifest, folder


def render_snunet(runtime, identity, proposal):
    manifest, folder = load_profile(runtime, identity)
    if not isinstance(proposal, dict) or set(proposal) != {'fusion_body'}:
        raise ValueError('Bound SNUNet implementation expects only fusion_body')
    patch = build_fusion_patch((folder/'baseline.py').read_text(encoding='utf-8'), proposal['fusion_body'])
    return {'source': patch['candidate_source'], 'config': {},
            'profile_id': identity, 'baseline_sha256': manifest['source_sha256'],
            'diff': patch['diff'], 'static_checks': patch['static_checks'],
            'fusion_body': patch['fusion_body'], 'expected_unused': patch['expected_unused']}
