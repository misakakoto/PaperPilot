"""The model-visible grouping description must match the declared PNG encoding."""
import json

from paperpilot_agentic.experiment_import import register_experiment
from paperpilot_agentic.research_diagnostics import load_bundle
from test_experiment_import import pack


def test_positive_one_encoding_is_described_as_one_not_255(tmp_path):
    runtime, path, value = pack(tmp_path)
    from PIL import Image
    import numpy as np
    for name in ('label', 'baseline', 'candidate'):
        image = np.asarray(Image.open(path.parent/(name+'.png'))) // 255
        Image.fromarray(image.astype(np.uint8)).save(path.parent/(name+'.png'))
    value['encoding']['positive'] = 1
    path.write_text(json.dumps(value), encoding='utf-8')
    bundle = load_bundle(runtime, register_experiment(runtime, path))
    assert 'equal to 1 in each image' in bundle['view_contracts']['coverage']['grouping_basis']
    # A retained old asset must not silently reuse the previously wrong contract.
    from paperpilot_agentic.reference_methods import encoded, sha
    from paperpilot_agentic.research_assets import immutable, owned_folder
    import pytest
    bundle['view_contracts']['coverage']['grouping_basis'] = 'Fraction of ground-truth label pixels equal to 255 in each image'
    identity = sha(encoded(bundle))
    immutable(owned_folder(runtime, 'diagnostic-inputs', identity), 'bundle.json', encoded(bundle))
    with pytest.raises(ValueError, match='re-register'):
        load_bundle(runtime, {'identity': identity, 'bundle_sha256': identity})
