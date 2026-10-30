import pytest

from paperpilot_agentic.research_assets import build_fusion_patch
from test_research_harness import SOURCE


def test_keepdim_boolean_is_supported_for_declared_tensor_reduction():
    result = build_fusion_patch(SOURCE, 'center = out.mean(dim=1, keepdim=True)\nout = out - center')
    assert 'keepdim=True' in result['candidate_source']


@pytest.mark.parametrize('body', ['out = out + True', 'out = out.mean(dim=True)', 'out = out + 128'])
def test_other_boolean_or_unbounded_constants_are_rejected(body):
    with pytest.raises(ValueError):
        build_fusion_patch(SOURCE, body)


def test_fusion_syntax_error_reports_location_as_validation_failure():
    with pytest.raises(ValueError, match='line 1'):
        build_fusion_patch(SOURCE, 'out = (')
