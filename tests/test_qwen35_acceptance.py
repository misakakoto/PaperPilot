from types import SimpleNamespace

import pytest

from evals.interview.budget import CampaignBudget
from evals.interview.run import validate_supplemental_model
from paperpilot_agentic.model_policy import DAILY_MODEL, PRIMARY_MODEL


def test_changed_snapshot_requires_explicit_authorization():
    settings = SimpleNamespace(model=DAILY_MODEL, provider='qwen')
    parent = {'identity': {'model': PRIMARY_MODEL, 'provider': 'qwen'}}
    with pytest.raises(ValueError, match='retains'):
        validate_supplemental_model(settings, parent, None, 'continue')
    with pytest.raises(ValueError, match='explicit'):
        validate_supplemental_model(settings, parent, DAILY_MODEL, None)
    validate_supplemental_model(settings, parent, DAILY_MODEL, 'User authorized 30/300000 on new model')
    assert parent['identity']['model'] == PRIMARY_MODEL


def test_snapshot_policy_preserves_full_image_bound_and_frozen_identity(tmp_path):
    identity = {'model': DAILY_MODEL, 'provider': 'qwen', 'request_cost_policy': 'qwen35-still-image-v1'}
    budget = CampaignBudget(tmp_path, identity, max_requests=30, max_tokens=300000)
    request = {'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fixture'}},
        {'type': 'text', 'text': 'Read source.'}]}], 'max_tokens': 128}
    assert budget.estimated_request_bound(request) >= 16384 + 128
    assert budget.snapshot()['actual_requests'] == 0
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, {**identity, 'model': PRIMARY_MODEL}, max_requests=30, max_tokens=300000)
    with pytest.raises(ValueError, match='ceiling changed'):
        CampaignBudget(tmp_path, identity, max_requests=30, max_tokens=600000)
