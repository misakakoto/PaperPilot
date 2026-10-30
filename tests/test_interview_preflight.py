"""Unavailable models stop before expensive source loading; probes are counted."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.interview.budget import CampaignBudget, RecordedProvider  # noqa: E402
from evals.interview.run import model_preflight  # noqa: E402
from test_interview_campaign import Provider
from paperpilot_agentic.providers import LLMResponse


def test_failed_probe_is_counted_and_replay_does_not_send_again(tmp_path):
    budget = CampaignBudget(tmp_path/'receipts', {'model': 'fixed'})
    provider = RecordedProvider(Provider(OSError('offline')), budget, phase='preflight')
    assert not model_preflight(provider, tmp_path)
    assert budget.snapshot()['actual_requests'] == 1 and budget.snapshot()['blocked']
    assert not model_preflight(provider, tmp_path)
    assert provider.calls == 1


def test_successful_probe_consumes_the_same_allowance_as_answers(tmp_path):
    budget = CampaignBudget(tmp_path/'receipts', {'model': 'fixed'}, max_requests=1)
    provider = RecordedProvider(Provider(LLMResponse('ready', 10, 1, finish_reason='stop')), budget, phase='preflight')
    assert model_preflight(provider, tmp_path)
    assert budget.snapshot()['actual_requests'] == 1
    assert budget.snapshot()['reported_tokens'] == 11
