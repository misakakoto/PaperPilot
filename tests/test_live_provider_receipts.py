import json

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from scripts.check_session_entry_live import RecordedProvider


@pytest.mark.parametrize('quota', [True, False])
def test_live_failure_retains_safe_category_without_secret_body_or_retry(tmp_path, quota):
    request = httpx.Request('POST', 'https://example.invalid', headers={'Authorization': 'SECRET'})
    error = (APIStatusError('SECRET', response=httpx.Response(403, request=request),
        body={'error': {'code': 'AllocationQuota.FreeTierOnly', 'message': 'SECRET'}})
        if quota else APIConnectionError(message='SECRET', request=request))
    class Provider:
        model = 'fixed-model'
        def complete(self, messages, **kwargs):
            kwargs['before_attempt']()
            raise error
    provider = RecordedProvider(Provider(), tmp_path, max_requests=1)
    with pytest.raises(type(error)):
        provider.complete([{'role': 'user', 'content': 'Synthetic'}], before_attempt=lambda: None)
    receipt = json.loads((tmp_path / 'failure-001.json').read_text())
    assert receipt['error_category'] == ('BILLING_OR_QUOTA' if quota else 'CONNECTION_ERROR')
    assert receipt['requested_model'] == 'fixed-model'
    assert receipt['http_status'] == (403 if quota else None)
    assert provider.calls == 1
    assert 'SECRET' not in ''.join(p.read_text() for p in tmp_path.iterdir())
