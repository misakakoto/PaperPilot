"""Zero-model-call transport check in the same process as acceptance execution.

TCP reachability is not model availability or answer correctness. Failures never
reserve campaign tokens, settle existing UNKNOWN, or authorize a continuation.
"""
import hashlib
from pathlib import Path
from uuid import uuid4

from paperpilot_agentic.web_app import probe_provider_connectivity

from .budget import save


def transport_preflight(settings, folder):
    result = {
        **probe_provider_connectivity(settings.base_url, timeout_seconds=5),
        'base_url_sha256': hashlib.sha256(settings.base_url.encode()).hexdigest(),
        'actual_requests': 0,
        'billable_requests': 0,
        'model_availability': 'NOT_TESTED',
    }
    path = Path(folder) / ('transport-probe-' + uuid4().hex + '.json')
    save(path, result)
    return result
