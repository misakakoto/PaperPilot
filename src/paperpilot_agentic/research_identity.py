"""Frozen model identity shared by research engines."""
from .reference_methods import encoded, sha


def research_model_identity(runtime):
    """Research freezes its actual model settings, not the unrelated QA retriever."""
    model = runtime.llm_runtime.model
    identity = getattr(model, 'policy_identity', None)
    return 'research-v1/'+sha(encoded({'model': getattr(model, 'model', None),
        'provider': getattr(model, 'provider', None), 'base_url': getattr(model, 'base_url', None),
        'temperature': getattr(model, 'temperature', None),
        'timeout': str(getattr(getattr(model, '_client', None), 'timeout', None)),
        'policy': identity('frozen_acceptance') if callable(identity) else None,
        'thinking': False, 'output_tokens': 5000, 'json_mode': True}))

