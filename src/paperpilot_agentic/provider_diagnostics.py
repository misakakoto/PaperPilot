"""Safe machine facts for provider failures; never serialize exception messages."""
from __future__ import annotations

import json
import re

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError


def provider_error_code(error):
    body = getattr(error, 'body', None)
    body = body if isinstance(body, dict) else {}
    detail = body.get('error', body)
    value = detail.get('code') if isinstance(detail, dict) else None
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value) else None


def failure_category(error):
    """Classify explicit status/codes, never arbitrary text or missing responses."""
    if isinstance(error, APITimeoutError):
        return 'TIMEOUT_UNKNOWN_OUTCOME'
    if isinstance(error, APIConnectionError):
        return 'CONNECTION_ERROR'
    if not isinstance(error, APIStatusError):
        return 'LOCAL_ERROR'
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get('error', body)
    detail = detail if isinstance(detail, dict) else {}
    codes = {str(detail.get(key, '')).casefold() for key in ('code', 'type')}
    if error.status_code == 402 or codes & {
        'insufficient_quota', 'arrearage', 'allocationquota.freetieronly', 'isv.out_of_service'
    }:
        return 'BILLING_OR_QUOTA'
    if error.status_code == 429:
        return 'RATE_LIMIT'
    if error.status_code in {401, 403}:
        return 'AUTH_OR_PERMISSION'
    if error.status_code == 404:
        return 'MODEL_OR_ENDPOINT_UNAVAILABLE'
    return 'PROVIDER_ERROR'


def _exception_chain(error):
    seen = set()
    for _ in range(8):
        if not isinstance(error, BaseException) or id(error) in seen:
            break
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def safe_provider_failure(error):
    """Bounded cause chain with types and OS codes, excluding bodies/URLs/keys."""
    chain = list(_exception_chain(error))
    causes = []
    phase = 'unknown'
    phases = (
        ((httpx.ConnectError, httpx.ConnectTimeout), 'connect'),
        ((httpx.ReadError, httpx.ReadTimeout), 'read'),
        ((httpx.WriteError, httpx.WriteTimeout), 'write'),
        ((httpx.PoolTimeout,), 'pool'),
        ((httpx.RemoteProtocolError,), 'protocol'),
    )
    for cause in chain:
        item = {'type': type(cause).__name__}
        for name in ('errno', 'winerror'):
            value = getattr(cause, name, None)
            if type(value) is int:
                item[name] = value
        causes.append(item)
        for types, label in phases:
            if isinstance(cause, types):
                phase = label
                break
    status = getattr(error, 'status_code', None)
    return {
        'error_type': type(error).__name__, 'category': failure_category(error),
        'http_status': status if type(status) is int else None,
        'provider_code': provider_error_code(error),
        'transport_phase': phase, 'cause_chain': causes,
    }


def safe_error_text(error):
    """Keep local diagnostics readable, redact provider failures at log boundaries."""
    if any(isinstance(cause, OpenAIError) for cause in _exception_chain(error)):
        return json.dumps(safe_provider_failure(error), ensure_ascii=True)
    return f'{type(error).__name__}: {error}'
