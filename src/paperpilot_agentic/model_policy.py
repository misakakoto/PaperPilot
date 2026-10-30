"""Explicit free-quota routing; availability is not answer-quality evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date

from .provider_diagnostics import provider_error_code

POLICY_VERSION = 'free-quota-v1'
PRIMARY_MODEL = 'qwen3-vl-30b-a3b-instruct'
BACKUP_MODEL = 'qwen3-vl-32b-instruct'
DAILY_MODEL = 'qwen3.5-plus-2026-02-15'
FREE_MODELS = {
    name: {'text': True, 'image': True, 'structured_output': True,
           'thinking': False, 'context_cache': False, 'context_window_tokens': 131072,
           'checked_on': '2026-10-07', 'free_quota_expires_on': '2026-11-27',
           'free_tier_only_confirmed': True}
    for name in (PRIMARY_MODEL, BACKUP_MODEL)
}
# Preserve the old pair's metadata/identity for historical checkpoints. The new
# daily snapshot is hybrid: non-streaming JSON must explicitly disable thinking.
FREE_MODELS[DAILY_MODEL] = {
    'text': True, 'image': True, 'structured_output': True,
    'thinking': True, 'context_cache': True, 'context_window_tokens': 1000000,
    'checked_on': '2026-10-11', 'free_quota_expires_on': '2026-11-27',
    'free_tier_only_confirmed': True,
}
# Console verified 2026-10-13: this older VL channel has free quota and
# FreeTierOnly enabled. Keep a conservative serving limit; the channel is
# unversioned, so a frozen campaign records its requested name and receipts.
FREE_MODELS['qwen3-vl-plus'] = {
    'text': True, 'image': True, 'structured_output': True,
    'thinking': True, 'context_cache': False, 'context_window_tokens': 131072,
    'checked_on': '2026-10-13', 'free_quota_expires_on': '2026-11-27',
    'free_tier_only_confirmed': True,
}
# Console re-login on 2026-10-15: dated Flash has 1M free tokens and
# FreeTierOnly enabled. Keep historical entries unchanged. A frozen experiment
# records the dated name; availability must still pass the counted API probe.
FREE_MODELS['qwen3.5-flash-2026-02-23'] = {
    'text': True, 'image': True, 'structured_output': True,
    'thinking': True, 'context_cache': False, 'context_window_tokens': 131072,
    'checked_on': '2026-10-15', 'free_quota_expires_on': '2026-11-27',
    'free_tier_only_confirmed': True,
}
# Same-day console and official capability check. Keep a conservative serving
# limit identical to Flash for the fixed-model workflow comparison.
FREE_MODELS['qwen3.5-plus-2026-04-20'] = {
    'text': True, 'image': True, 'structured_output': True,
    'thinking': True, 'context_cache': True, 'context_window_tokens': 131072,
    'checked_on': '2026-10-15', 'free_quota_expires_on': '2026-11-27',
    'free_tier_only_confirmed': True,
}


class ModelQuotaExhausted(RuntimeError):
    request_dispatched = False


def policy_identity(provider, model, fallback_models, base_url, profile):
    if profile not in {'daily', 'frozen_acceptance'}:
        raise ValueError('Unknown execution_profile')
    names = [model] + (list(fallback_models) if profile == 'daily' else [])
    value = {'version': POLICY_VERSION, 'profile': profile, 'provider': provider,
             'models': list(dict.fromkeys(names)),
             'endpoint_sha256': hashlib.sha256(base_url.encode()).hexdigest(),
             'capabilities': {name: FREE_MODELS[name] for name in names if name in FREE_MODELS}}
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


class FreeQuotaPolicy:
    """A circuit stays open until an operator supplies a new policy identity.

    No timer, browser automation or health probe spends quota to clear it.
    Only the explicit FreeTierOnly provider code permits fallback.
    """
    def __init__(self, db, identity):
        self.db, self.identity = db, identity

    def candidates(self):
        names = self.identity['models']
        if self.identity['profile'] == 'frozen_acceptance':
            names = names[:1]
        if len(names) > 2:
            raise ValueError('Daily free routing accepts one primary and one backup')
        result = []
        for name in names:
            details = FREE_MODELS.get(name)
            if ((len(names) > 1 and not details) or (details and
                    date.today() >= date.fromisoformat(details['free_quota_expires_on']))):
                continue
            with self.db.connect() as conn:
                row = conn.execute('SELECT state FROM provider_model_state WHERE policy_sha256=? AND model=?',
                                   (self.identity['sha256'], name)).fetchone()
            if row is None or row['state'] != 'FREE_QUOTA_EXHAUSTED':
                result.append(name)
        if not result:
            raise ModelQuotaExhausted('No confirmed free model is available; verify quota before a new policy')
        return result

    def execute(self, send):
        from openai import APIStatusError
        candidates = self.candidates()
        for index, name in enumerate(candidates):
            try:
                return send(name)
            except Exception as error:
                if (not isinstance(error, APIStatusError) or error.status_code != 403 or
                        provider_error_code(error) != 'AllocationQuota.FreeTierOnly'):
                    raise
                with self.db.transaction() as conn:
                    conn.execute('INSERT INTO provider_model_state(policy_sha256,model,state) VALUES(?,?,?) '
                                 'ON CONFLICT(policy_sha256,model) DO UPDATE SET state=excluded.state',
                                 (self.identity['sha256'], name, 'FREE_QUOTA_EXHAUSTED'))
                if self.identity['profile'] == 'frozen_acceptance' or index == len(candidates) - 1:
                    raise


class BoundFreeQuotaProvider:
    """Each leaf adapter still invokes the physical-attempt accounting hooks."""
    def __init__(self, original, db, identity):
        self.original = original
        self.policy = FreeQuotaPolicy(db, identity)
        self.model = original.model
        from .context_window import window_limit
        limits = [window_limit(original)] + [FREE_MODELS[name]['context_window_tokens']
                  for name in identity['models'] if name in FREE_MODELS]
        self.context_window_tokens = min(limits)
        self.policy_version = identity['sha256']

    def complete(self, messages, *, json_mode=False, max_tokens=None,
                 thinking_enabled=None, before_attempt=None, after_attempt=None):
        def send(name):
            self.model = name
            leaf = copy.copy(self.original)
            leaf.model = name
            return leaf.complete(messages, json_mode=json_mode, max_tokens=max_tokens,
                                 thinking_enabled=thinking_enabled, before_attempt=before_attempt,
                                 after_attempt=after_attempt)
        return self.policy.execute(send)

    def ensure_capacity(self, requests):
        self.policy.candidates()
