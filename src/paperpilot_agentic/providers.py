"""OpenAI-compatible text provider used by Planner, Controller and Synthesizer."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

from .config import RuntimeSettings


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    request_id: str | None = None
    model: str | None = None
    usage_known: bool = True
    cached_tokens_known: bool = False

    @property
    def total_tokens(self) -> int:
        return max(0, self.prompt_tokens) + max(0, self.completion_tokens)


class OpenAICompatibleLLM:
    """Provider-neutral adapter with explicit Qwen and DeepSeek thinking controls."""

    def __init__(self, settings: RuntimeSettings, *, max_retries: int = 2) -> None:
        if not settings.api_key:
            raise ValueError("LLM API key is required")
        self.provider = settings.provider
        self.model = settings.model
        self.thinking_enabled = settings.thinking_enabled
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.max_retries = max(0, int(max_retries))
        self.base_url = settings.base_url
        self.fallback_models = getattr(settings, 'fallback_models', ())
        from .model_policy import FREE_MODELS
        if self.fallback_models and (self.provider != 'qwen' or self.model not in FREE_MODELS or
                len(self.fallback_models) != 1 or self.fallback_models[0] not in FREE_MODELS or
                self.fallback_models[0] == self.model or
                self.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1'):
            raise ValueError('Free fallback requires the confirmed Beijing Qwen model pair')
        from .context_window import window_limit
        self.context_window_tokens = window_limit()
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    def policy_identity(self, profile):
        from .model_policy import policy_identity
        return policy_identity(self.provider, self.model, getattr(self, 'fallback_models', ()),
                               getattr(self, 'base_url', ''), profile)

    def for_run(self, db, run_id):
        from .model_policy import BoundFreeQuotaProvider
        run = db.get_run(run_id)
        identity = self.policy_identity(run['metadata'].get('execution_profile', 'frozen_acceptance'))
        previous = run['metadata'].get('model_policy')
        if previous is not None and previous != identity:
            raise ValueError('Model policy changed; use a new run rather than changing a checkpoint')
        if previous is None:
            db.update_run_metadata(run_id, {'model_policy': identity})
        return BoundFreeQuotaProvider(self, db, identity)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        before_attempt: Callable[[], None] | None = None,
        after_attempt: Callable[..., None] | None = None,
    ) -> LLMResponse:
        from .context_window import enforce_window, window_limit
        enforce_window(messages, max_tokens or self.max_tokens, limit=window_limit(self))
        last_error: Exception | None = None
        thinking = self.thinking_enabled if thinking_enabled is None else bool(thinking_enabled)
        from .model_policy import FREE_MODELS
        fixed_instruct = FREE_MODELS.get(self.model, {}).get('thinking') is False
        if fixed_instruct:
            thinking = False
        for attempt in range(self.max_retries + 1):
            dispatched = False
            started = time.perf_counter()
            try:
                kwargs: dict[str, object] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens or self.max_tokens,
                }
                if self.provider == "qwen" and not fixed_instruct:
                    kwargs["extra_body"] = {"enable_thinking": thinking}
                elif self.provider != 'qwen':
                    kwargs["extra_body"] = {
                        "thinking": {"type": "enabled" if thinking else "disabled"}
                    }
                if not thinking:
                    kwargs["temperature"] = self.temperature
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if before_attempt is not None:
                    try:
                        before_attempt()
                    except Exception as guard_error:
                        if last_error is not None:
                            raise guard_error from last_error
                        raise
                dispatched = True
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                usage = response.usage
                completion_details = getattr(usage, "completion_tokens_details", None)
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                cached = (
                    getattr(usage, "prompt_cache_hit_tokens", 0)
                    or getattr(prompt_details, "cached_tokens", 0)
                    or 0
                )
                result = LLMResponse(
                    text=choice.message.content or "",
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    cached_tokens=int(cached),
                    reasoning_tokens=int(
                        getattr(completion_details, "reasoning_tokens", 0) or 0
                    ),
                    finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
                    request_id=str(getattr(response, "id", "") or "") or None,
                    model=str(getattr(response, "model", "") or self.model),
                    usage_known=(usage is not None and getattr(usage, 'prompt_tokens', None) is not None
                                 and getattr(usage, 'completion_tokens', None) is not None),
                    cached_tokens_known=(getattr(usage, 'prompt_cache_hit_tokens', None) is not None
                                         or getattr(prompt_details, 'cached_tokens', None) is not None),
                )
                if after_attempt is not None:
                    after_attempt(response=result, model=self.model, thinking_enabled=thinking,
                                  duration_ms=(time.perf_counter() - started) * 1000)
                return result
            except OpenAIError as exc:
                if dispatched and after_attempt is not None:
                    after_attempt(error=exc, model=self.model, thinking_enabled=thinking,
                                  duration_ms=(time.perf_counter() - started) * 1000)
                # A connection/timeout failure has no confirmed response. It
                # may have reached the provider; leave the original outcome
                # unresolved instead of silently sending the same work again.
                if isinstance(exc, APIConnectionError):
                    raise
                from .provider_diagnostics import failure_category
                if failure_category(exc) == 'BILLING_OR_QUOTA':
                    raise
                # A serving window can differ from the configured model limit.
                # Signal the caller to split its source batch, never blind retry.
                if isinstance(exc, APIStatusError) and exc.status_code in {400, 413}:
                    detail = str(exc).lower()
                    if any(x in detail for x in ('context_length_exceeded', 'maximum context length', 'context window', 'too many tokens')):
                        from .context_window import ContextWindowExceeded
                        raise ContextWindowExceeded('Provider rejected context length; split the source workset') from exc
                # Retrying invalid credentials, permissions, model IDs or request
                # parameters cannot repair them and may mask the original error
                # with a local attempt-budget failure on the next iteration.
                if isinstance(exc, APIStatusError) and 400 <= exc.status_code < 500:
                    if exc.status_code not in {408, 409, 429}:
                        raise
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        assert last_error is not None
        raise last_error
