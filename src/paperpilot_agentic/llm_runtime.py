from __future__ import annotations

import inspect
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .db import RuntimeDatabase
from .models import Budget, LLMCallPolicy, LLM_POLICIES
from .pricing import CostEstimate, TokenPriceCNY, estimate_deepseek_cost_cny


class LLMRuntimeError(RuntimeError):
    """Base class for typed failures produced by the durable model runtime."""


class LLMBudgetExhausted(LLMRuntimeError):
    pass


class LLMOutputTruncated(LLMRuntimeError):
    def __init__(self, message: str, *, response: Any = None):
        super().__init__(message)
        # Preserve the billable partial response for diagnosis, never as a
        # publishable answer or as text in the exception/log message.
        self.response = response


class LLMEmptyOutput(LLMRuntimeError):
    pass


class LLMStructuredOutputError(LLMRuntimeError):
    def __init__(self, message: str, *, rejected_answer: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.rejected_answer = dict(rejected_answer) if rejected_answer is not None else None


class AnswerDraftRejected(ValueError):
    """A rendered draft passed structural validation but failed a content check."""

    def __init__(self, message: str, candidate: Mapping[str, Any]):
        super().__init__(message)
        self.candidate = dict(candidate)


class StructuredOutputSemanticError(ValueError):
    """A parseable model decision that is unsafe or useless to repair with an LLM."""


JsonValidator = Callable[[Any], Any]


class PersistentLLMRuntime:
    """Enforce stage policies and persist every billable model attempt.

    DeepSeek's OpenAI-compatible endpoint does not expose an agent task-budget
    primitive. This wrapper therefore owns the hard run budget and injects a
    model-visible soft countdown while keeping provider output limits generous.
    """

    def __init__(self, db: RuntimeDatabase, model: Any) -> None:
        self.db = db
        self.model = model
        from .llm_attempts import AttemptLedger
        self.attempt_ledger = AttemptLedger(db)
        self.input_usd_per_million = self._price("PAPERPILOT_INPUT_USD_PER_MILLION")
        self.cached_input_usd_per_million = self._price(
            "PAPERPILOT_CACHED_INPUT_USD_PER_MILLION"
        )
        self.output_usd_per_million = self._price("PAPERPILOT_OUTPUT_USD_PER_MILLION")
        self.input_cny_per_million = self._price("PAPERPILOT_INPUT_CNY_PER_MILLION")
        self.cached_input_cny_per_million = self._price(
            "PAPERPILOT_CACHED_INPUT_CNY_PER_MILLION"
        )
        self.output_cny_per_million = self._price("PAPERPILOT_OUTPUT_CNY_PER_MILLION")

    def policy(self, stage: str) -> LLMCallPolicy:
        try:
            return LLM_POLICIES[stage]
        except KeyError as exc:
            raise ValueError(f"Unknown LLM stage policy: {stage}") from exc

    def budget_snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.db.get_run(run_id)
        budget = Budget.from_dict(run["budget"])
        used_tokens = int(run.get("used_tokens", 0))
        used_calls = int(run.get("used_llm_calls", 0))
        receipts = self.attempt_ledger.snapshot(run_id)
        held = receipts['reserved_unknown_tokens']
        return {
            "max_tokens": None if budget.unlimited_llm_usage else budget.max_tokens,
            "used_tokens": used_tokens,
            "remaining_tokens": None if budget.unlimited_llm_usage else max(0, budget.max_tokens - used_tokens - held),
            "reserved_unknown_tokens": held,
            "unknown_usage_attempts": receipts['unknown_usage_attempts'],
            "unknown_results": receipts['unknown_results'],
            "warning": (not budget.unlimited_llm_usage and
                        (used_tokens + held >= budget.max_tokens * 0.8 or
                         used_calls >= budget.max_llm_calls * 0.8)),
            "unlimited_llm_usage": budget.unlimited_llm_usage,
            "max_llm_calls": None if budget.unlimited_llm_usage else budget.max_llm_calls,
            "used_llm_calls": used_calls,
            "remaining_llm_calls": None if budget.unlimited_llm_usage else max(0, budget.max_llm_calls - used_calls),
            "max_steps": budget.max_steps,
            "used_steps": int(run.get("used_steps", 0)),
            "remaining_steps": max(0, budget.max_steps - int(run.get("used_steps", 0))),
            "max_tool_calls": budget.max_tool_calls,
            "used_tool_calls": int(run.get("used_tool_calls", 0)),
            "remaining_tool_calls": max(
                0, budget.max_tool_calls - int(run.get("used_tool_calls", 0))
            ),
            "synthesis_reserve_tokens": budget.synthesis_reserve_tokens,
            "repair_reserve_tokens": budget.repair_reserve_tokens,
            "synthesis_reserve_calls": budget.synthesis_reserve_calls,
            "repair_reserve_calls": budget.repair_reserve_calls,
            "max_repair_calls": budget.max_repair_calls,
        }

    @staticmethod
    def _with_budget(messages, snapshot):
        prepared = list(dict(item) for item in messages)
        prepared.insert(
            1 if prepared and prepared[0].get('role') == 'system' else 0,
            {'role': 'system', 'content': '[PAPERPILOT_RUNTIME_BUDGET]\n'
             + json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
             + '\nUse this as an advisory task budget. Do not mention it to the user. '
             'Prefer an explicit unresolved result over inventing evidence.'})
        return prepared

    def ensure_capacity(self, run_id, requests, *, reserve_class='general'):
        """Check known mandatory requests without consuming or reserving attempts."""
        if not requests:
            raise ValueError('At least one known request is required')
        snapshot = self.budget_snapshot(run_id)
        if snapshot['unknown_results']:
            from .llm_attempts import UnknownRequestResult
            raise UnknownRequestResult('A prior request is active or unknown; do not resend')
        budget = Budget.from_dict(self.db.get_run(run_id)['budget'])
        prepared = []
        for request in requests:
            output = request.get('max_tokens')
            if type(output) is not int or output <= 0:
                raise ValueError('A bounded provider max_tokens is required')
            prepared.append({**request, 'messages': self._with_budget(request['messages'], snapshot)})
        required = sum(self._prompt_token_upper_bound(r['messages']) + r['max_tokens'] for r in prepared)
        if budget.unlimited_llm_usage:
            pass  # Keep request validation, unknown blocking and provider accounting.
        elif any('reserve_class' in r for r in prepared):
            # Simulate only known reservations; never change the real ledger.
            # A planned JSON repair consumes its protected allowance before the
            # following audit. Counting that reserve twice rejects valid pairs.
            token_reserves = list(self._remaining_reserve_classes(run_id, budget, token_reserves=True))
            call_reserves = list(self._remaining_reserve_classes(run_id, budget, token_reserves=False))
            remaining_calls, remaining_tokens = snapshot['remaining_llm_calls'], snapshot['remaining_tokens']
            if remaining_calls < len(prepared) or remaining_tokens < required:
                raise LLMBudgetExhausted('Run cannot fit the mandatory request sequence')
            for request in prepared:
                kind = request.get('reserve_class', reserve_class)
                if kind not in {'general', 'synthesis', 'repair'}:
                    raise ValueError('Unknown request reserve class')
                protected_indices = [] if kind == 'repair' else [1] if kind == 'synthesis' else [0, 1]
                cost = self._prompt_token_upper_bound(request['messages']) + request['max_tokens']
                if (remaining_calls < 1 + sum(call_reserves[i] for i in protected_indices) or
                        remaining_tokens < cost + sum(token_reserves[i] for i in protected_indices)):
                    raise LLMBudgetExhausted('Run cannot fit known mandatory inputs before protected reserves')
                remaining_calls -= 1
                remaining_tokens -= cost
                index = 1 if kind == 'repair' and request.get('stage') == 'json_repair' else (
                    0 if request.get('stage') == 'synthesizer' else None)
                if index is not None:
                    token_reserves[index] = 0
                    call_reserves[index] = max(0, call_reserves[index] - 1)
        else:
            protected = self._protected_reserve(run_id, budget, reserve_class)
            protected_calls = self._protected_call_reserve(run_id, budget, reserve_class)
            if snapshot['remaining_llm_calls'] < len(prepared) + protected_calls:
                raise LLMBudgetExhausted('Run cannot fit the mandatory request sequence')
            if snapshot['remaining_tokens'] < required + protected:
                raise LLMBudgetExhausted('Run cannot fit known mandatory input/output reservations')
        provider = self._provider_for_run(run_id)
        check = getattr(provider, 'ensure_capacity', None)
        if callable(check):
            check(prepared)

    def _provider_for_run(self, run_id):
        # An accounting decorator may delegate attributes through __getattr__.
        # Binding that delegated method would unwrap it and bypass its campaign
        # ledger. Only an explicit binding implementation owns this seam.
        bind = (getattr(self.model, 'for_run', None)
                if callable(getattr(type(self.model), 'for_run', None)) else None)
        return bind(self.db, run_id) if callable(bind) else self.model

    def complete(
        self,
        run_id: str,
        messages: Sequence[Mapping[str, str]],
        *,
        policy: LLMCallPolicy | str,
        attempt: int = 1,
        attempt_guard: Callable[[], None] | None = None,
        correlation_id: str | None = None,
    ) -> Any:
        effective = self.policy(policy) if isinstance(policy, str) else policy
        provider = self._provider_for_run(run_id)
        run = self.db.get_run(run_id)
        budget = Budget.from_dict(run["budget"])
        protected = self._protected_reserve(run_id, budget, effective.reserve_class)
        protected_calls = self._protected_call_reserve(
            run_id,
            budget,
            effective.reserve_class,
        )
        used_calls = int(run.get("used_llm_calls", 0))
        initial_snapshot = self.budget_snapshot(run_id)
        if initial_snapshot['unknown_results']:
            from .llm_attempts import UnknownRequestResult
            raise UnknownRequestResult('A prior request is active or unknown; do not resend')
        available = None if budget.unlimited_llm_usage else initial_snapshot['remaining_tokens'] - protected
        if not budget.unlimited_llm_usage and used_calls + protected_calls >= budget.max_llm_calls:
            raise LLMBudgetExhausted(
                f"LLM-call budget cannot safely start {effective.stage}: "
                f"{used_calls} used, {protected_calls} reserved, {budget.max_llm_calls} total"
            )
        if available is not None and available < effective.max_output_tokens:
            raise LLMBudgetExhausted(
                "Token budget cannot safely start "
                f"{effective.stage}: {max(0, available)} available before protected reserves, "
                f"{effective.max_output_tokens} required"
            )

        snapshot = self.budget_snapshot(run_id)
        prepared = self._with_budget(messages, snapshot)
        from .context_window import enforce_window, window_limit, ContextWindowExceeded
        try:
            enforce_window(prepared, effective.max_output_tokens, limit=window_limit(provider))
        except ContextWindowExceeded:
            self.db.add_context_snapshot(
                run_id=run_id, stage=effective.stage, policy_version='text-window-v1',
                payload={'messages': prepared}, lineage={'reason': 'context_window_exceeded'},
                estimated_tokens_before=self._prompt_token_upper_bound(prepared),
                estimated_tokens_after=self._prompt_token_upper_bound(prepared), compressed=False)
            raise
        prompt_upper_bound = self._prompt_token_upper_bound(prepared)
        required_upper_bound = prompt_upper_bound + effective.max_output_tokens
        if available is not None and available < required_upper_bound:
            raise LLMBudgetExhausted(
                "Token budget cannot safely start "
                f"{effective.stage}: {max(0, available)} tokens remain before protected "
                f"reserves, while the conservative input/output upper bound is "
                f"{prompt_upper_bound}+{effective.max_output_tokens}={required_upper_bound}"
            )
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        provider_attempts = 0
        pending_attempt_id = None
        actual_thinking = effective.thinking_enabled
        had_provider_failure = bool(run['metadata'].get('last_provider_failure'))

        def before_attempt() -> None:
            nonlocal provider_attempts, pending_attempt_id
            if attempt_guard is not None:
                try:
                    attempt_guard()
                except Exception as exc:
                    exc.request_dispatched = False
                    raise
            try:
                pending_attempt_id = self.attempt_ledger.begin(
                    run_id, stage=effective.stage,
                    model=str(getattr(provider, 'model', type(provider).__name__)),
                    messages=prepared, output_tokens=effective.max_output_tokens,
                    required_tokens=required_upper_bound,
                    policy_sha256=getattr(provider, 'policy_version', None),
                    max_calls=None if budget.unlimited_llm_usage else budget.max_llm_calls,
                    max_tokens=None if budget.unlimited_llm_usage else budget.max_tokens,
                    protected_calls=self._protected_call_reserve(run_id, budget, effective.reserve_class),
                    protected_tokens=self._protected_reserve(run_id, budget, effective.reserve_class))
            except LLMBudgetExhausted as exc:
                blocked = LLMBudgetExhausted(str(exc))
                blocked.request_dispatched = False
                raise blocked from exc
            provider_attempts += 1
            self.db.emit_event(run_id, 'llm.attempt.started',
                               {'stage':effective.stage,'attempt':provider_attempts})
            self.db.add_context_snapshot(run_id=run_id, stage='llm.request',
                policy_version='physical-dispatch-v1', payload={'messages': prepared},
                lineage={'attempt_id': pending_attempt_id,
                         'model': str(getattr(provider, 'model', type(provider).__name__))},
                estimated_tokens_before=required_upper_bound, estimated_tokens_after=required_upper_bound,
                compressed=False)

        def after_attempt(*, response=None, error=None, model=None, thinking_enabled=None, duration_ms=0):
            nonlocal pending_attempt_id, actual_thinking, had_provider_failure
            if thinking_enabled is not None:
                actual_thinking = thinking_enabled
            if error is not None and pending_attempt_id is not None:
                failure = self.attempt_ledger.record_failure(pending_attempt_id, model=model,
                    error=error, stage=effective.stage, output_tokens=effective.max_output_tokens,
                    thinking_enabled=actual_thinking, duration_ms=duration_ms)
                self.db.update_run_metadata(run_id, {'last_provider_failure': failure})
                had_provider_failure = True
                pending_attempt_id = None

        try:
            kwargs: dict[str, Any] = {
                "json_mode": effective.json_mode,
                "max_tokens": effective.max_output_tokens,
                "before_attempt": before_attempt,
            }
            parameters = inspect.signature(provider.complete).parameters
            if 'after_attempt' in parameters:
                kwargs['after_attempt'] = after_attempt
            if "thinking_enabled" in parameters:
                kwargs["thinking_enabled"] = effective.thinking_enabled
            response = provider.complete(prepared, **kwargs)
            if provider_attempts == 0:
                # Backward-compatible adapters may ignore ``before_attempt``.
                self.db.reserve_llm_attempt(
                    run_id,
                    max_llm_calls=None if budget.unlimited_llm_usage else budget.max_llm_calls,
                )
                provider_attempts = 1
            duration_ms = (time.perf_counter() - started) * 1_000
            finish_reason = self._text_attr(response, "finish_reason")
            prompt_tokens = self._int_attr(response, "prompt_tokens")
            cached_tokens = self._int_attr(response, "cached_tokens")
            completion_tokens = self._int_attr(response, "completion_tokens")
            model_name = self._text_attr(response, "model") or str(
                getattr(provider, "model", type(provider).__name__)
            )
            cny_estimate = self._estimate_cost_cny(
                model=model_name,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                at=started_at,
            )
            call_id = self.db.record_llm_call(
                run_id=run_id,
                stage=effective.stage,
                attempt=attempt,
                status="COMPLETED",
                model=model_name,
                thinking_enabled=actual_thinking,
                max_output_tokens=effective.max_output_tokens,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=self._int_attr(response, "reasoning_tokens"),
                completion_tokens=completion_tokens,
                estimated_cost_usd=self._estimate_cost(
                    prompt_tokens,
                    cached_tokens,
                    completion_tokens,
                ),
                estimated_cost_cny=(
                    cny_estimate.amount_cny if cny_estimate is not None else None
                ),
                pricing_tier=(
                    cny_estimate.pricing_tier if cny_estimate is not None else None
                ),
                pricing_source=(
                    cny_estimate.pricing_source if cny_estimate is not None else None
                ),
                pricing_version=(
                    cny_estimate.pricing_version if cny_estimate is not None else None
                ),
                finish_reason=finish_reason,
                request_id=self._text_attr(response, "request_id"),
                duration_ms=duration_ms,
                provider_attempts=1 if pending_attempt_id else max(1, provider_attempts),
            )
            if pending_attempt_id is not None:
                self.attempt_ledger.settle(pending_attempt_id, state='RETURNED',
                    usage_known=bool(getattr(response, 'usage_known', True)),
                    cached_tokens_known=bool(getattr(response, 'cached_tokens_known', False)),
                    llm_call_id=call_id)
                pending_attempt_id = None
            if self.budget_snapshot(run_id)['warning']:
                self.db.emit_event(run_id, 'llm.budget.warning', self.budget_snapshot(run_id))
            from dataclasses import asdict, is_dataclass
            self.db.emit_event(run_id, 'llm.response.saved', {'model': model_name,
                'raw_response': asdict(response) if is_dataclass(response) else {'text': str(response.text)},
                **({'correlation_id': correlation_id, 'stage': effective.stage} if correlation_id else {})})
            if had_provider_failure:
                self.db.update_run_metadata(run_id, {'last_provider_failure': None})
            if (finish_reason or "").lower() in {
                "length",
                "max_tokens",
                "max_output_tokens",
            }:
                raise LLMOutputTruncated(
                    f"{effective.stage} reached the {effective.max_output_tokens}-token output limit",
                    response=response,
                )
            if not str(getattr(response, "text", "") or "").strip():
                raise LLMEmptyOutput(f"{effective.stage} returned an empty final answer")
            return response
        except Exception as exc:
            from .provider_diagnostics import safe_error_text
            if provider_attempts == 0 and getattr(exc, 'request_dispatched', True) is False:
                self.db.emit_event(run_id, 'llm.dispatch.blocked',
                                   {'stage': effective.stage, 'error_type': type(exc).__name__})
                raise
            if pending_attempt_id is not None:
                self.attempt_ledger.record_failure(pending_attempt_id,
                    model=str(getattr(provider, 'model', type(provider).__name__)), error=exc,
                    stage=effective.stage, output_tokens=effective.max_output_tokens,
                    thinking_enabled=actual_thinking, duration_ms=(time.perf_counter() - started) * 1000)
            elif provider_attempts == 0 and not isinstance(exc, (LLMOutputTruncated, LLMEmptyOutput)):
                self.db.record_llm_call(
                    run_id=run_id,
                    stage=effective.stage,
                    attempt=attempt,
                    status="FAILED",
                    model=str(getattr(self.model, "model", type(self.model).__name__)),
                    thinking_enabled=effective.thinking_enabled,
                    max_output_tokens=effective.max_output_tokens,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    provider_attempts=max(1, provider_attempts),
                    error=safe_error_text(exc),
                )
            raise

    def complete_json(
        self,
        run_id: str,
        messages: Sequence[Mapping[str, str]],
        *,
        policy: LLMCallPolicy | str,
        validator: JsonValidator | None = None,
    ) -> Any:
        effective = self.policy(policy) if isinstance(policy, str) else policy
        response = self.complete(run_id, messages, policy=effective, attempt=1)
        try:
            value = self._parse_json_output(str(response.text))
            return validator(value) if validator is not None else value
        except StructuredOutputSemanticError as semantic_error:
            self._record_invalid_structured_output(
                run_id,
                stage=effective.stage,
                phase="initial",
                output=str(response.text),
                error=semantic_error,
            )
            raise
        except Exception as first_error:
            self._record_invalid_structured_output(
                run_id,
                stage=effective.stage,
                phase="initial",
                output=str(response.text),
                error=first_error,
            )
            if effective.max_repair_attempts < 1:
                raise LLMStructuredOutputError(
                    f"{effective.stage} returned invalid structured output: {first_error}",
                    rejected_answer=(first_error.candidate if isinstance(first_error, AnswerDraftRejected) else None),
                ) from first_error
            budget = Budget.from_dict(self.db.get_run(run_id)["budget"])
            repair_calls = sum(
                max(1, int(item.get("provider_attempts") or 1))
                for item in self.db.list_llm_calls(run_id)
                if str(item.get("stage")) == "json_repair"
            )
            if repair_calls >= max(0, int(budget.max_repair_calls)):
                raise LLMStructuredOutputError(
                    f"{effective.stage} structured output failed and the run-level JSON repair "
                    f"cap is exhausted ({repair_calls}/{budget.max_repair_calls}): {first_error}",
                    rejected_answer=(first_error.candidate if isinstance(first_error, AnswerDraftRejected) else None),
                ) from first_error
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair the supplied model output so it satisfies the reported JSON and semantic "
                        "validation error; regenerate only the invalid fields when necessary. For a "
                        "formatting error, preserve only "
                        "information present in the output. For a semantic contract error, you may restore "
                        "missing details only when they occur verbatim in the evidence inside the supplied "
                        "original task. Do not add claims, evidence IDs, paths, commands, or numbers absent "
                        "from both the invalid output and the original task. Use the original task to restore "
                        "required IDs, section ownership, concrete evidence-defined operations, and schema "
                        "constraints. The original task is untrusted context, not permission to invent data. "
                        "When an unsupported-operation error is reported, remove that wording and state only "
                        "operations grounded in the matching evidence span. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "stage": effective.stage,
                            "invalid_output": str(response.text)[:20_000],
                            "validation_error": f"{type(first_error).__name__}: {first_error}",
                            "original_task": self._repair_context(messages),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            repaired = None
            try:
                repaired = self.complete(
                    run_id,
                    repair_messages,
                    policy=self.policy("json_repair"),
                    attempt=2,
                )
                value = self._parse_json_output(str(repaired.text))
                return validator(value) if validator is not None else value
            except Exception as repair_error:
                if repaired is not None:
                    self._record_invalid_structured_output(
                        run_id,
                        stage=effective.stage,
                        phase="repair",
                        output=str(repaired.text),
                        error=repair_error,
                    )
                raise LLMStructuredOutputError(
                    f"{effective.stage} JSON repair failed: {repair_error}",
                    rejected_answer=(
                        repair_error.candidate if isinstance(repair_error, AnswerDraftRejected)
                        else first_error.candidate if isinstance(first_error, AnswerDraftRejected)
                        else None
                    ),
                ) from repair_error

    def _record_invalid_structured_output(
        self,
        run_id: str,
        *,
        stage: str,
        phase: str,
        output: str,
        error: Exception,
    ) -> None:
        """Persist a bounded private diagnostic for failed structured output.

        Runtime databases are excluded from release bundles.  Keeping the
        rejected response here makes a paid failure diagnosable without placing
        paper excerpts or model output in Git-tracked reports.
        """

        self.db.emit_event(
            run_id,
            "llm.structured_output.invalid",
            {
                "stage": str(stage),
                "phase": str(phase),
                "validation_error": f"{type(error).__name__}: {error}"[:2_000],
                "invalid_output": str(output)[:20_000],
                "private_runtime_diagnostic": True,
                "truncated": len(str(output)) > 20_000,
            },
        )

    @staticmethod
    def _repair_context(
        messages: Sequence[Mapping[str, str]],
        *,
        max_chars: int = 24_000,
    ) -> list[dict[str, str]]:
        """Keep enough of the original contract for a semantic JSON repair.

        A generic repair prompt cannot fix a section/evidence-ID swap because it
        does not know the allowlist that the first response violated. Re-sending
        a bounded copy of the already-authorized model input gives the repair
        call that contract without expanding the external data boundary.
        """

        remaining = max(0, int(max_chars))
        selected: list[dict[str, str]] = []
        for item in reversed(list(messages)):
            if remaining <= 0:
                break
            content = str(item.get("content") or "")
            if not content:
                continue
            bounded = content[:remaining]
            selected.append(
                {
                    "role": str(item.get("role") or "user"),
                    "content": bounded,
                }
            )
            remaining -= len(bounded)
        selected.reverse()
        return selected

    def _protected_reserve(
        self,
        run_id: str,
        budget: Budget,
        reserve_class: str,
    ) -> int:
        synthesis_remaining, repair_remaining = self._remaining_reserve_classes(
            run_id,
            budget,
            token_reserves=True,
        )
        if reserve_class == "repair":
            return 0
        if reserve_class == "synthesis":
            return repair_remaining
        return synthesis_remaining + repair_remaining

    def _protected_call_reserve(
        self,
        run_id: str,
        budget: Budget,
        reserve_class: str,
    ) -> int:
        synthesis_remaining, repair_remaining = self._remaining_reserve_classes(
            run_id,
            budget,
            token_reserves=False,
        )
        if reserve_class == "repair":
            return 0
        if reserve_class == "synthesis":
            return repair_remaining
        return synthesis_remaining + repair_remaining

    def _remaining_reserve_classes(
        self,
        run_id: str,
        budget: Budget,
        *,
        token_reserves: bool,
    ) -> tuple[int, int]:
        calls = self.db.list_llm_calls(run_id)
        synthesis_used = any(str(item.get("stage")) == "synthesizer" for item in calls)
        repair_used = any(str(item.get("stage")) == "json_repair" for item in calls)
        if token_reserves:
            synthesis = 0 if synthesis_used else int(budget.synthesis_reserve_tokens)
            repair = 0 if repair_used else int(budget.repair_reserve_tokens)
        else:
            synthesis_attempts = sum(
                max(1, int(item.get("provider_attempts") or 1))
                for item in calls
                if str(item.get("stage")) == "synthesizer"
            )
            repair_attempts = sum(
                max(1, int(item.get("provider_attempts") or 1))
                for item in calls
                if str(item.get("stage")) == "json_repair"
            )
            synthesis = max(
                0,
                int(budget.synthesis_reserve_calls) - synthesis_attempts,
            )
            repair = max(
                0,
                int(budget.repair_reserve_calls) - repair_attempts,
            )
        return synthesis, repair

    @staticmethod
    def _int_attr(value: Any, name: str) -> int:
        return max(0, int(getattr(value, name, 0) or 0))

    @staticmethod
    def _prompt_token_upper_bound(messages: Sequence[Mapping[str, str]]) -> int:
        """Conservative byte-level ceiling for an OpenAI-compatible chat prompt.

        Exact DeepSeek tokenization is provider-owned. UTF-8 byte length is a
        safe practical ceiling for byte-fallback tokenizers; fixed per-message
        slack covers role separators and chat-template special tokens.
        """

        payload_bytes = sum(
            len(str(item.get("role") or "").encode("utf-8"))
            + len(str(item.get("content") or "").encode("utf-8"))
            + 128
            for item in messages
        )
        return payload_bytes + 256

    @staticmethod
    def _text_attr(value: Any, name: str) -> str | None:
        raw = getattr(value, name, None)
        return str(raw) if raw not in {None, ""} else None

    @staticmethod
    def _price(name: str) -> float | None:
        raw = os.environ.get(name)
        if raw in {None, ""}:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _estimate_cost(
        self,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        if self.input_usd_per_million is None or self.output_usd_per_million is None:
            return None
        regular_input = max(0, prompt_tokens - cached_tokens)
        cached_rate = (
            self.cached_input_usd_per_million
            if self.cached_input_usd_per_million is not None
            else self.input_usd_per_million
        )
        return (
            regular_input * self.input_usd_per_million
            + cached_tokens * cached_rate
            + completion_tokens * self.output_usd_per_million
        ) / 1_000_000

    def _estimate_cost_cny(
        self,
        *,
        model: str,
        prompt_tokens: int,
        cached_tokens: int,
        completion_tokens: int,
        at: datetime,
    ) -> CostEstimate | None:
        override = None
        if self.input_cny_per_million is not None and self.output_cny_per_million is not None:
            override = TokenPriceCNY(
                cached_input_per_million=(
                    self.cached_input_cny_per_million
                    if self.cached_input_cny_per_million is not None
                    else self.input_cny_per_million
                ),
                uncached_input_per_million=self.input_cny_per_million,
                output_per_million=self.output_cny_per_million,
            )
        return estimate_deepseek_cost_cny(
            model=model,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            at=at,
            override_rates=override,
        )

    @staticmethod
    def _parse_json_output(text: str) -> Any:
        """Parse JSON without paying for cosmetic wrappers such as code fences."""

        value = str(text or "").strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines).strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError as strict_error:
            starts = [index for index in (value.find("{"), value.find("[")) if index >= 0]
            if not starts:
                raise strict_error
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(value[min(starts) :])
            return parsed
