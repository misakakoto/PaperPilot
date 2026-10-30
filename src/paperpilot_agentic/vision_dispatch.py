"""Qwen3-VL SDK dispatch with the same durable allowance as text calls."""
import time

from .llm_attempts import AttemptLedger
from .llm_runtime import PersistentLLMRuntime
from .model_policy import FreeQuotaPolicy, policy_identity
from .models import Budget


def vision_input_upper_bound(messages):
    # Beijing Qwen3-VL and Qwen3.5 document 16,384 tokens per still image.
    # Count that full ceiling rather than treating base64 bytes as text tokens.
    # https://help.aliyun.com/zh/model-studio/vision/
    total = 256
    for message in messages:
        total += 2048
        content = message.get('content', '')
        if isinstance(content, str):
            total += len(content.encode('utf-8'))
            continue
        for part in content:
            if part.get('type') == 'image_url':
                total += 16384
            elif part.get('type') == 'text':
                total += len(part.get('text', '').encode('utf-8'))
            else:
                raise ValueError('Only text and still-image inputs have a verified vision bound')
    return total


class VisionDispatch:
    def __init__(self, adapter, db, run_id):
        self.adapter, self.db, self.run_id = adapter, db, run_id
        self.runtime = PersistentLLMRuntime(db, None)
        self.ledger = AttemptLedger(db)
        profile = db.get_run(run_id)['metadata'].get('execution_profile', 'frozen_acceptance')
        self.identity = policy_identity('qwen', adapter.model, adapter.fallback_models, adapter.base_url, profile)
        self.policy = FreeQuotaPolicy(db, self.identity)
        metadata = db.get_run(run_id)['metadata']
        previous = metadata.get('vision_model_policy')
        if previous is not None and previous != self.identity:
            raise ValueError('Vision model policy changed; use a new run')
        if previous is None:
            db.update_run_metadata(run_id, {'vision_model_policy': self.identity})

    def send(self, kwargs, *, stage, before_attempt=None, on_dispatch=None):
        def send(model):
            run = self.db.get_run(self.run_id)
            budget = Budget.from_dict(run['budget'])
            if before_attempt:
                try:
                    before_attempt()
                except Exception as error:
                    error.request_dispatched = False
                    raise
            output = kwargs['max_tokens']
            needed = vision_input_upper_bound(kwargs['messages']) + output
            from .context_window import ContextWindowExceeded
            if needed + 1024 > 131072:
                raise ContextWindowExceeded('Vision input exceeds the confirmed serving window')
            attempt_id = self.ledger.begin(self.run_id, stage='vision', model=model,
                messages=kwargs['messages'], output_tokens=output, required_tokens=needed,
                policy_sha256=self.identity['sha256'],
                max_calls=None if budget.unlimited_llm_usage else budget.max_llm_calls,
                max_tokens=None if budget.unlimited_llm_usage else budget.max_tokens,
                protected_calls=self.runtime._protected_call_reserve(self.run_id, budget, 'general'),
                protected_tokens=self.runtime._protected_reserve(self.run_id, budget, 'general'))
            if on_dispatch:
                on_dispatch()
            started = time.perf_counter()
            self.db.add_context_snapshot(run_id=self.run_id, stage='vision.request',
                policy_version='qwen3-vl-still-image-v1', payload={'messages': kwargs['messages']},
                lineage={'attempt_id': attempt_id, 'model': model, 'substage': stage},
                estimated_tokens_before=needed, estimated_tokens_after=needed, compressed=False)
            try:
                payload = {**kwargs, 'model': model}
                from .model_policy import FREE_MODELS
                if FREE_MODELS.get(model, {}).get('thinking') is False:
                    payload.pop('extra_body', None)
                result = self.adapter.client.chat.completions.create(**payload)
            except Exception as error:
                failure = self.ledger.record_failure(attempt_id, model=model, error=error,
                    stage='vision', output_tokens=output, thinking_enabled=False,
                    duration_ms=(time.perf_counter() - started) * 1000)
                self.db.update_run_metadata(self.run_id, {'last_provider_failure': failure})
                raise
            usage = getattr(result, 'usage', None)
            known = (usage is not None and getattr(usage, 'prompt_tokens', None) is not None
                     and getattr(usage, 'completion_tokens', None) is not None)
            call_id = self.db.record_llm_call(run_id=self.run_id, stage='vision', attempt=1, status='COMPLETED',
                model=model, thinking_enabled=False, max_output_tokens=output,
                prompt_tokens=int(getattr(usage, 'prompt_tokens', 0) or 0),
                completion_tokens=int(getattr(usage, 'completion_tokens', 0) or 0),
                request_id=getattr(result, 'id', None),
                finish_reason=getattr(result.choices[0], 'finish_reason', None) if result.choices else None,
                duration_ms=(time.perf_counter() - started) * 1000)
            details = getattr(usage, 'prompt_tokens_details', None)
            self.ledger.settle(attempt_id, state='RETURNED', usage_known=known,
                cached_tokens_known=getattr(details, 'cached_tokens', None) is not None, llm_call_id=call_id)
            self.db.update_run_metadata(self.run_id, {'last_provider_failure': None})
            self.db.emit_event(self.run_id, 'llm.response.saved', {'attempt_id': attempt_id,
                'model': model, 'substage': stage,
                'raw_response': result.model_dump(mode='json') if hasattr(result, 'model_dump') else {
                    'content': getattr(result.choices[0].message, 'content', '') if result.choices else ''}})
            return result
        return self.policy.execute(send)
