"""One conservative text-request window contract, independent of task spending.

UTF-8 bytes plus framing are a practical upper estimate for byte-fallback models,
not an exact provider tokenizer. Deployments must set their actual serving limit.
"""
from __future__ import annotations

import os


class ContextWindowExceeded(ValueError):
    """Caller must split/reload the task; never retry an unchanged oversized input."""


def prompt_upper_bound(messages):
    return 256 + sum(128 + sum(len(str(v).encode('utf-8')) for v in m.values()) for m in messages)


def window_limit(model=None):
    value = getattr(model, 'context_window_tokens', None)
    limit = int(value if value is not None else os.getenv('PAPERPILOT_CONTEXT_WINDOW_TOKENS', '131072'))
    if limit < 4096:
        raise ValueError('Context window must be at least 4096 tokens')
    return limit


def enforce_window(messages, output_tokens, *, limit):
    needed = prompt_upper_bound(messages) + int(output_tokens) + 1024
    if output_tokens <= 0 or needed > limit:
        raise ContextWindowExceeded(
            f'Text context requires upper estimate {needed}, serving window {limit}; '
            'split material into persisted batches or reload a smaller evidence workset')
    return needed


def split_text(text, max_bytes):
    """Lossless Unicode slices with character offsets; never drop a long passage."""
    if max_bytes < 4:
        raise ValueError('Text slice budget too small')
    start, used = 0, 0
    for i, char in enumerate(text):
        size = len(char.encode('utf-8'))
        if used + size > max_bytes:
            yield start, i, text[start:i]
            start, used = i, 0
        used += size
    if start < len(text):
        yield start, len(text), text[start:]
