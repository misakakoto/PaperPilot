"""Lossless request-local aliases for long citation IDs, never fuzzy correction."""
from __future__ import annotations

import json
from typing import Any


class CitationAliases:
    def __init__(self, identifiers):
        originals = set(identifiers)
        self.encode_map = {}
        index = 1
        for identifier in sorted(originals):
            if len(identifier) <= 24:
                continue
            while f"ref{index}" in originals:
                index += 1
            self.encode_map[identifier] = f"ref{index}"
            index += 1
        self.decode_map = {v: k for k, v in self.encode_map.items()}

    def transform(self, value: Any, *, decode=False, field=None):
        mapping = self.decode_map if decode else self.encode_map
        if isinstance(value, dict):
            return {k: self.transform(v, decode=decode, field=k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.transform(v, decode=decode, field=field) for v in value]
        if isinstance(value, str) and field in {"evidence_id", "evidence_ids"}:
            return mapping.get(value, value)
        return value

    def messages(self, messages):
        result = []
        for message in messages:
            item = dict(message)
            if item.get("role") != "system":
                try:
                    value = json.loads(item["content"])
                except (ValueError, TypeError):
                    pass  # Preserve malformed repair output and ordinary instructions.
                else:
                    item["content"] = json.dumps(self.transform(value), ensure_ascii=False, separators=(',', ':'))
            result.append(item)
        return result
