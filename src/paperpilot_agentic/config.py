"""Small, provider-aware configuration layer for the slim PaperPilot runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from .errors import ConfigurationError

ProviderName = Literal["qwen", "deepseek"]

_DEEPSEEK_KEY_NAMES = {"DEEPSEEK_API_KEY", "LLM_API_KEY", "API_KEY"}
_DASHSCOPE_KEY_NAMES = {"DASHSCOPE_API_KEY", "QWEN_API_KEY"}
_DEEPSEEK_KEY_LABELS = {"deepseek", "ds"}
_DASHSCOPE_KEY_LABELS = {"dashscope", "qwen"}


def read_api_key_file(path: Path) -> str:
    return _read_provider_api_key(path, provider="deepseek")


def read_dashscope_api_key_file(path: Path) -> str:
    return _read_provider_api_key(path, provider="qwen")


def resolve_provider_api_key(provider: ProviderName) -> str | None:
    """Resolve one provider from environment or an external secret bundle."""

    if provider == "qwen":
        key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        key_file = os.getenv("PAPERPILOT_QWEN_API_KEY_FILE") or os.getenv(
            "PAPERPILOT_API_KEY_FILE"
        )
        reader = read_dashscope_api_key_file
    else:
        key = os.getenv("DEEPSEEK_API_KEY")
        key_file = os.getenv("PAPERPILOT_DEEPSEEK_API_KEY_FILE") or os.getenv(
            "PAPERPILOT_API_KEY_FILE"
        )
        reader = read_api_key_file
    if not key and key_file:
        key = reader(Path(key_file))
    normalized = str(key or "").strip()
    return normalized or None


def _read_provider_api_key(path: Path, *, provider: ProviderName) -> str:
    """Resolve one provider without logging secret contents.

    Named ``KEY=value`` entries are preferred. A legacy raw two-line bundle maps
    line one to DeepSeek and line two to DashScope.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"API key file does not exist: {resolved}")
    text = resolved.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ConfigurationError(f"API key file is empty: {resolved}")

    key_names = _DEEPSEEK_KEY_NAMES if provider == "deepseek" else _DASHSCOPE_KEY_NAMES
    labels = _DEEPSEEK_KEY_LABELS if provider == "deepseek" else _DASHSCOPE_KEY_LABELS
    raw_values: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        if "=" in candidate:
            name, value = candidate.split("=", 1)
            if name.strip() in key_names and value.strip():
                return value.strip().strip("\"'")
            continue
        if ":" in candidate and not candidate.casefold().startswith(("http://", "https://")):
            label, value = candidate.split(":", 1)
            if label.strip().casefold() in labels and value.strip():
                return value.strip().strip("\"'")
            continue
        raw_values.append(candidate.strip("\"'"))

    raw_index = 0 if provider == "deepseek" else 1
    if len(raw_values) > raw_index and raw_values[raw_index]:
        return raw_values[raw_index]
    expected = "DEEPSEEK_API_KEY" if provider == "deepseek" else "DASHSCOPE_API_KEY"
    raise ConfigurationError(f"API key file does not contain {expected}")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    provider: ProviderName
    api_key: str | None
    base_url: str
    model: str
    thinking_enabled: bool
    temperature: float
    max_tokens: int
    timeout_seconds: float
    workspace: Path
    runtime_db: Path
    memory_db: Path | None
    memory_namespace: str
    paper_roots: tuple[Path, ...]
    fallback_models: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, *, require_api_key: bool = False) -> "RuntimeSettings":
        load_dotenv(override=False)
        provider = os.getenv("PAPERPILOT_LLM_PROVIDER", "qwen").strip().casefold()
        if provider not in {"qwen", "deepseek"}:
            raise ConfigurationError("PAPERPILOT_LLM_PROVIDER must be qwen or deepseek")

        workspace = Path(os.getenv("PAPERPILOT_WORKSPACE", ".paperpilot")).expanduser().resolve()
        runtime_db = Path(
            os.getenv("PAPERPILOT_AGENTIC_DB", str(workspace / "runtime.sqlite3"))
        ).expanduser().resolve()
        memory_value = os.getenv("PAPERPILOT_MEMORY_DB", "").strip()
        memory_db = Path(memory_value).expanduser().resolve() if memory_value else None
        roots = tuple(
            Path(value.strip()).expanduser().resolve()
            for value in os.getenv("PAPERPILOT_PAPER_ROOTS", str(workspace / "papers")).split(os.pathsep)
            if value.strip()
        )

        if provider == "qwen":
            from .model_policy import DAILY_MODEL
            key = resolve_provider_api_key("qwen")
            base_url = os.getenv(
                "PAPERPILOT_LLM_BASE_URL",
                os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
            model = os.getenv(
                "PAPERPILOT_LLM_MODEL",
                os.getenv("QWEN_MODEL", DAILY_MODEL),
            )
        else:
            key = resolve_provider_api_key("deepseek")
            base_url = os.getenv(
                "PAPERPILOT_LLM_BASE_URL",
                os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
            model = os.getenv(
                "PAPERPILOT_LLM_MODEL",
                os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            )
        if require_api_key and not str(key or "").strip():
            raise ConfigurationError(f"{provider} API key is not configured")

        return cls(
            provider=provider,  # type: ignore[arg-type]
            api_key=str(key).strip() if key else None,
            base_url=base_url.rstrip("/"),
            model=model,
            thinking_enabled=os.getenv("PAPERPILOT_LLM_THINKING", "true").casefold()
            in {"1", "true", "yes", "on"},
            temperature=float(os.getenv("PAPERPILOT_LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("PAPERPILOT_LLM_MAX_TOKENS", "8000")),
            timeout_seconds=float(os.getenv("PAPERPILOT_LLM_TIMEOUT", "120")),
            workspace=workspace,
            runtime_db=runtime_db,
            memory_db=memory_db,
            memory_namespace=os.getenv("PAPERPILOT_MEMORY_NAMESPACE", "default"),
            paper_roots=roots,
            fallback_models=tuple(name.strip() for name in os.getenv(
                'PAPERPILOT_LLM_FALLBACK_MODELS',
                'qwen3-vl-32b-instruct' if provider == 'qwen' and model == 'qwen3-vl-30b-a3b-instruct' else ''
            ).split(',') if name.strip()),
        )
