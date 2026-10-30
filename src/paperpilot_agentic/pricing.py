from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

DEEPSEEK_PRICING_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
DEEPSEEK_PRICING_VERSION = "official-snapshot-2026-08-26"
_BEIJING = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class TokenPriceCNY:
    cached_input_per_million: float
    uncached_input_per_million: float
    output_per_million: float


@dataclass(frozen=True, slots=True)
class CostEstimate:
    amount_cny: float
    currency: str
    pricing_tier: str
    pricing_source: str
    pricing_version: str
    model_family: str
    rates_cny_per_million: TokenPriceCNY

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["amount_cny"] = round(self.amount_cny, 9)
        return value


_OFF_PEAK = {
    "deepseek-v4-flash": TokenPriceCNY(0.05, 1.5, 4.5),
    "deepseek-v4-pro": TokenPriceCNY(0.15, 4.5, 13.5),
    "deepseek-v4-flash-vision-exp": TokenPriceCNY(0.05, 1.5, 4.5),
}
_PEAK = {
    "deepseek-v4-flash": TokenPriceCNY(0.10, 3.0, 9.0),
    "deepseek-v4-pro": TokenPriceCNY(0.30, 9.0, 27.0),
    "deepseek-v4-flash-vision-exp": TokenPriceCNY(0.10, 3.0, 9.0),
}


def normalize_deepseek_model(model: str) -> str | None:
    value = str(model or "").strip().casefold()
    if value.startswith("deepseek-v4-flash-vision"):
        return "deepseek-v4-flash-vision-exp"
    if value.startswith("deepseek-v4-pro"):
        return "deepseek-v4-pro"
    if value.startswith("deepseek-v4-flash"):
        return "deepseek-v4-flash"
    return None


def deepseek_pricing_tier(at: datetime | None = None) -> str:
    current = at or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(_BEIJING)
    weekday = local.weekday() < 5
    clock = local.time().replace(tzinfo=None)
    peak = weekday and (
        time(9, 0) <= clock < time(12, 0)
        or time(14, 0) <= clock < time(18, 0)
    )
    return "peak" if peak else "off_peak"


def estimate_deepseek_cost_cny(
    *,
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    at: datetime | None = None,
    override_rates: TokenPriceCNY | None = None,
) -> CostEstimate | None:
    """Estimate one DeepSeek request in CNY from provider-reported usage.

    DeepSeek's ``completion_tokens`` already includes reasoning tokens.  The
    latter is recorded separately for observability but must not be charged a
    second time.
    """

    family = normalize_deepseek_model(model)
    if family is None and override_rates is None:
        return None
    tier = deepseek_pricing_tier(at)
    rates = override_rates or (_PEAK if tier == "peak" else _OFF_PEAK)[str(family)]
    prompt = max(0, int(prompt_tokens))
    cached = min(prompt, max(0, int(cached_tokens)))
    uncached = prompt - cached
    completion = max(0, int(completion_tokens))
    amount = (
        cached * rates.cached_input_per_million
        + uncached * rates.uncached_input_per_million
        + completion * rates.output_per_million
    ) / 1_000_000
    source = (
        "caller-supplied DeepSeek-compatible CNY rate override"
        if override_rates is not None
        else DEEPSEEK_PRICING_SOURCE
    )
    version = "runtime-rate-override" if override_rates is not None else DEEPSEEK_PRICING_VERSION
    return CostEstimate(
        amount_cny=amount,
        currency="CNY",
        pricing_tier=tier,
        pricing_source=source,
        pricing_version=version,
        model_family=family or str(model),
        rates_cny_per_million=rates,
    )
