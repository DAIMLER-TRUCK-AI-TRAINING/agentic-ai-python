"""Part 3 — count the tokens, time one real request per tier, and model the monthly cost."""

from __future__ import annotations

import functools
import time

import pandas as pd
import tiktoken

from .common import (
    ANTHROPIC_MODEL,
    ANTHROPIC_SMALL_MODEL,
    OPENAI_MODEL,
    OPENAI_SMALL_MODEL,
    claude_client,
    claude_text,
    configured,
    openai_client,
    show,
    step,
)
from .part1_scenarios import CATALOGUE, SCENARIOS

# Your infrastructure estimate for self-hosting, in USD/month (GPUs, ops time). None = not estimated yet.
SELF_HOSTED_MONTHLY_USD: dict[str, float | None] = {"gpt-oss-120b": None, "gpt-oss-20b": None}


@functools.cache
def o200k() -> tiktoken.Encoding | None:
    """An offline approximation of OpenAI's tokeniser, cached by labs/setup/prefetch_models.py."""
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # not cached and no network
        print(f"⚠️  o200k_base unavailable ({exc.__class__.__name__}); the offline column is skipped.")
        return None


def sample_tokens(scenario: dict) -> dict:
    """Tokens in the scenario's sample request (system + user), per available counter."""
    sample = scenario["sample_request"]
    row = {"scenario": scenario["id"], "planning avg input": scenario["avg_input_tokens"]}
    if (encoding := o200k()) is not None:
        row["o200k (offline)"] = len(encoding.encode(sample["system"] + sample["user"]))
    if configured("openai"):
        row["OpenAI count"] = openai_client().responses.input_tokens.count(
            model=OPENAI_MODEL, instructions=sample["system"], input=sample["user"]
        ).input_tokens
    if configured("anthropic"):
        row["Claude count"] = claude_client().messages.count_tokens(
            model=ANTHROPIC_MODEL, system=sample["system"], messages=[{"role": "user", "content": sample["user"]}]
        ).input_tokens
    return row


def tier_probe() -> pd.DataFrame:
    """Measure, don't guess: the UC-1 sample request on the main and the small tier of each provider."""
    uc1 = SCENARIOS["UC-1"]["sample_request"]
    rows = []
    targets = [
        ("openai", OPENAI_MODEL),
        ("openai", OPENAI_SMALL_MODEL),
        ("anthropic", ANTHROPIC_MODEL),
        ("anthropic", ANTHROPIC_SMALL_MODEL),
    ]
    for provider, model_id in targets:
        if not model_id or not configured(provider):
            continue
        started = time.perf_counter()
        if provider == "openai":
            r = openai_client().responses.create(
                model=model_id, instructions=uc1["system"], input=uc1["user"], reasoning={"effort": "low"}, max_output_tokens=1_000
            )
            text, tokens_in, tokens_out = r.output_text, r.usage.input_tokens, r.usage.output_tokens
        else:
            m = claude_client().messages.create(
                model=model_id,
                max_tokens=1_000,
                system=uc1["system"],
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": uc1["user"]}],
            )
            text, tokens_in, tokens_out = claude_text(m), m.usage.input_tokens, m.usage.output_tokens
        price = CATALOGUE.get(model_id, {})
        cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6 if price.get("input") is not None else None
        rows.append(
            {
                "model": model_id,
                "latency (s)": round(time.perf_counter() - started, 2),
                "in": tokens_in,
                "out": tokens_out,
                "cost per 1,000 requests ($)": round(cost * 1000, 3) if cost is not None else None,
                "answer": text.strip()[:60],
            }
        )
    return pd.DataFrame(rows)


def monthly_cost(
    model: dict,
    scenario: dict,
    *,
    cache_hit_rate: float = 0.0,
    batch: bool = False,
    days: int = 30,
    self_hosted_monthly_usd: float | None = None,
) -> float:
    """USD per month for `scenario` on `model`.

    1. Catalogue prices are USD per 1M tokens; a request costs avg input × input price + avg output × output price.
    2. A fraction `cache_hit_rate` of the input is billed at `cached_input` (cache-write premiums are ignored).
    3. Above a `long_context` threshold, the long-context prices apply to the whole request.
    4. `batch=True` halves the total (both providers' Batch APIs bill at 50%).
    5. Monthly = per-request cost × requests_per_day × days, rounded to cents.
    6. A model without token prices (open weights) costs `self_hosted_monthly_usd`; None raises ValueError.
    """
    if model["input"] is None:
        if self_hosted_monthly_usd is None:
            raise ValueError(f"{model['id']} has no token price: pass self_hosted_monthly_usd")
        return round(float(self_hosted_monthly_usd), 2)
    prices = model
    long_context = model.get("long_context")
    if long_context and scenario["avg_input_tokens"] > long_context["threshold_tokens"]:
        prices = long_context
    cached_price = prices["cached_input"] if prices.get("cached_input") is not None else prices["input"]
    tokens_in, tokens_out = scenario["avg_input_tokens"], scenario["avg_output_tokens"]
    per_request = (
        tokens_in * (1 - cache_hit_rate) * prices["input"]
        + tokens_in * cache_hit_rate * cached_price
        + tokens_out * prices["output"]
    ) / 1_000_000
    if batch:
        per_request *= 0.5
    return round(per_request * scenario["requests_per_day"] * days, 2)


def cost_or_none(model: dict, scenario: dict) -> float | None:
    """Monthly cost with the caching and batching the scenario allows; None for an unpriced open model."""
    try:
        return monthly_cost(
            model,
            scenario,
            cache_hit_rate=scenario["cacheable_prefix_share"],
            batch=scenario["batch_eligible"],
            self_hosted_monthly_usd=SELF_HOSTED_MONTHLY_USD.get(model["id"]),
        )
    except ValueError:
        return None


def cost_table() -> pd.DataFrame:
    """Every model × every scenario, in USD per month."""
    return pd.DataFrame({sid: {mid: cost_or_none(m, s) for mid, m in CATALOGUE.items()} for sid, s in SCENARIOS.items()})


def run() -> None:
    step("Tokens in each scenario's sample request, per counter")
    show(pd.DataFrame([sample_tokens(s) for s in SCENARIOS.values()]).set_index("scenario"))

    step("Measure, don't guess: the UC-1 request on each provider's main and small tier")
    probe = tier_probe()
    if probe.empty:
        print("⚠️  No provider configured: skipping the tier probe.")
    else:
        show(probe, index=False)

    step("Monthly cost, USD, with the caching and batching each scenario allows (— = no price)")
    show(cost_table().map(lambda v: "—" if v is None or pd.isna(v) else f"${v:,.0f}"))
