"""Part 4 — context-window edges: limits, token counts, truncation, long prompts and a pre-flight guard."""

from __future__ import annotations

import functools
import json
import random
import re

import anthropic
import openai
import pandas as pd
import tiktoken

from .common import (
    ANTHROPIC_MODEL,
    DATA_DIR,
    OPENAI_MODEL,
    active_providers,
    available,
    claude_client,
    claude_text,
    configured,
    openai_client,
    show,
    step,
)
from .part1_first_calls import ASK

HAYSTACK_LINES = 600  # about 15K tokens per prompt; lower it if your budget is tight
NEEDLE = (
    "2026-09-14 02:00 CHG-20931 approved by the change advisory board: core-switch firmware upgrade, "
    "maintenance window 02:00-04:00 UTC Sunday."
)
TEAMS = ["Service Desk", "Network Ops", "Platform Engineering", "Finance", "Legal"]
EVENTS = [
    "password reset completed", "VPN tunnel flapped", "disk usage above 85%", "printer queue cleared",
    "certificate renewed", "backup job finished", "MFA push timed out", "patch window opened",
]
LONG_ASK = "Explain, in about 300 words, how a VPN gateway authenticates a remote user."
NEEDLE_QUESTION = (
    "Which change request did the change advisory board approve, and what is its maintenance window? "
    "Answer in one line."
)
COUNT_QUESTION = "How many P1 incidents for [Network Ops] are in the log? Reply with the number only."


@functools.cache
def catalogue() -> dict[str, dict]:
    return {m["id"]: m for m in json.loads((DATA_DIR / "model_catalogue.json").read_text())["models"]}


@functools.cache
def o200k() -> tiktoken.Encoding | None:
    """An offline approximation of OpenAI's tokeniser, cached by labs/setup/prefetch_models.py."""
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # not cached and no network
        print(f"⚠️  o200k_base unavailable ({exc.__class__.__name__}); offline token estimates are skipped.")
        return None


def count_openai(text: str) -> int:
    return openai_client().responses.input_tokens.count(model=OPENAI_MODEL, input=text).input_tokens


def count_claude(text: str) -> int:
    return claude_client().messages.count_tokens(
        model=ANTHROPIC_MODEL, messages=[{"role": "user", "content": text}]
    ).input_tokens


def model_limits() -> None:
    step("The limits: Claude reports them through the Models API; OpenAI's come from its docs")
    rows = []
    if available("anthropic"):
        info = claude_client().models.retrieve(ANTHROPIC_MODEL)
        rows.append(
            {"model": info.id, "source": "Models API", "max input tokens": info.max_input_tokens, "max output tokens": info.max_tokens}
        )
    if OPENAI_MODEL:
        entry = catalogue().get(OPENAI_MODEL)
        rows.append(
            {
                "model": OPENAI_MODEL,
                "source": "docs, via data/model_catalogue.json" if entry else "unknown: add it to the catalogue",
                "max input tokens": entry["max_input_tokens"] if entry else None,
                "max output tokens": entry["max_output_tokens"] if entry else None,
            }
        )
    show(pd.DataFrame(rows), index=False)


def token_table() -> pd.DataFrame:
    """The same ticket in four languages: characters versus tokens, per counter."""
    tickets = pd.read_csv(DATA_DIR / "multilingual_tickets.csv")
    rows = []
    for ticket in tickets[tickets.gold_kb == "KB-0042"].itertuples():
        row = {"language": ticket.language, "characters": len(ticket.text)}
        if (encoding := o200k()) is not None:
            row["o200k_base (offline)"] = len(encoding.encode(ticket.text))
        if configured("openai"):
            row["OpenAI count API"] = count_openai(ticket.text)
        if configured("anthropic"):
            row["Claude count API"] = count_claude(ticket.text)
        rows.append(row)
    return pd.DataFrame(rows).set_index("language")


def token_counts() -> None:
    step("The same ticket in four languages: characters versus tokens")
    if OPENAI_MODEL:
        try:
            print("tiktoken maps", OPENAI_MODEL, "to", tiktoken.encoding_for_model(OPENAI_MODEL).name)
        except KeyError:
            print(f"tiktoken does not know {OPENAI_MODEL}'s tokeniser → use the provider's count endpoint.")
    show(token_table())


def truncation() -> None:
    step("The output budget runs out: reasoning/thinking tokens share the cap with visible text")
    rows = []
    if available("openai"):
        try:
            r = openai_client().responses.create(model=OPENAI_MODEL, input=LONG_ASK, max_output_tokens=64)
            rows.append(
                {
                    "provider": "openai",
                    "finished": f"status={r.status} · reason={r.incomplete_details.reason if r.incomplete_details else None}",
                    "output tokens": r.usage.output_tokens,
                    "of which reasoning": r.usage.output_tokens_details.reasoning_tokens,
                    "visible characters": len(r.output_text),
                }
            )
        except openai.BadRequestError as exc:
            print("OpenAI rejected the request:", exc.message)
    if available("anthropic"):
        try:
            m = claude_client().messages.create(
                model=ANTHROPIC_MODEL, max_tokens=64, messages=[{"role": "user", "content": LONG_ASK}]
            )
            details = m.usage.output_tokens_details
            rows.append(
                {
                    "provider": "anthropic",
                    "finished": f"stop_reason={m.stop_reason} · blocks={[b.type for b in m.content]}",
                    "output tokens": m.usage.output_tokens,
                    "of which reasoning": details.thinking_tokens if details else None,
                    "visible characters": len(claude_text(m)),
                }
            )
        except anthropic.BadRequestError as exc:
            print("Claude rejected the request:", exc.message)
    show(pd.DataFrame(rows), index=False)


def build_ops_log(lines: int, needle: str, position: float, seed: int = 42) -> tuple[str, dict[str, int]]:
    """Return (log text, P1 count per team) with `needle` inserted at `position` (0.0-1.0)."""
    rng = random.Random(seed)
    rows, p1_by_team = [], {team: 0 for team in TEAMS}
    for i in range(lines):
        team = rng.choice(TEAMS)
        priority = rng.choices(["P1", "P2", "P3", "P4"], weights=[1, 3, 8, 8])[0]
        p1_by_team[team] += priority == "P1"
        day = 1 + i * 29 // lines
        stamp = f"2026-09-{day:02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
        rows.append(f"{stamp} INC-{105000 + i} {priority} [{team}] {rng.choice(EVENTS)}")
    rows.insert(int(position * lines), needle)
    return "\n".join(rows), p1_by_team


def long_prompts(providers: list[str]) -> str:
    """Needle retrieval at three positions, then a counting question; returns the 50% log."""
    sample_log, p1_by_team = build_ops_log(HAYSTACK_LINES, NEEDLE, 0.5)
    encoding = o200k()
    approx_tokens = len(encoding.encode(sample_log)) if encoding is not None else len(sample_log) // 4
    step("A synthetic operations log with one 'needle' line")
    print(sample_log.splitlines()[0], "\n…")
    print(f"{HAYSTACK_LINES} lines ≈ {approx_tokens:,} tokens per prompt (o200k_base estimate)")
    for model_id in (OPENAI_MODEL, ANTHROPIC_MODEL):
        if model_id in catalogue() and catalogue()[model_id]["input"]:
            cost = 4 * approx_tokens * catalogue()[model_id]["input"] / 1_000_000
            print(f"  the next step sends 4 such prompts to {model_id}: ≈ ${cost:.2f} of input")
    if not providers:
        return sample_log

    step("Needle retrieval at 5%, 50% and 95% of the log")
    needle_results = []
    for position in (0.05, 0.5, 0.95):
        log, _ = build_ops_log(HAYSTACK_LINES, NEEDLE, position)
        prompt = f"<ops_log>\n{log}\n</ops_log>\n\n{NEEDLE_QUESTION}"
        for provider in providers:
            answer = ASK[provider](prompt)
            found = all(token in answer for token in ("CHG-20931", "02:00", "04:00"))
            needle_results.append({"position": position, "provider": provider, "found": found, "answer": answer[:90]})
    show(pd.DataFrame(needle_results), index=False)

    step("Counting across the whole log")
    truth = p1_by_team["Network Ops"]
    counted_by_code = sum(" P1 [Network Ops] " in line for line in sample_log.splitlines())
    print(f"Ground truth from code: {truth} P1 incidents for Network Ops (a one-line scan finds {counted_by_code})")
    counting = []
    for provider in providers:
        answer = ASK[provider](f"<ops_log>\n{sample_log}\n</ops_log>\n\n{COUNT_QUESTION}")
        number = re.search(r"\d+", answer)
        counting.append({"provider": provider, "answer": answer[:30], "correct": bool(number) and int(number.group()) == truth})
    show(pd.DataFrame(counting), index=False)
    return sample_log


class ContextBudgetError(RuntimeError):
    """The prompt plus the reserved output would exceed the context budget."""


def count_tokens(provider: str, text: str) -> int:
    """Count with the provider's own endpoint; fall back to o200k_base offline."""
    if provider == "openai" and configured("openai"):
        return count_openai(text)
    if provider == "anthropic" and configured("anthropic"):
        return count_claude(text)
    if (encoding := o200k()) is not None:
        return len(encoding.encode(text))
    raise RuntimeError("no token counter available")


def preflight(provider: str, prompt: str, *, max_output_tokens: int, budget_tokens: int) -> int:
    """Return the tokens the request needs, or raise ContextBudgetError if it would not fit."""
    needed = count_tokens(provider, prompt) + max_output_tokens
    if needed > budget_tokens:
        raise ContextBudgetError(
            f"{provider}: needs {needed:,} tokens (prompt + {max_output_tokens:,} reserved for output) "
            f"but the budget is {budget_tokens:,}"
        )
    return needed


def preflight_demo(providers: list[str], prompt: str) -> None:
    # An 8K budget stands in for a small model or a cost cap; the real windows are 1M tokens.
    step("A pre-flight guard: count first, refuse before sending (8,000-token budget)")
    counters = providers or (["offline"] if o200k() is not None else [])
    if not counters:
        print("⚠️  No token counter available (no provider, no cached o200k_base): skipping the guard demo.")
    for provider in counters:
        try:
            needed = preflight(provider, prompt, max_output_tokens=2_000, budget_tokens=8_000)
            print(f"{provider}: fits ({needed:,} tokens)")
        except ContextBudgetError as exc:
            print("🛑", exc)


def run() -> None:
    providers = active_providers()
    model_limits()
    token_counts()
    truncation()
    sample_log = long_prompts(providers)
    preflight_demo(providers, sample_log)
