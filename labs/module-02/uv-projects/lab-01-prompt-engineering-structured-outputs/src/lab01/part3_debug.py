"""Part 3 — error analysis, a v2 prompt with the policy, a regression check, token counts and prompt caching."""

from __future__ import annotations

import time

import pandas as pd

from .common import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    active_providers,
    claude_client,
    openai_client,
    show,
    step,
)
from .part1_baseline import EXAMPLES, FIELDS, POLICY, TEST
from .part2_techniques import (
    ROLE,
    V1,
    PromptTemplate,
    examples_block,
    run_templates,
    summarise,
    v1_runs,
)

# v2: the fix the error analysis points to. Give the model the policy it is graded against.
ZERO_SHOT_V2 = PromptTemplate("zero-shot", "v2", f"{ROLE}\n\nApply this policy exactly:\n<policy>\n{POLICY}\n</policy>")
FEW_SHOT_V2 = PromptTemplate(
    "few-shot",
    "v2",
    f"{ZERO_SHOT_V2.system}\n\nLabelled examples, with the reason for each label:\n{examples_block(EXAMPLES, with_reason=True)}",
)
V2 = [ZERO_SHOT_V2, FEW_SHOT_V2]


def wrong_fields(runs: pd.DataFrame) -> pd.DataFrame:
    """The runs joined to the gold labels, with one <field>_wrong column per field."""
    merged = TEST.merge(runs, on="ticket_id", suffixes=("", "_pred"))
    for f in FIELDS:
        merged[f"{f}_wrong"] = merged[f] != merged[f + "_pred"]
    return merged


def hardest_tickets(runs: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """The tickets most runs got wrong, with the gold rationale next to them."""
    wf = wrong_fields(runs)
    return (
        wf.groupby("ticket_id")[[f"{f}_wrong" for f in FIELDS]].sum()
        .assign(total=lambda d: d.sum(axis=1)).sort_values("total", ascending=False).head(n)
        .join(TEST.set_index("ticket_id")[["text", "priority", "rationale"]])
    )


def regressions(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Tickets that were fully right before and are wrong after, per provider and technique."""
    keys = ["provider", "technique", "ticket_id"]
    b = wrong_fields(before).assign(ok=lambda d: ~d[[f"{f}_wrong" for f in FIELDS]].any(axis=1))[keys + ["ok"]]
    a = wrong_fields(after).assign(ok=lambda d: ~d[[f"{f}_wrong" for f in FIELDS]].any(axis=1))[keys + ["ok"]]
    both = b.merge(a, on=keys, suffixes=("_v1", "_v2"))
    return both[both.ok_v1 & ~both.ok_v2]


def prompt_tokens(provider: str, template: PromptTemplate) -> int:
    """Prompt tokens for one ticket, counted by the provider's own endpoint."""
    system, user = template.render(TEST.text[0])
    if provider == "openai":
        return openai_client().responses.input_tokens.count(model=OPENAI_MODEL, instructions=system, input=user).input_tokens
    return claude_client().messages.count_tokens(
        model=ANTHROPIC_MODEL, system=system, messages=[{"role": "user", "content": user}]
    ).input_tokens


def token_table(providers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"template": f"{t.name} {t.version}", **{p: prompt_tokens(p, t) for p in providers}} for t in V1 + V2]
    ).set_index("template")


def cached_call(provider: str, template: PromptTemplate, text: str) -> dict:
    """One call with the system prompt cached: automatic on OpenAI, an explicit breakpoint on Claude."""
    system, user = template.render(text)
    if provider == "openai":  # automatic for prompts of 1,024+ tokens; the key routes requests to the same cache
        r = openai_client().responses.parse(
            model=OPENAI_MODEL, instructions=system, input=user, text_format=template.schema,
            reasoning={"effort": "low"}, max_output_tokens=4_000, prompt_cache_key="lab01-few-shot-v2",
        )
        u = r.usage
        return {"provider": provider, "input total": u.input_tokens, "read from cache": u.input_tokens_details.cached_tokens}
    m = claude_client().messages.parse(  # explicit breakpoint on the system block
        model=ANTHROPIC_MODEL, max_tokens=4_000, output_config={"effort": "low"}, output_format=template.schema,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    u = m.usage
    return {
        "provider": provider,
        "input total": u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens,
        "written to cache": u.cache_creation_input_tokens,
        "read from cache": u.cache_read_input_tokens,
    }


def run() -> None:
    providers = active_providers()
    if not providers:
        print("⚠️  No provider configured: this part needs model runs to analyse.")
        return
    step("Error analysis: which tickets do most v1 runs get wrong, and on which field?")
    runs_v1 = v1_runs()
    print(f"Wrong-label counts out of {len(providers) * len(V1)} runs per ticket:")
    show(hardest_tickets(runs_v1), max_width=60)
    wf = wrong_fields(runs_v1)
    print("\nPriority confusion (rows gold, columns predicted):")
    print(pd.crosstab(wf.priority, wf.priority_pred, rownames=["gold"], colnames=["predicted"]).to_string())

    step("v2: give the model the policy it is graded against")
    runs_v2 = run_templates(V2)
    show(summarise(pd.concat([runs_v1, runs_v2])), index=False)

    step("Regression check: a fix that breaks other tickets is not a fix")
    broken = regressions(runs_v1, runs_v2)
    print(f"{len(broken)} regression(s) from v1 to v2")
    show(broken, index=False)

    step("Price each template: prompt tokens per ticket, counted by each provider")
    tokens = token_table(providers)
    show(tokens)
    ok = (tokens.loc["few-shot v2"] > tokens.loc["zero-shot v1"]).all()
    print(f"{'✅' if ok else '⚠️ '} few-shot v2 costs more input tokens than zero-shot v1")

    step("Prompt caching: the few-shot v2 system prompt is identical for every ticket")
    rows = []
    for provider in providers:
        for i, text in enumerate(TEST.text[:4]):
            rows.append({"call": i + 1, **cached_call(provider, FEW_SHOT_V2, text)})
            time.sleep(2)  # a cache entry becomes readable shortly after the request that wrote it
    show(pd.DataFrame(rows).fillna(0), index=False)
    print("OpenAI caching is best-effort: if 'read from cache' stays 0, run the step again before concluding it is broken.")
