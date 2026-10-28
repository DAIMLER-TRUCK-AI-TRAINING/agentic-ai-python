"""Part 2 — zero-shot vs few-shot vs chain-of-thought, on both providers, with the schema enforced."""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass
from string import Template

import pandas as pd
from pydantic import BaseModel, Field

from .common import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    active_providers,
    claude_client,
    openai_client,
    run_parallel,
    show,
    step,
)
from .part1_baseline import BASELINE_SCORE, EXAMPLES, FIELDS, TEST, Category, Group, Priority, TriageLabel, score


@dataclass(frozen=True)
class PromptTemplate:
    """A named, versioned prompt, rendered in one place."""

    name: str
    version: str
    system: str
    user: str = "<ticket>\n$ticket\n</ticket>"
    schema: type[BaseModel] = TriageLabel

    def render(self, ticket: str) -> tuple[str, str]:
        """Return (system prompt, user message) for one ticket."""
        return self.system, Template(self.user).substitute(ticket=ticket)


class TriageWithReasoning(BaseModel):
    reasoning: str = Field(description="Brief step-by-step reasoning, written BEFORE the labels")
    category: Category
    priority: Priority
    assignment_group: Group


ROLE = "You are an L1 service-desk analyst at a large enterprise. Triage the IT ticket inside <ticket> tags."
SHORT_GUIDE = (
    "Categories: access (accounts, passwords, MFA, permissions) · network (VPN, Wi-Fi, LAN) · endpoint (laptops, "
    "printers, peripherals, software on a device) · email_collab (email, calendars, collaboration tools) · "
    "business_app (ERP and finance applications) · platform (servers, storage, VMs, pipelines) · security "
    "(phishing, malware, lost devices).\n"
    "Priorities: P1 critical · P2 high · P3 medium · P4 low.\n"
    "Assignment groups: Service Desk · Network Ops · Finance · Platform Engineering · Security Operations."
)


def examples_block(frame: pd.DataFrame, *, with_reason: bool = False) -> str:
    """Render labelled examples as XML-tagged few-shot demonstrations."""
    items = []
    for r in frame.itertuples():
        label = json.dumps({f: getattr(r, f) for f in FIELDS})
        reason = f"\n<why>{r.rationale}</why>" if with_reason else ""
        items.append(f"<example>\n<ticket>{r.text}</ticket>\n<label>{label}</label>{reason}\n</example>")
    return "<examples>\n" + "\n".join(items) + "\n</examples>"


ZERO_SHOT = PromptTemplate("zero-shot", "v1", f"{ROLE}\n\n{SHORT_GUIDE}")
FEW_SHOT = PromptTemplate(
    "few-shot", "v1", f"{ROLE}\n\n{SHORT_GUIDE}\n\nLabelled examples from this service desk:\n{examples_block(EXAMPLES)}"
)
CHAIN_OF_THOUGHT = PromptTemplate(
    "chain-of-thought",
    "v1",
    f"{ROLE}\n\n{SHORT_GUIDE}\n\nThink before you label. Step 1: who or what is affected, and how many people? "
    "Step 2: has work stopped, is it degraded, or is this a request? Step 3: is there a security aspect? "
    "Write the steps briefly in `reasoning`, then give the labels.",
    schema=TriageWithReasoning,
)
V1 = [ZERO_SHOT, FEW_SHOT, CHAIN_OF_THOUGHT]


def classify(provider: str, template: PromptTemplate, ticket_id: str, text: str) -> dict:
    """One classification call, on either provider, with the template's schema enforced."""
    system, user = template.render(text)
    row: dict[str, object] = {"provider": provider, "technique": template.name, "version": template.version, "ticket_id": ticket_id}
    started = time.perf_counter()
    try:
        if provider == "openai":
            r = openai_client().responses.parse(
                model=OPENAI_MODEL, instructions=system, input=user, text_format=template.schema,
                reasoning={"effort": "low"}, max_output_tokens=4_000,
            )
            label, tokens = r.output_parsed, (r.usage.input_tokens, r.usage.output_tokens)
        else:
            m = claude_client().messages.parse(
                model=ANTHROPIC_MODEL, max_tokens=4_000, system=system, output_config={"effort": "low"},
                output_format=template.schema, messages=[{"role": "user", "content": user}],
            )
            label, tokens = m.parsed_output, (m.usage.input_tokens, m.usage.output_tokens)
        row |= {f: getattr(label, f) for f in FIELDS}
        row |= {"reasoning": getattr(label, "reasoning", None), "input_tokens": tokens[0], "output_tokens": tokens[1]}
    except Exception as exc:  # a failed call is a data point, not a crash
        row["error"] = f"{exc.__class__.__name__}: {exc}"[:200]
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per provider × technique × version: accuracy, errors, tokens and latency."""
    rows = [{"provider": "—", "technique": "keyword baseline", "version": "—", **BASELINE_SCORE}]
    for (provider, technique, version), g in runs.groupby(["provider", "technique", "version"], sort=False):
        errors = int(g["error"].notna().sum()) if "error" in g else 0
        rows.append({
            "provider": provider, "technique": technique, "version": version, **score(g), "errors": errors,
            "avg output tokens": round(g.output_tokens.mean()), "avg s": round(g.seconds.mean(), 1),
        })
    return pd.DataFrame(rows)


def run_templates(templates: list[PromptTemplate]) -> pd.DataFrame:
    """Every template × every test ticket × every configured provider, 8 calls at a time."""
    jobs = [(p, t, r.ticket_id, r.text) for p in active_providers() for t in templates for r in TEST.itertuples()]
    return pd.DataFrame(run_parallel(classify, jobs))


@functools.cache
def v1_runs() -> pd.DataFrame:
    """The v1 comparison, run once per process: Part 3 reuses it."""
    started = time.perf_counter()
    runs = run_templates(V1)
    if not runs.empty:
        print(f"{len(runs)} calls in {time.perf_counter() - started:.0f}s")
    return runs


def run() -> None:
    step("The few-shot template, rendered for the first test ticket")
    system, user = FEW_SHOT.render(TEST.text[0])
    print(system[:900], "…\n")
    print(user)

    step("3 techniques × 24 tickets on every configured provider")
    runs = v1_runs()
    if runs.empty:
        print("⚠️  No provider configured: skipping the comparison.")
        return
    show(summarise(runs), index=False)
    failed = runs["error"].notna().mean() if "error" in runs else 0.0
    print(f"\n{'✅' if failed <= 0.1 else '⚠️ '} {failed:.0%} of calls failed (the notebook's check allows up to 10%)")
    print("📝 Record the table in the lab guide: which technique helped most, on which field, at what cost?")
