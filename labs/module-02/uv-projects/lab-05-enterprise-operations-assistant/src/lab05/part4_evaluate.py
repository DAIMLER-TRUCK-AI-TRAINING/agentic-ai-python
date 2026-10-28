"""Part 4 — every case through every architecture on both providers, graded on labels, never on wording."""

from __future__ import annotations

import functools
import json
import time

import pandas as pd

from .common import DATA_DIR, MODELS, run_parallel, show, step
from .ops import EscalationRecord
from .part1_brief import CASE_BY_ID, CASES
from .part2_build import ARCHITECTURES, Result, handle, models_by_provider

PRICES = json.loads((DATA_DIR / "model_prices.json").read_text(encoding="utf-8"))["models"]


def grade(rec: EscalationRecord | None, case: dict) -> dict:
    """Structure and labels against the case's expected outcome."""
    e = case["expected"]
    if rec is None:
        return {"valid": False, "priority": False, "group": False, "escalate": False, "kb": False, "all": False}
    g = {
        "valid": True,
        "priority": rec.priority == e["priority"],
        "group": rec.assignment_group == e["assignment_group"],
        "escalate": rec.escalate == e["escalate"],
        "kb": not e["kb_any_of"] or bool(set(rec.kb_ids) & set(e["kb_any_of"])),
    }
    return {**g, "all": all(g.values())}


def cost(r: Result) -> float | None:
    price = PRICES.get(MODELS[r.provider])
    return (r.input_tokens * price["input"] + r.output_tokens * price["output"]) / 1e6 if price else None


def evaluation_rows(results: list[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        final, raw = grade(r.record, CASE_BY_ID[r.case_id]), grade(r.raw, CASE_BY_ID[r.case_id])
        rows.append({
            "provider": r.provider, "architecture": r.architecture, "case": r.case_id, **final, "raw all": raw["all"],
            "corrections": sum(not n.startswith("warning") for n in r.notes), "notes": "; ".join(r.notes),
            "seconds": r.seconds, "input tokens": r.input_tokens, "model calls": r.model_calls,
            "cost ($)": cost(r), "error": r.error,
        })
    return pd.DataFrame(rows)


def summarise(evaluation: pd.DataFrame) -> pd.DataFrame:
    """One row per provider × architecture: accuracy per field, the policy layer's effect, latency and cost."""
    return evaluation.groupby(["provider", "architecture"]).agg(
        valid=("valid", "mean"), priority=("priority", "mean"), group=("group", "mean"), escalate=("escalate", "mean"),
        kb=("kb", "mean"), all_correct=("all", "mean"), raw_all_correct=("raw all", "mean"), corrections=("corrections", "sum"),
        avg_s=("seconds", "mean"), avg_input_tokens=("input tokens", "mean"), cost_per_ticket=("cost ($)", "mean"),
    ).round(3)


def escalation_recall(evaluation: pd.DataFrame) -> float:
    """Of the runs whose case must escalate, the share that did."""
    must = evaluation[[CASE_BY_ID[c]["expected"]["escalate"] for c in evaluation.case]]
    return float(must.escalate.mean())


@functools.cache
def evaluate(architectures: tuple[str, ...] = tuple(ARCHITECTURES)) -> pd.DataFrame:
    """8 cases × the architectures × every configured provider, run once per process: Part 5 reuses it."""
    jobs = [(p, a, c) for p in models_by_provider() for a in architectures for c in CASES]
    started = time.perf_counter()
    results = run_parallel(handle, jobs, max_workers=6)
    if results:
        print(f"{len(results)} runs in {time.perf_counter() - started:.0f}s")
    return evaluation_rows(results)


def run(architectures: list[str] = ARCHITECTURES) -> None:
    step(f"{len(CASES)} cases × {', '.join(architectures)} × {', '.join(models_by_provider()) or 'no provider'}")
    evaluation = evaluate(tuple(architectures))
    if evaluation.empty:
        print("⚠️  No provider configured: skipping the evaluation.")
        return
    show(summarise(evaluation))

    step("The runs that missed, and what the policy layer changed")
    misses = evaluation[~evaluation["all"]][["provider", "architecture", "case", "priority", "group", "escalate", "kb", "notes", "error"]]
    show(misses.fillna("—"), index=False, max_width=70)

    expected_runs = len(models_by_provider()) * len(architectures) * len(CASES)
    print(f"\n{'✅' if len(evaluation) == expected_runs else '❌'} {len(evaluation)} of {expected_runs} runs recorded")
    print(f"{'✅' if evaluation.valid.mean() >= 0.9 else '❌'} {evaluation.valid.mean():.0%} of runs returned a valid record (the bar is 90%)")
    print(f"   escalation recall over all runs: {escalation_recall(evaluation):.0%} (the spec asks for 100%)")
    print("\n📝 Decision: which architecture and provider would you ship? Cite accuracy, recall, corrections, latency and cost.")
