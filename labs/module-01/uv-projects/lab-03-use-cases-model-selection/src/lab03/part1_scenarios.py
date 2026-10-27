"""Part 1 — frame the four enterprise IT scenarios, the agent types and the model catalogue."""

from __future__ import annotations

import json

import pandas as pd

from .common import DATA_DIR, show, step

SCENARIOS: dict[str, dict] = {
    s["id"]: s for s in json.loads((DATA_DIR / "use_case_scenarios.json").read_text(encoding="utf-8"))
}
CATALOGUE_DOC = json.loads((DATA_DIR / "model_catalogue.json").read_text(encoding="utf-8"))
CATALOGUE: dict[str, dict] = {m["id"]: m for m in CATALOGUE_DOC["models"]}

# The agent types you choose from (Module 2 builds the first four; Module 4 the last).
AGENT_TYPES = {
    "single-call classifier": "One request, one structured answer: routing, triage, tagging. No loop, no tools.",
    "extraction pipeline": "A fixed workflow (OCR → LLM extraction → validation) your code orchestrates. Batchable.",
    "RAG assistant": "Retrieve relevant passages, answer from them, cite them. Conversational.",
    "tool-using agent": "The model decides which tools to call in a loop (ReAct); a human approves risky actions.",
    "multi-agent orchestration": "A supervisor routes work to specialist agents; for multi-domain, multi-step work.",
}


def run() -> None:
    step(f"{len(SCENARIOS)} scenarios · {len(CATALOGUE)} models · prices as of {CATALOGUE_DOC['prices_as_of']}")
    show(
        pd.DataFrame(
            [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "team": s["team"],
                    "requests/day": s["requests_per_day"],
                    "avg in/out tokens": f"{s['avg_input_tokens']:,} / {s['avg_output_tokens']:,}",
                    "max input": f"{s['max_input_tokens']:,}",
                    "latency SLO (s)": s["latency_slo_seconds"],
                    "tools": s["needs_tools"],
                    "documents": s["needs_documents"],
                    "batch OK": s["batch_eligible"],
                    "residency": s["residency"],
                }
                for s in SCENARIOS.values()
            ]
        ),
        index=False,
        max_width=40,
    )

    step("The agent types")
    show(pd.Series(AGENT_TYPES, name="what it is").to_frame())

    step("The model catalogue (USD per 1M tokens)")
    show(
        pd.DataFrame(CATALOGUE.values())[
            ["id", "provider", "family", "tier", "deployment", "input", "output", "context_window", "vision"]
        ],
        index=False,
    )
