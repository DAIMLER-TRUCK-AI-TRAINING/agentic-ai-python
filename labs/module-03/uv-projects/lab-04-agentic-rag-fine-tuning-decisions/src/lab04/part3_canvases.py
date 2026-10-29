"""Part 3 — the fine-tuning decision workshop: one canvas per scenario card (the reference decisions)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .common import check, show, step

DECISIONS = {"prompt engineering", "RAG", "fine-tuning", "hybrid"}


@dataclass
class Canvas:
    scenario: str
    gap: str              # "knowledge" or "behaviour"
    decision: str         # one of DECISIONS
    why: str              # data availability, latency, cost, update frequency
    overturn_eval: str    # the evaluation result that would make you change the decision


CANVASES = [
    Canvas("A · Workshop repair assistant", "knowledge", "RAG",
           "3,000 bulletins and guides change weekly and technicians must see the source; re-indexing takes minutes, "
           "retraining weeks, and weights cannot cite or forget a withdrawn bulletin.",
           "Recall@4 below 0.9 on 300 labelled workshop questions after hybrid retrieval and re-ranking."),
    Canvas("B · Warranty claim coding", "behaviour", "fine-tuning",
           "A fixed set of 380 damage codes, 200,000 labelled historical claims, 25,000 claims a month and a 300 ms "
           "budget on dealer systems: a small fine-tuned open-weight classifier beats a large model on cost and "
           "latency, after a few-shot baseline on the small tier sets the bar.",
           "The few-shot small-tier baseline reaches the target accuracy and p95 latency on a held-out month of claims."),
    Canvas("C · Goodwill decision letters", "behaviour", "prompt engineering",
           "Tone, structure and the 'goodwill is not warranty' rule fit in a template with three examples per language; "
           "the facts come from the case; 40 letters a day make cost irrelevant.",
           "Reviewers reject more than 10% of letters for tone in two languages after two prompt iterations."),
    Canvas("D · Engineering specification assistant", "knowledge", "hybrid",
           "Specs change weekly and need exact values with citations (RAG), but part codes and German-English jargon "
           "defeat dense retrieval: hybrid BM25 + vector retrieval first, and a RAFT-style fine-tune only if retrieval "
           "is right but answers still misread the domain.",
           "With the right passages retrieved, answer accuracy on 400 engineer Q&A pairs is already above 95%: then skip the fine-tune."),
]


def check_part3(canvases: list[Canvas] = CANVASES):
    assert len(canvases) == 4, "one canvas per scenario card"
    for c in canvases:
        assert "TODO" not in asdict(c).values(), f"{c.scenario}: fill in every field"
        assert c.gap in {"knowledge", "behaviour"} and c.decision in DECISIONS, c.scenario
        assert len(c.overturn_eval) > 20, f"{c.scenario}: name a measurable evaluation"


def run() -> None:
    step("Four scenario cards (templates/scenario-cards.md), one canvas each (templates/fine-tuning-decision-canvas.md)")
    show(pd.DataFrame([asdict(c) for c in CANVASES])[["scenario", "gap", "decision", "overturn_eval"]], index=False, max_width=70)
    print("\nOutline vs current API (§3.3): the OpenAI fine-tuning API is being wound down and Claude cannot be "
          "fine-tuned, so 'fine-tuning' here means an open-weight model you host and evaluate.")
    check("Part 3 — decision canvases", check_part3)
