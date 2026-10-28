"""Part 1 — the labelled ticket set, the label contract, the scorer and a keyword baseline to beat."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel

from .common import DATA_DIR, show, step

# 6 worked examples (for few-shot prompts) and 24 test tickets, labelled under POL-OPS-004.
TICKETS = pd.read_csv(DATA_DIR / "triage_tickets.csv")
EXAMPLES = TICKETS[TICKETS.split == "example"].reset_index(drop=True)
TEST = TICKETS[TICKETS.split == "test"].reset_index(drop=True)
POLICY = (DATA_DIR / "triage_policy.md").read_text(encoding="utf-8")

Category = Literal["access", "network", "endpoint", "email_collab", "business_app", "platform", "security"]
Priority = Literal["P1", "P2", "P3", "P4"]
Group = Literal["Service Desk", "Network Ops", "Finance", "Platform Engineering", "Security Operations"]
FIELDS = ["category", "priority", "assignment_group"]


class TriageLabel(BaseModel):
    category: Category
    priority: Priority
    assignment_group: Group


def score(predictions: pd.DataFrame) -> dict[str, float]:
    """Accuracy per field, and for all three at once, against the gold labels of TEST."""
    merged = TEST.merge(predictions[["ticket_id", *FIELDS]], on="ticket_id", suffixes=("", "_pred"), how="left")
    hits = pd.DataFrame({f: merged[f] == merged[f + "_pred"] for f in FIELDS})
    return {**{f: round(float(hits[f].mean()), 3) for f in FIELDS}, "all three": round(float(hits.all(axis=1).mean()), 3)}


# A keyword baseline: ten lines of rules, no model. Every prompt in this lab must beat it.
KEYWORDS = {
    "security": ["phishing", "malware", "trojan", "antivirus", "clicked", "lost", "stolen", "suspicious"],
    "network": ["vpn", "wi-fi", "wifi", "network"],
    "business_app": ["erp", "invoice", "payment", "journal", "zahlungslauf"],
    "platform": ["server", "storage", "build", "pipeline", "cluster"],
    "email_collab": ["email", "mail", "outbox", "calendar", "out-of-office"],
    "access": ["password", "account", "locked", "mfa", "sign in", "access"],
}
ROUTING = {
    "access": "Service Desk",
    "endpoint": "Service Desk",
    "email_collab": "Service Desk",
    "network": "Network Ops",
    "business_app": "Finance",
    "platform": "Platform Engineering",
    "security": "Security Operations",
}


def keyword_baseline(text: str) -> dict:
    t = text.lower()
    category = next((c for c, words in KEYWORDS.items() if any(w in t for w in words)), "endpoint")
    if any(w in t for w in ("everyone", "nobody", "whole", "hundreds", "all ")):
        priority = "P1"
    elif any(w in t for w in ("when convenient", "next week", "next month", "how to", "can you tell")):
        priority = "P4"
    else:
        priority = "P3"
    return {"category": category, "priority": priority, "assignment_group": ROUTING[category]}


def baseline_predictions() -> pd.DataFrame:
    return pd.DataFrame([{"ticket_id": r.ticket_id, **keyword_baseline(r.text)} for r in TEST.itertuples()])


BASELINE_SCORE = score(baseline_predictions())


def run() -> None:
    step("The labelled set")
    print(f"{len(EXAMPLES)} examples · {len(TEST)} test tickets · languages {TEST.language.value_counts().to_dict()}")
    print("priorities:", TEST.priority.value_counts().sort_index().to_dict())
    show(TEST[["ticket_id", "text", "category", "priority", "assignment_group"]].head(6), index=False, max_width=70)

    step("The keyword baseline: no model, ten lines of rules")
    print("keyword baseline:", BASELINE_SCORE)
    print("Every prompt in Parts 2 and 3 has to beat this.")
