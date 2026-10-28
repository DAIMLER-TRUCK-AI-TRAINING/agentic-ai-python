"""Part 3 — policy as code: what the spec says must always hold, enforced after the model answers."""

from __future__ import annotations

import re

import pandas as pd

from .common import show, step
from .ops import MATRIX, ROUTING, EscalationRecord, ops_tools

RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
KNOWN_KB = ops_tools.known_kb_ids()
PROMISED_TIME = re.compile(
    r"\b(will|'ll|shall)\b[^.!?]{0,60}?\b(within|in|by)\s+(\d+\s*(minutes?|mins?|hours?|hrs?|days?)|\d{1,2}[:.]\d{2})\b", re.IGNORECASE
)


def allowed_priorities(category: str, impact: str, urgency: str) -> set[str]:
    """The matrix result, plus P2 for security (the floor for a confirmed incident)."""
    expected = MATRIX[(impact, urgency)]
    return {expected, "P2"} if category == "security" and RANK[expected] > 2 else {expected}


def allowed_groups(category: str, impact: str) -> set[str]:
    """The routing table, plus Platform Engineering for service-wide identity or email outages."""
    groups = {ROUTING[category]}
    return groups | {"Platform Engineering"} if category in ("access", "email_collab") and impact == "high" else groups


def enforce_policy(record: EscalationRecord) -> tuple[EscalationRecord, list[str]]:
    """Return a corrected copy of `record` and one note per change. The rules, in order:

    1. priority not in allowed_priorities(...)     -> the most urgent allowed value.
    2. assignment_group not in allowed_groups(...) -> ROUTING[category].
    3. kb_ids not in KNOWN_KB                       -> dropped.
    4. priority P1/P2 or category security, but escalate is false -> escalate = True (E1/E2).
    5. escalate true and no escalation_target       -> "Security Operations" for security, else f"{group} L2";
       escalate false and a target                  -> target = None.
    6. user_reply matches PROMISED_TIME             -> a "warning:" note (a human checks it; the text is not edited).
    """
    r, notes = record.model_copy(deep=True), []
    priorities = allowed_priorities(r.category, r.impact, r.urgency)
    if r.priority not in priorities:
        fixed = min(priorities, key=RANK.__getitem__)
        notes.append(f"priority {r.priority} -> {fixed}")
        r.priority = fixed
    if r.assignment_group not in allowed_groups(r.category, r.impact):
        notes.append(f"group {r.assignment_group} -> {ROUTING[r.category]}")
        r.assignment_group = ROUTING[r.category]
    unknown = [k for k in r.kb_ids if k not in KNOWN_KB]
    if unknown:
        notes.append(f"dropped unknown KB {unknown}")
        r.kb_ids = [k for k in r.kb_ids if k in KNOWN_KB]
    if (RANK[r.priority] <= 2 or r.category == "security") and not r.escalate:
        notes.append("escalate -> true (" + ("E2 security" if r.category == "security" else f"E1 {r.priority}") + ")")
        r.escalate = True
    if r.escalate and not r.escalation_target:
        r.escalation_target = "Security Operations" if r.category == "security" else f"{r.assignment_group} L2"
        notes.append(f"target -> {r.escalation_target}")
    elif not r.escalate and r.escalation_target:
        notes.append("target cleared (escalate is false)")
        r.escalation_target = None
    if PROMISED_TIME.search(r.user_reply):
        notes.append("warning: the reply promises a time; route to human review")
    return r, notes


def record(**overrides) -> EscalationRecord:
    """A hand-made record with known answers: a low/low endpoint ticket, changed by `overrides`."""
    base = dict(
        incident_id="INC-104700", category="endpoint", impact="low", urgency="low", priority="P4",
        assignment_group="Service Desk", kb_ids=[], escalate=False, escalation_target=None,
        escalation_reason="none", user_reply="Thanks, an analyst will look at this.", next_actions=["Check the device"],
    )
    return EscalationRecord(**{**base, **overrides})


EXAMPLES = {
    "security P3, one made-up KB ID": record(category="security", urgency="high", priority="P3", kb_ids=["KB-0733", "KB-9999"]),
    "low/low ticket marked P1": record(priority="P1", escalate=True, escalation_target="Service Desk L2"),
    "email outage routed to Platform Engineering": record(
        category="email_collab", impact="high", urgency="high", priority="P1", assignment_group="Platform Engineering", escalate=True
    ),
    "reply promises a time": record(user_reply="Your printer will be fixed within 2 hours."),
    "nothing to correct": record(),
}


def run() -> None:
    step("enforce_policy on hand-made records with known answers (no model)")
    rows = []
    for label, rec in EXAMPLES.items():
        fixed, notes = enforce_policy(rec)
        rows.append({
            "record": label, "priority": f"{rec.priority} → {fixed.priority}", "group": fixed.assignment_group,
            "escalate": fixed.escalate, "target": fixed.escalation_target, "kb": fixed.kb_ids, "notes": "; ".join(notes) or "—",
        })
    show(pd.DataFrame(rows).fillna("—"), index=False, max_width=70)
    print("\nThe model judges category, impact, urgency and which article applies; code enforces what must always hold.")
