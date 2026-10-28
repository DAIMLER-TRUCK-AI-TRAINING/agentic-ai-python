"""Policy as code: deterministic checks applied to every EscalationRecord after the model answers.

The model judges (category, impact, urgency, which article applies); code enforces what must
always hold (SPEC-OPS-001 §5 E1-E2 and §6). Every correction is returned as a note, so it can be
logged and counted in evals.
"""

from __future__ import annotations

import re

from .schemas import MATRIX, ROUTING, EscalationRecord

RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
MAX_REPLY_WORDS = 120
# "will be fixed within 2 hours", "we'll have it working by 14:00": a commitment, not a KB condition.
_PROMISED_TIME = re.compile(
    r"\b(will|'ll|shall)\b[^.!?]{0,60}?\b(within|in|by)\s+(\d+\s*(minutes?|mins?|hours?|hrs?|days?)|\d{1,2}[:.]\d{2})\b",
    re.IGNORECASE,
)


def allowed_priorities(category: str, impact: str, urgency: str) -> set[str]:
    """Matrix result, plus P2 for security (the floor applies to confirmed incidents)."""
    expected = MATRIX[(impact, urgency)]
    allowed = {expected}
    if category == "security" and RANK[expected] > 2:
        allowed.add("P2")
    return allowed


def allowed_groups(category: str, impact: str) -> set[str]:
    """Routing table, plus Platform Engineering for service-wide identity or email outages."""
    groups = {ROUTING[category]}
    if category in ("access", "email_collab") and impact == "high":
        groups.add("Platform Engineering")
    return groups


def enforce_policy(record: EscalationRecord, *, known_kb_ids: set[str]) -> tuple[EscalationRecord, list[str]]:
    """Return a corrected copy of ``record`` and one note per correction or warning."""
    r = record.model_copy(deep=True)
    notes: list[str] = []

    priorities = allowed_priorities(r.category, r.impact, r.urgency)
    if r.priority not in priorities:
        fixed = min(priorities, key=RANK.get)  # type: ignore[arg-type]
        notes.append(f"priority {r.priority} -> {fixed} (impact={r.impact}, urgency={r.urgency}, category={r.category})")
        r.priority = fixed  # type: ignore[assignment]

    groups = allowed_groups(r.category, r.impact)
    if r.assignment_group not in groups:
        fixed_group = ROUTING[r.category]
        notes.append(f"assignment group {r.assignment_group} -> {fixed_group} (category {r.category})")
        r.assignment_group = fixed_group  # type: ignore[assignment]

    unknown = [k for k in r.kb_ids if k not in known_kb_ids]
    if unknown:
        notes.append(f"dropped unknown KB IDs {unknown}")
        r.kb_ids = [k for k in r.kb_ids if k in known_kb_ids]

    must_escalate = RANK[r.priority] <= 2 or r.category == "security"
    if must_escalate and not r.escalate:
        notes.append(f"escalate false -> true ({'E2 security' if r.category == 'security' else 'E1 ' + r.priority})")
        r.escalate = True
    if r.escalate and not r.escalation_target:
        r.escalation_target = "Security Operations" if r.category == "security" else f"{r.assignment_group} L2"
        notes.append(f"escalation target set to {r.escalation_target}")
    if not r.escalate and r.escalation_target:
        notes.append(f"escalation target {r.escalation_target!r} cleared (escalate is false)")
        r.escalation_target = None

    words = len(r.user_reply.split())
    if words > MAX_REPLY_WORDS:
        notes.append(f"warning: user reply has {words} words (limit {MAX_REPLY_WORDS})")
    if _PROMISED_TIME.search(r.user_reply):
        notes.append("warning: user reply appears to promise a time; route to human review")
    if not 1 <= len(r.next_actions) <= 5:
        notes.append(f"next actions trimmed from {len(r.next_actions)} to at most 5")
        r.next_actions = (r.next_actions or ["Review the ticket and confirm the triage."])[:5]
    return r, notes
