"""Shared types for the Enterprise Operations Assistant: labels, the routing table and the output contract.

The schemas avoid JSON-Schema keywords that one provider's structured outputs might not accept
(lengths, item counts, patterns). Those rules live in ``policy.enforce_policy`` instead, where
they are deterministic and testable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["access", "network", "endpoint", "email_collab", "business_app", "platform", "security"]
Priority = Literal["P1", "P2", "P3", "P4"]
Level = Literal["high", "medium", "low"]
AssignmentGroup = Literal["Service Desk", "Network Ops", "Finance", "Platform Engineering", "Security Operations"]

# POL-OPS-004 §4: the default assignment group per category.
ROUTING: dict[str, str] = {
    "access": "Service Desk",
    "endpoint": "Service Desk",
    "email_collab": "Service Desk",
    "network": "Network Ops",
    "business_app": "Finance",
    "platform": "Platform Engineering",
    "security": "Security Operations",
}

# POL-OPS-004 §2: impact x urgency -> priority.
MATRIX: dict[tuple[str, str], str] = {
    ("high", "high"): "P1", ("high", "medium"): "P2", ("high", "low"): "P3",
    ("medium", "high"): "P2", ("medium", "medium"): "P3", ("medium", "low"): "P4",
    ("low", "high"): "P3", ("low", "medium"): "P4", ("low", "low"): "P4",
}


class Triage(BaseModel):
    """Step 1 of the workflow architecture: the triage decision alone."""

    category: Category
    impact: Level = Field(description="high: a site, a service or 20+ users; medium: a team or 2-19 users; low: one user")
    urgency: Level = Field(description="high: stopped, deadline today or active threat; medium: degraded or workaround; low: request or no impact yet")
    priority: Priority
    assignment_group: AssignmentGroup
    rationale: str = Field(description="One sentence: the facts that set impact and urgency, and any override applied")


class EscalationRecord(BaseModel):
    """The assistant's output contract (SPEC-OPS-001 §6)."""

    incident_id: str | None = Field(description="The ticket's INC- ID, or null if it has none")
    category: Category
    impact: Level
    urgency: Level
    priority: Priority
    assignment_group: AssignmentGroup
    kb_ids: list[str] = Field(description="IDs of knowledge-base articles returned by search_kb that apply; empty if none")
    escalate: bool
    escalation_target: str | None = Field(description="Queue to escalate to, e.g. 'Network Ops L2'; null when escalate is false")
    escalation_reason: str = Field(description="The rule that triggered escalation (E1-E4), or why none applies")
    user_reply: str = Field(description="Reply to the user, at most 120 words; never promise a resolution time or claim an action")
    next_actions: list[str] = Field(description="One to five next steps for the analyst")
