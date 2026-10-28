#!/usr/bin/env python3
"""Validate a triage result against POL-OPS-004: the matrix, the security floor and the routing table.

    python check_triage.py '{"category": "network", "impact": "low", "urgency": "high", "priority": "P3", "assignment_group": "Network Ops"}'

Exit code 0 and "ok" when the result is consistent; exit code 1 and one line per problem otherwise.
Importable: ``from check_triage import problems``.
"""

from __future__ import annotations

import json
import sys

MATRIX = {
    ("high", "high"): "P1", ("high", "medium"): "P2", ("high", "low"): "P3",
    ("medium", "high"): "P2", ("medium", "medium"): "P3", ("medium", "low"): "P4",
    ("low", "high"): "P3", ("low", "medium"): "P4", ("low", "low"): "P4",
}
ROUTING = {
    "access": "Service Desk", "endpoint": "Service Desk", "email_collab": "Service Desk",
    "network": "Network Ops", "business_app": "Finance", "platform": "Platform Engineering",
    "security": "Security Operations",
}
RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def problems(result: dict) -> list[str]:
    """Return every way ``result`` breaks the policy; an empty list means it is consistent."""
    found = []
    category, priority = result.get("category"), result.get("priority")
    impact, urgency, group = result.get("impact"), result.get("urgency"), result.get("assignment_group")
    if category not in ROUTING:
        return [f"unknown category {category!r}"]
    if (impact, urgency) not in MATRIX:
        return [f"impact/urgency must be high, medium or low, got {impact!r}/{urgency!r}"]
    if priority not in RANK:
        return [f"unknown priority {priority!r}"]
    expected = MATRIX[(impact, urgency)]
    allowed_priorities = {expected}
    if category == "security" and RANK[expected] > 2:
        allowed_priorities.add("P2")  # the security floor, for a confirmed incident
    if priority not in allowed_priorities:
        found.append(f"impact={impact}, urgency={urgency} gives {' or '.join(sorted(allowed_priorities))}, not {priority}")
    allowed = {ROUTING[category]}
    if category in ("access", "email_collab") and impact == "high":
        allowed.add("Platform Engineering")  # service-wide identity or email outage
    if group not in allowed:
        found.append(f"category {category} routes to {' or '.join(sorted(allowed))}, not {group!r}")
    return found


if __name__ == "__main__":
    issues = problems(json.loads(sys.argv[1]))
    print("\n".join(issues) or "ok")
    sys.exit(1 if issues else 0)
