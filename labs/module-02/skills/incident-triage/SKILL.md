---
name: incident-triage
description: Triage an IT incident ticket under policy POL-OPS-004. Returns the category, impact, urgency, priority (P1-P4) and assignment group as JSON. Use when a user or analyst asks to triage, classify, prioritise or route an incident, support ticket or outage report.
---

# Incident triage (POL-OPS-004)

Triage exactly one ticket per request. Work from the facts in the ticket, never from the
requester's job title, tone or the priority they ask for.

## Steps

1. Pick the **category** from the table in [reference/priority-matrix.md](reference/priority-matrix.md).
   A VPN problem is `network` even when the error mentions credentials.
2. Rate **impact**: high (a whole site or service, or 20+ users), medium (a team or 2–19 users),
   low (one user).
3. Rate **urgency**: high (work stopped with no workaround, a deadline today, or an active
   security threat), medium (degraded, or a workaround exists), low (a request, question or risk
   with no impact yet).
4. Read the **priority** from the matrix, then apply the overrides:
   - a *confirmed* security incident (credentials entered on a suspicious page, a malware alert,
     a lost or stolen device) is at least P2; reported phishing that nobody acted on is not confirmed;
   - seniority, tone and requested priority never change the result.
5. Pick the **assignment group** from the routing table. Service-wide identity or email outages go to
   Platform Engineering.
6. If the ticket is ambiguous, choose the most likely reading and say what you assumed in `notes`.

For borderline cases, compare with the worked examples in [reference/examples.md](reference/examples.md).

## Output

Return only this JSON object:

```json
{"category": "network", "impact": "low", "urgency": "high", "priority": "P3",
 "assignment_group": "Network Ops", "notes": "VPN is network even with a credential error."}
```

The calling code validates the result with `scripts/check_triage.py`; a result that breaks the
matrix or the routing table is rejected.
