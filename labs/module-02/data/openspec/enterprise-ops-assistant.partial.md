# Enterprise Operations Assistant — agent spec

> **Partially filled spec for Lab 03 Part 4.** The Lab 03 notebook copies this file to
> `labs/module-02/outputs/enterprise-ops-assistant.spec.md`. Complete the copy as a team: replace
> every `TODO`. Write every **THEN** bullet as checkable clauses, for example
> `priority is P2, assignment group is Network Ops, escalate is true` or `kb includes KB-0042`:
> the notebook turns each scenario into a test.

| Field | Value |
| ----- | ----- |
| Spec ID | SPEC-OPS-001 |
| Owner team | Service Desk |
| Version | 0.2 |
| Status | draft |
| Related policies | POL-OPS-004 (incident triage and priority) |

## 1. Problem

The Service Desk receives about 1,800 incident tickets a day across the EU, US and APAC. L1 analysts
spend roughly four minutes per ticket choosing a category, a priority and an assignment group, and
about one ticket in six is re-routed at least once. The assistant drafts the triage, finds the
matching knowledge-base article and produces a structured escalation record that an analyst
confirms in one click.

**In scope:** new incident tickets in English, German, French or Spanish, from the portal, email or chat.

**Out of scope:** service requests that need an approval workflow, changes to existing incidents,
and any action on a user's account or device.

**Success measures:** TODO: give measurable targets for priority accuracy, escalation recall, schema validity, latency and cost per ticket.

## 2. Requirements

### Requirement: Triage by policy

The assistant SHALL assign a category, a priority and an assignment group according to POL-OPS-004,
using impact × urgency and the two overrides.

#### Scenario: Single user locked out of the VPN

- **WHEN** one user reports that the VPN rejects their new password and they cannot work from home
- **THEN** priority is P3, assignment group is Network Ops, escalate is false

### Requirement: Security floor

The assistant SHALL treat credentials entered on a suspicious page, a malware alert or a lost device
as a confirmed security incident of at least P2, assigned to Security Operations and escalated.

#### Scenario: Password typed into a phishing page

- **WHEN** a user says they entered their password on a page linked from a 'confirm payroll details' email
- **THEN** priority is P2, assignment group is Security Operations, escalate is true

### Requirement: No seniority bias

TODO: one SHALL sentence about job titles, tone and requested priorities.

#### Scenario: TODO

- **WHEN** TODO
- **THEN** TODO

### Requirement: Grounded knowledge lookup

TODO: one SHALL sentence about which knowledge-base articles the assistant may cite, and when an article's own escalation rule applies.

#### Scenario: TODO

- **WHEN** TODO
- **THEN** TODO

### Requirement: Read-only and honest

TODO: one SHALL NOT sentence about actions and promised resolution times.

#### Scenario: TODO

- **WHEN** TODO
- **THEN** TODO

## 3. Tool contracts

### Tool: `lookup_incident`

| Field | Value |
| ----- | ----- |
| Purpose | Return one incident record by its ID, for tickets that mention an existing incident |
| Inputs | incident_id: string, pattern `INC-\d{6}` |
| Output | JSON object with status, priority, category, assignment group, site, CI, summary, related change |
| Errors | `{"error": "not_found", "incident_id": …}` for an unknown ID; `{"error": "invalid_id"}` for a bad format |
| Side effects | None (read-only) |
| Permission | Any authenticated L1 analyst session |

### Tool: `search_kb`

| Field | Value |
| ----- | ----- |
| Purpose | Find knowledge-base articles that match a problem description |
| Inputs | query: string (1–200 characters); max_results: integer 1–5, default 3 |
| Output | List of up to 5 articles: kb_id, title, summary, escalation rule, score |
| Errors | TODO |
| Side effects | TODO |
| Permission | TODO |

### Tool: `get_service_status`

| Field | Value |
| ----- | ----- |
| Purpose | TODO |
| Inputs | TODO |
| Output | TODO |
| Errors | TODO |
| Side effects | TODO |
| Permission | TODO |

### Tool: `query_ops_db`

| Field | Value |
| ----- | ----- |
| Purpose | Answer counting and trend questions over incident history with one read-only SQL SELECT |
| Inputs | TODO |
| Output | TODO |
| Errors | TODO |
| Side effects | TODO |
| Permission | TODO |

## 4. Role boundaries

| The agent MAY | The agent MUST NOT |
| ------------- | ------------------ |
| Read incidents, KB articles, service status and incident history through the four tools | Close, resolve, re-prioritise or re-assign an existing incident |
| TODO | TODO |
| TODO | TODO |

## 5. Escalation rules

| ID | Trigger (measurable) | Escalate to | Hand-over must include |
| -- | -------------------- | ----------- | ---------------------- |
| E1 | Priority is P1 or P2 | Assignment group's L2 queue; P1 also pages on-call | Ticket ID, priority with impact and urgency, affected site and users, KB articles tried |
| E2 | TODO | TODO | TODO |
| E3 | TODO | TODO | TODO |

## 6. Output contract

The assistant returns an `EscalationRecord`: `incident_id`, `category`, `priority`, `impact`,
`urgency`, `assignment_group`, `kb_ids` (only IDs returned by `search_kb`), `escalate`,
`escalation_target` (null when `escalate` is false), `escalation_reason`, `user_reply`
(at most 120 words, no promised times) and `next_actions` (1–5 items). After the model answers, code
enforces E1 and E2, drops unknown KB IDs and checks the assignment group against the category
before anything is stored.

## 7. Open questions

- TODO: at least one decision still to make, with an owner.
