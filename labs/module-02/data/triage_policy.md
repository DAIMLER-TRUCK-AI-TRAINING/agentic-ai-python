# POL-OPS-004 — Incident triage and priority

| Field | Value |
| ----- | ----- |
| Policy ID | POL-OPS-004 |
| Owner | Service Desk |
| Applies to | Every incident raised through the portal, email or chat (EU, US, APAC) |
| Last reviewed | 2026-09-15 |
| Classification | Internal · synthetic training data |

## 1. Categories

| Category | Use it for |
| -------- | ---------- |
| `access` | Accounts, passwords, lock-outs, MFA, permissions to folders or applications, and the identity service itself |
| `network` | VPN, Wi-Fi, wired network, remote access, connectivity between sites |
| `endpoint` | Laptops, desktops, monitors, printers, peripherals and software on a user's device |
| `email_collab` | Email, calendars, shared mailboxes and collaboration tools |
| `business_app` | The ERP and finance applications: invoices, payment runs, month-end close, journals |
| `platform` | Servers, storage, virtual machines, file servers, build and deployment pipelines |
| `security` | Credentials entered on a suspicious page, malware alerts, lost or stolen devices, reported phishing |

A VPN problem is `network` even when the error mentions credentials.

## 2. Priority: impact × urgency

**Impact**

- **High:** a whole site, a whole service, or 20 or more users.
- **Medium:** one team, or 2 to 19 users.
- **Low:** one user.

**Urgency**

- **High:** work has stopped with no workaround, a business deadline falls today, or there is an active security threat.
- **Medium:** service is degraded or a workaround exists.
- **Low:** a request, a question, a future need, or a risk with no impact yet.

| Impact \ Urgency | High | Medium | Low |
| ---------------- | ---- | ------ | --- |
| **High** | P1 | P2 | P3 |
| **Medium** | P2 | P3 | P4 |
| **Low** | P3 | P4 | P4 |

## 3. Overrides

- **O1 Security floor.** A *confirmed* security incident is at least **P2**: credentials typed into a
  suspicious page, a malware alert on a device, or a lost or stolen device. A phishing email that was
  *reported but not acted on* is not confirmed; use the matrix.
- **O2 No seniority bias.** The requester's job title, tone, capital letters or the priority they ask
  for never change the priority. Apply the matrix to the facts.

## 4. Assignment groups

| Category | Assignment group |
| -------- | ---------------- |
| `access`, `endpoint`, `email_collab` | Service Desk |
| `network` | Network Ops |
| `business_app` | Finance |
| `platform` | Platform Engineering |
| `security` | Security Operations |

**Exception — service-wide outages.** When impact is high because a whole *service* is down for many
users, route it to the team that runs the service: identity and email outages go to **Platform
Engineering**, not the Service Desk.

## 5. Escalation

- P1 and P2 incidents are escalated to the assignment group's L2 queue at once; P1 also pages the on-call engineer.
- Every `security` incident is escalated to Security Operations, whatever its priority.
- Escalate when the matching knowledge-base article's own escalation rule is met (for example, more than
  five users at one site for KB-0042).
- Otherwise the incident stays in the assignment group's L1 queue, with the matching article attached when
  there is one.
- The assistant never closes, resolves or re-prioritises an incident after a human has set it, and never
  promises a resolution time.
