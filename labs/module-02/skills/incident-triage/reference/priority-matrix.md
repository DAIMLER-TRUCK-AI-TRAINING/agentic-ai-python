# Priority matrix and routing (from POL-OPS-004)

## Categories

| Category | Use it for | Assignment group |
| -------- | ---------- | ---------------- |
| `access` | Accounts, passwords, lock-outs, MFA, folder and application permissions, the identity service | Service Desk |
| `network` | VPN, Wi-Fi, wired network, remote access | Network Ops |
| `endpoint` | Laptops, desktops, monitors, printers, peripherals, software on a device | Service Desk |
| `email_collab` | Email, calendars, shared mailboxes, collaboration tools | Service Desk |
| `business_app` | ERP and finance applications: invoices, payment runs, month-end close | Finance |
| `platform` | Servers, storage, VMs, file servers, build and deployment pipelines | Platform Engineering |
| `security` | Credentials entered on a suspicious page, malware, lost or stolen devices, phishing | Security Operations |

Exception: a service-wide identity or email outage (high impact) goes to **Platform Engineering**.

## Impact × urgency

| Impact \ Urgency | High | Medium | Low |
| ---------------- | ---- | ------ | --- |
| **High** | P1 | P2 | P3 |
| **Medium** | P2 | P3 | P4 |
| **Low** | P3 | P4 | P4 |

Overrides: confirmed security incident ⇒ at least P2. Seniority and requested priority are ignored.
