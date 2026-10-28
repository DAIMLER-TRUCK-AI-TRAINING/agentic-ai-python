#!/usr/bin/env python3
"""Build the synthetic IT-operations SQLite database used by Module 2 Labs 04 and 05.

    python labs/module-02/data/build_ops_db.py            # writes data/ops.db if it is missing
    python labs/module-02/data/build_ops_db.py --force    # rebuild from scratch

The database is generated, not committed (it is git-ignored). Everything in it is
synthetic: about 430 historical incidents (seeded, so every build is identical),
the open incidents from ``incidents.json``, the knowledge base from
``kb_articles.json``, the configuration items from ``service_status.json`` and the
five change requests from ``change_requests/ground_truth.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_DB = DATA_DIR / "ops.db"
SEED = 20261027

SCHEMA_SQL = """
CREATE TABLE incidents (
    incident_id      TEXT PRIMARY KEY,           -- INC-nnnnnn
    opened_at        TEXT NOT NULL,              -- ISO 8601 in UTC, e.g. 2026-09-14T08:05+00:00
    closed_at        TEXT,                       -- ISO 8601 in UTC; NULL while open
    status           TEXT NOT NULL,              -- New | In progress | Resolved | Closed
    priority         TEXT NOT NULL,              -- P1 | P2 | P3 | P4
    category         TEXT NOT NULL,              -- access | network | endpoint | email_collab | business_app | platform | security
    assignment_group TEXT NOT NULL,              -- Service Desk | Network Ops | Finance | Platform Engineering | Security Operations
    region           TEXT NOT NULL,              -- EU | US | APAC
    site             TEXT NOT NULL,              -- e.g. EU-AMS-02
    ci_id            TEXT,                       -- configuration item, e.g. vpn-gateway-eu-01
    summary          TEXT NOT NULL,
    kb_id            TEXT,                       -- knowledge-base article used, if any
    related_change   TEXT                        -- CHG-nnnnn, if any
);
CREATE TABLE kb_articles (
    kb_id       TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL,
    owner_team  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    escalation  TEXT NOT NULL,
    tags        TEXT NOT NULL                    -- comma-separated
);
CREATE TABLE services (
    ci_id       TEXT PRIMARY KEY,
    service     TEXT NOT NULL,
    name        TEXT NOT NULL,
    owner_team  TEXT NOT NULL,
    region      TEXT NOT NULL,
    status      TEXT NOT NULL,                   -- operational | degraded | down
    note        TEXT NOT NULL
);
CREATE TABLE changes (
    change_id     TEXT PRIMARY KEY,
    change_type   TEXT NOT NULL,                 -- standard | normal | emergency
    risk          TEXT NOT NULL,                 -- low | medium | high
    team          TEXT NOT NULL,
    region        TEXT NOT NULL,
    window_start  TEXT NOT NULL,
    window_end    TEXT NOT NULL,
    cab_required  INTEGER NOT NULL,              -- 0 | 1
    status        TEXT NOT NULL                  -- Requested | Approved | Scheduled
);
CREATE INDEX idx_incidents_group ON incidents (assignment_group, priority);
CREATE INDEX idx_incidents_opened ON incidents (opened_at);
"""

def _utc(timestamp: str) -> str:
    """Normalise an ISO 8601 timestamp with any offset to UTC, so that string order is time order."""
    return datetime.fromisoformat(timestamp).astimezone(timezone.utc).isoformat(timespec="minutes")


SITES = {
    "EU": ["EU-AMS-02", "EU-FRA-01", "EU-LON-01"],
    "US": ["US-DAL-01", "US-NYC-03"],
    "APAC": ["APAC-SIN-01", "APAC-SYD-02"],
}

# category -> (assignment group, weight, configuration items, typical summaries)
CATEGORY_PROFILE = {
    "access": ("Service Desk", 22, ["idp-global"], [
        "Account locked after failed sign-ins", "MFA push not arriving", "Access request for shared folder",
        "Password reset not accepted"]),
    "network": ("Network Ops", 18, ["vpn-gateway-{r}-01", "wifi-{s}", "core-switch-{s}"], [
        "VPN rejects credentials after password change", "Wi-Fi drops in meeting rooms",
        "Wired network down on one floor", "VPN disconnects every few minutes"]),
    "endpoint": ("Service Desk", 24, ["laptop", "printer"], [
        "Laptop asks for encryption recovery key", "Print queue stuck", "Monitor flickers",
        "Software install request"]),
    "email_collab": ("Service Desk", 12, ["mail-relay-{r}"], [
        "Email stuck in outbox", "Calendar shows wrong times", "Shared mailbox access missing"]),
    "business_app": ("Finance", 8, ["erp-finance-{r}"], [
        "Payment run failed with PAY-17", "Invoice does not match purchase order", "Month-end close job failed"]),
    "platform": ("Platform Engineering", 9, ["st-{r}-01", "build-agents", "fs-{r}-03"], [
        "File server disk filling up", "Build agents offline", "Storage latency spikes"]),
    "security": ("Security Operations", 7, ["mail-gateway-{r}", "laptop"], [
        "Phishing email reported", "Malware alert on laptop", "Lost laptop reported"]),
}
KB_BY_SUMMARY = {
    "Account locked after failed sign-ins": "KB-0107", "MFA push not arriving": "KB-0608",
    "VPN rejects credentials after password change": "KB-0042", "Wi-Fi drops in meeting rooms": "KB-0715",
    "Laptop asks for encryption recovery key": "KB-0450", "Print queue stuck": "KB-0213",
    "Email stuck in outbox": "KB-0521", "Payment run failed with PAY-17": "KB-0790",
    "Invoice does not match purchase order": "KB-0318", "Month-end close job failed": "KB-0790",
    "Phishing email reported": "KB-0733", "Malware alert on laptop": "KB-0733",
}
PRIORITY_WEIGHTS = {"P1": 3, "P2": 12, "P3": 40, "P4": 45}


def _history(rng: random.Random, count: int = 430) -> list[tuple]:
    """Generate ``count`` closed or resolved historical incidents between 1 Aug and 25 Oct 2026."""
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    span_minutes = int((datetime(2026, 10, 25, tzinfo=timezone.utc) - start).total_seconds() // 60)
    categories = list(CATEGORY_PROFILE)
    weights = [CATEGORY_PROFILE[c][1] for c in categories]
    rows = []
    for n, minute in enumerate(sorted(rng.randrange(span_minutes) for _ in range(count))):
        category = rng.choices(categories, weights)[0]
        group, _, cis, summaries = CATEGORY_PROFILE[category]
        region = rng.choices(["EU", "US", "APAC"], [5, 3, 2])[0]
        site = rng.choice(SITES[region])
        priority = rng.choices(list(PRIORITY_WEIGHTS), list(PRIORITY_WEIGHTS.values()))[0]
        if category == "security" and priority in ("P3", "P4") and rng.random() < 0.5:
            priority = "P2"
        summary = rng.choice(summaries)
        ci = rng.choice(cis)
        if ci == "laptop":
            ci = f"LT-{rng.randrange(4000, 7000)}"
        elif ci == "printer":
            ci = f"PR-{rng.randrange(1, 6)}F-{rng.randrange(1, 12):02d}"
        else:
            ci = ci.format(r=region.lower(), s=site.lower())
        opened = start + timedelta(minutes=minute)
        hours_open = {"P1": 4, "P2": 10, "P3": 30, "P4": 72}[priority] * rng.uniform(0.3, 1.6)
        closed = opened + timedelta(hours=hours_open)
        rows.append((
            f"INC-{103780 + n}", opened.isoformat(timespec="minutes"), closed.isoformat(timespec="minutes"),
            "Closed", priority, category, group, region, site, ci, summary,
            KB_BY_SUMMARY.get(summary) if rng.random() < 0.8 else None, None,
        ))
    return rows


def build_ops_db(db_path: Path = DEFAULT_DB, *, force: bool = False) -> Path:
    """Create the database at ``db_path`` (skipped if it exists, unless ``force``) and return its path."""
    db_path = Path(db_path)
    if db_path.exists() and not force:
        return db_path
    db_path.unlink(missing_ok=True)
    rng = random.Random(SEED)
    kb = json.loads((DATA_DIR / "kb_articles.json").read_text(encoding="utf-8"))
    open_incidents = json.loads((DATA_DIR / "incidents.json").read_text(encoding="utf-8"))
    services = json.loads((DATA_DIR / "service_status.json").read_text(encoding="utf-8"))
    changes = json.loads((DATA_DIR / "change_requests" / "ground_truth.json").read_text(encoding="utf-8"))
    change_status = {"CHG-20931": "Requested", "CHG-20932": "Approved", "CHG-20933": "Approved",
                     "CHG-20934": "Requested", "CHG-20935": "Scheduled"}

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.executemany("INSERT INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", _history(rng))
        conn.executemany(
            "INSERT INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(i["incident_id"], _utc(i["opened_at"]), None if i["status"] != "Resolved" else _utc(i["updates"][-1]["at"]),
              i["status"], i["priority"], i["category"], i["assignment_group"], i["region"], i["site"], i["ci_id"],
              i["summary"], i["kb_id"], i["related_change"]) for i in open_incidents],
        )
        conn.executemany(
            "INSERT INTO kb_articles VALUES (?,?,?,?,?,?,?)",
            [(a["kb_id"], a["title"], a["category"], a["owner_team"], a["summary"], a["escalation"], ",".join(a["tags"]))
             for a in kb],
        )
        conn.executemany(
            "INSERT INTO services VALUES (?,?,?,?,?,?,?)",
            [(s["ci_id"], s["service"], s["name"], s["owner_team"], s["region"], s["status"], s["note"]) for s in services],
        )
        conn.executemany(
            "INSERT INTO changes VALUES (?,?,?,?,?,?,?,?,?)",
            [(c["change_id"], c["change_type"], c["risk"], c["requested_by_team"], c["region"], c["window_start"],
              c["window_end"], int(c["cab_required"]), change_status[cid])
             for cid, c in changes.items() if cid.startswith("CHG-")],
        )
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic IT-operations SQLite database.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"output path (default: {DEFAULT_DB.name})")
    parser.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = parser.parse_args()
    path = build_ops_db(args.db, force=args.force)
    with sqlite3.connect(path) as conn:
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("incidents", "kb_articles", "services", "changes")}
    print(f"{path} · " + " · ".join(f"{n} {t}" for t, n in counts.items()))


if __name__ == "__main__":
    main()
