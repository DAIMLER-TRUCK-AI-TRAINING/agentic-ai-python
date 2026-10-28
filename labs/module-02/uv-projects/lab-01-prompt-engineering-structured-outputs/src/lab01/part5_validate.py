"""Part 5 — validators the schema cannot express, one retry with the errors, and routing to a person."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
from pydantic import ValidationError, field_validator, model_validator

from .common import active_providers, run_parallel, show, step
from .part4_schema import DOCS, GOLD, ChangeRequest, extract_native


class CheckedChangeRequest(ChangeRequest):
    """Layer 1: format rules the JSON Schema did not express."""

    @field_validator("change_id")
    @classmethod
    def id_format(cls, value: str) -> str:
        if not re.fullmatch(r"CHG-\d{5}", value):
            raise ValueError("change_id must look like CHG-12345")
        return value

    @field_validator("window_start", "window_end")
    @classmethod
    def has_offset(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps need a UTC offset, e.g. 2026-11-07T22:00:00+01:00")
        return value

    @model_validator(mode="after")
    def window_is_sane(self):
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        if self.window_end - self.window_start > timedelta(hours=12):
            raise ValueError("windows longer than 12 hours need a separate review")
        return self


def business_rule_violations(cr: ChangeRequest) -> list[str]:
    """Layer 2: rules about the change itself. A person decides; the model cannot fix them."""
    found = []
    if cr.risk == "high" and not cr.cab_required:
        found.append("high-risk change without CAB approval")
    if cr.change_type == "standard" and cr.risk != "low":
        found.append("a standard (pre-approved) change must be low risk")
    if cr.change_type != "standard" and cr.rollback_plan is None:
        found.append("normal and emergency changes need a rollback plan")
    return found


# Two records with known answers, built from the gold extraction.
REVERSED_WINDOW = {
    **GOLD["CHG-20931"], "title": "test", "rollback_plan": "re-flash",
    "window_start": "2026-11-08T02:00:00+01:00", "window_end": "2026-11-07T22:00:00+01:00",
}
GOLD_20935 = ChangeRequest.model_validate({**GOLD["CHG-20935"], "title": "test", "rollback_plan": "vendor downgrade"})


def extract_with_retry(provider: str, doc_id: str, max_attempts: int = 2) -> dict:
    """Extract → validate → one retry with the errors → route."""
    messages = [{"role": "user", "content": DOCS[doc_id]}]
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        cr = extract_native(provider, messages)
        try:
            checked = CheckedChangeRequest.model_validate(cr.model_dump())
        except ValidationError as exc:
            errors = [f"{'.'.join(map(str, e['loc'])) or 'record'}: {e['msg']}" for e in exc.errors()]
            messages += [
                {"role": "assistant", "content": cr.model_dump_json()},
                {"role": "user", "content": "That extraction failed validation:\n- " + "\n- ".join(errors)
                 + "\nRe-read the document and return a corrected extraction."},
            ]
            continue
        violations = business_rule_violations(checked)
        route = "human review: " + "; ".join(violations) if violations else "auto-accept"
        return {"provider": provider, "doc": doc_id, "attempts": attempt, "route": route, "record": checked}
    return {"provider": provider, "doc": doc_id, "attempts": max_attempts,
            "route": "human review: failed validation twice", "errors": errors, "record": None}


def field_check(cr: ChangeRequest, gold: dict) -> dict:
    """Which of 8 scored fields match the gold extraction (windows compare as instants)."""
    return {
        "change_type": cr.change_type == gold["change_type"],
        "risk": cr.risk == gold["risk"],
        "team": cr.requested_by_team == gold["requested_by_team"],
        "region": cr.region == gold["region"],
        "start": cr.window_start == datetime.fromisoformat(gold["window_start"]),
        "end": cr.window_end == datetime.fromisoformat(gold["window_end"]),
        "cab": cr.cab_required == gold["cab_required"],
        "rollback": (cr.rollback_plan is not None) == gold["rollback_plan_present"],
    }


def run() -> None:
    step("The two layers on records with known answers (no model)")
    try:
        CheckedChangeRequest.model_validate(REVERSED_WINDOW)
    except ValidationError as exc:
        print("layer 1 →", exc.errors()[0]["msg"])
    print("layer 2 →", business_rule_violations(GOLD_20935))

    providers = active_providers()
    if not providers:
        print("⚠️  No provider configured: the validators ran offline; skipping extraction.")
        return
    step("Extract → validate → retry once with the errors → route, for every document")
    routed = run_parallel(extract_with_retry, [(p, d) for p in providers for d in DOCS])
    rows = []
    for r in routed:
        fields = field_check(r["record"], GOLD[r["doc"]]) if r["record"] else {}
        rows.append({
            "provider": r["provider"], "doc": r["doc"], "attempts": r["attempts"], "route": r["route"],
            "fields correct": f"{sum(fields.values())}/8" if fields else "—",
            "wrong": ", ".join(k for k, ok in fields.items() if not ok),
        })
    show(pd.DataFrame(rows), index=False)
    for r in routed:
        if r["doc"] == "CHG-20935" and r["route"] == "auto-accept":
            print(f"⚠️  {r['provider']} auto-accepted CHG-20935: look at its cab_required value. That is a finding.")
    print("\n📝 Write-up: which technique would you ship for triage, what did structured output fix and not fix,")
    print("   and which change requests should a person see?")
