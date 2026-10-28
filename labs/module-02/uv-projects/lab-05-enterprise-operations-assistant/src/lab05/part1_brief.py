"""Part 1 — the brief: the spec becomes the system prompt, and eight labelled cases define "works"."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import DATA_DIR, LAB_DIR, show, step
from .ops import EscalationRecord, render_system_prompt

POLICY = (DATA_DIR / "triage_policy.md").read_text(encoding="utf-8")
TEAM_SPEC = LAB_DIR / "outputs" / "enterprise-ops-assistant.spec.md"
REFERENCE_SPEC = LAB_DIR / "solutions" / "enterprise-ops-assistant.spec.md"
CASES = json.loads((DATA_DIR / "eval_incidents.json").read_text(encoding="utf-8"))["cases"]
CASE_BY_ID = {c["case_id"]: c for c in CASES}

_spec_override: list[Path] = []


def use_spec(path: Path | None) -> None:
    """Use `path` instead of the default spec (the CLI's --spec)."""
    _spec_override[:] = [path] if path else []


def team_spec_ready() -> bool:
    return TEAM_SPEC.exists() and "TODO" not in TEAM_SPEC.read_text(encoding="utf-8")


def spec_path() -> Path:
    """--spec if given, else your Lab 03 spec in outputs/ if it is finished, else the reference spec."""
    if _spec_override:
        return _spec_override[0]
    return TEAM_SPEC if team_spec_ready() else REFERENCE_SPEC


def system_prompt() -> str:
    return render_system_prompt(spec_path().read_text(encoding="utf-8"), POLICY)


def run() -> None:
    step("The spec becomes the system prompt")
    path = spec_path()
    note = "" if _spec_override or team_spec_ready() else " (your outputs/ spec is missing or has TODOs)"
    print(f"Spec: {path.relative_to(LAB_DIR) if path.is_relative_to(LAB_DIR) else path}{note}")
    print(f"System prompt: {len(system_prompt()):,} characters")

    step("The evaluation set: eight labelled cases")
    show(pd.DataFrame([{"case": c["case_id"], "ticket": c["text"][:70] + "…", **c["expected"]} for c in CASES]), index=False)
    print("\nEscalationRecord fields:", ", ".join(EscalationRecord.model_fields))
