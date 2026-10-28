"""Part 1 — the repeated prompting pattern, and what a Skill keeps in context instead."""

from __future__ import annotations

import pandas as pd

from .common import DATA_DIR, LAB_DIR, show, step

SKILLS_DIR = LAB_DIR / "skills"
TRIAGE_SKILL = SKILLS_DIR / "incident-triage"
POLICY = (DATA_DIR / "triage_policy.md").read_text(encoding="utf-8")


def approx_tokens(text: str) -> int:
    """Offline estimate (about 4 characters per token for English). Use the providers' count endpoints for real numbers."""
    return round(len(text) / 4)


def skill_files() -> list[tuple[str, int]]:
    """(path inside skills/, characters) for the Skill's source files; __pycache__ and the like are skipped."""
    return [
        (str(path.relative_to(SKILLS_DIR)), len(path.read_text(encoding="utf-8")))
        for path in sorted(TRIAGE_SKILL.rglob("*"))
        if path.is_file() and path.suffix in {".md", ".py"}
    ]


def context_cost() -> pd.DataFrame:
    """What the repeated pattern costs on every call, and what each Skill level costs only when it is needed."""
    skill_md = (TRIAGE_SKILL / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = skill_md.split("---")[1]
    reference = (TRIAGE_SKILL / "reference" / "priority-matrix.md").read_text(encoding="utf-8")
    return pd.DataFrame([
        {"what": "Lab 01 few-shot v2: policy + 6 examples, in every request", "≈ tokens": approx_tokens(POLICY) + 600},
        {"what": "Skill level 1: name + description, in every request", "≈ tokens": approx_tokens(frontmatter)},
        {"what": "Skill level 2: SKILL.md body, only when triaging", "≈ tokens": approx_tokens(skill_md)},
        {"what": "Skill level 3: one reference file, only if the body sends the model there", "≈ tokens": approx_tokens(reference)},
    ])


def run() -> None:
    step("The example Skill, skills/incident-triage/")
    for path, chars in skill_files():
        print(f"{path}  ({chars:,} chars)")
    print()
    print((TRIAGE_SKILL / "SKILL.md").read_text(encoding="utf-8")[:420], "…")

    step("Pasted into every prompt vs loaded on demand (offline estimate)")
    show(context_cost(), index=False)
