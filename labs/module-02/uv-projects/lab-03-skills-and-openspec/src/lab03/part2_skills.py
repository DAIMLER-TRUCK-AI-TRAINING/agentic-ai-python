"""Part 2 — a Skill loader with the published rules, a sandboxed file tool, and a second Skill of your own."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .common import DATA_DIR, LAB_DIR, OUTPUT_DIR, active_providers, run_parallel, show, step
from .part1_pattern import SKILLS_DIR
from .tool_loop import Tool, run_with_tools


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


class SkillError(ValueError):
    """A Skill folder that breaks the SKILL.md rules."""


RESERVED = ("anthropic", "claude")


def parse_skill(skill_dir: Path) -> Skill:
    """Read skill_dir/SKILL.md and validate its YAML frontmatter.

    Rules (from the Agent Skills docs, plus one convention):
    - the file starts with a frontmatter block between two '---' lines, parsed with yaml.safe_load;
    - name: 1-64 characters, only lowercase letters, digits and hyphens; contains neither
      "anthropic" nor "claude"; equals the folder name (convention);
    - description: non-empty, at most 1,024 characters;
    - neither field contains an XML tag such as <ticket>.
    Raise SkillError with a clear message for the first rule broken.
    """
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise SkillError(f"{skill_dir.name}: SKILL.md must start with a --- frontmatter block")
    meta = yaml.safe_load(match.group(1)) or {}
    name, description = str(meta.get("name", "")), str(meta.get("description", "")).strip()
    if not re.fullmatch(r"[a-z0-9-]{1,64}", name):
        raise SkillError(f"{skill_dir.name}: name must be 1-64 lowercase letters, digits or hyphens, got {name!r}")
    if any(word in name for word in RESERVED):
        raise SkillError(f"{skill_dir.name}: name must not contain {RESERVED}")
    if name != skill_dir.name:
        raise SkillError(f"{skill_dir.name}: name {name!r} must match the folder name")
    if not description or len(description) > 1024:
        raise SkillError(f"{skill_dir.name}: description must be 1-1,024 characters, got {len(description)}")
    if re.search(r"<[^>]+>", name + description):
        raise SkillError(f"{skill_dir.name}: name and description must not contain XML tags")
    return Skill(name=name, description=description, path=skill_dir)


# The reference answer to the notebook's "your turn": Lab 01's change-request pattern as a Skill.
CHANGE_REQUEST_SKILL_MD = """---
name: change-request-extraction
description: Extract a change request (RFC) from an email or document into the change-management JSON record and flag rule breaches for human review. Use when a user pastes or forwards a change request, CHG- document, maintenance window or CAB request.
---

# Change-request extraction

Extract exactly one change request per document.

## Fields

`change_id` (as written, CHG-nnnnn) · `title` (short, English) · `change_type` (standard, normal, emergency) ·
`risk` (low, medium, high, as stated) · `requested_by_team` · `region` (EU, US, APAC) · `affected_services`
(host names or CI IDs, lower case) · `window_start` and `window_end` (ISO 8601 with UTC offset) ·
`cab_required` (true only if the document says so) · `rollback_plan` (one sentence, or null).

## Rules

1. If the document reschedules the change, use the current window, not the original one.
2. Use null for anything the document does not state. Never invent a value.
3. Documents may be in German, French or Spanish; output values in English.
4. After extracting, flag for human review if: risk is high and CAB is not required; a standard change is
   not low risk; a normal or emergency change has no rollback plan; the window ends before it starts.

Return only the JSON record plus a `review_flags` list (empty when nothing breaks a rule).
"""

# Written by this project, beside (not over) the Skill you write from the notebook in outputs/skills/.
GENERATED_SKILLS_DIR = OUTPUT_DIR / "skills"


def write_change_request_skill(root: Path = GENERATED_SKILLS_DIR) -> Path:
    folder = root / "change-request-extraction"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(CHANGE_REQUEST_SKILL_MD, encoding="utf-8")
    return folder


# name → Skill. read_skill_file() serves files only from Skills registered here.
SKILLS: dict[str, Skill] = {}


def discover_skills(roots: list[Path]) -> dict[str, Skill]:
    """Every valid Skill under `roots`; invalid ones are reported and skipped."""
    found = {}
    for root in roots:
        for folder in sorted(p for p in root.glob("*") if (p / "SKILL.md").exists()):
            try:
                skill = parse_skill(folder)
                found[skill.name] = skill
            except SkillError as exc:
                print(f"⚠️  skipped {exc}")
    return found


def read_skill_file(skill: str, path: str) -> dict:
    """Return one file from a Skill folder. The path must stay inside that folder."""
    if skill not in SKILLS:
        return {"error": "unknown_skill", "skill": skill, "available": sorted(SKILLS)}
    root = SKILLS[skill].path.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return {"error": "not_found_or_outside_skill", "path": path}
    return {"skill": skill, "path": path, "content": target.read_text(encoding="utf-8")}


READ_SKILL_FILE = Tool(
    "read_skill_file",
    "Read a file from an installed Skill. Start with 'SKILL.md'; open other files only when SKILL.md points to them.",
    {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Skill name"},
            "path": {"type": "string", "description": "Path inside the Skill folder, e.g. SKILL.md"},
        },
        "required": ["skill", "path"],
        "additionalProperties": False,
    },
    read_skill_file,
)


def system_with_skills(skills: dict[str, Skill]) -> str:
    """Level 1 of every Skill in the system prompt; levels 2-3 load through read_skill_file."""
    catalogue = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    return (
        "You are an IT operations assistant. You have these Skills. When a request matches a Skill's description, "
        "read its SKILL.md with read_skill_file and follow it. Otherwise answer directly.\n"
        f"<skills>\n{catalogue}\n</skills>"
    )


def load_check_triage():
    """The Skill's level-3 script: your harness runs it, the model never sees it."""
    spec = importlib.util.spec_from_file_location("check_triage", SKILLS_DIR / "incident-triage" / "scripts" / "check_triage.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


REQUESTS = {
    "triage": "Triage this ticket: 'Six of us at site US-DAL-01 can't connect to the VPN since 08:00, all with error 809.'",
    "unrelated": "In one sentence: why does a stable system prompt make prompt caching effective?",
}


def skill_run(provider: str, label: str, skills: dict[str, Skill], check_triage) -> dict:
    """Does `provider` load the Skill when, and only when, the request needs it?"""
    r = run_with_tools(provider, system_with_skills(skills), REQUESTS[label], [READ_SKILL_FILE])
    row = {
        "provider": provider, "request": label, "files read": [f"{a['skill']}/{a['path']}" for _, a in r["calls"]],
        "input tokens": r["in"], "answer": r["answer"][:120],
    }
    if label == "triage":
        try:
            result = json.loads(r["answer"][r["answer"].index("{"): r["answer"].rindex("}") + 1])
            row["policy problems"] = check_triage.problems(result) or "none"
        except ValueError:
            row["policy problems"] = "no JSON in the answer"
    return row


def run() -> None:
    step("Discover the example Skill, and give the model a sandboxed file tool")
    SKILLS.clear()
    SKILLS.update(discover_skills([SKILLS_DIR]))
    print(system_with_skills(SKILLS))
    print("\nEscape attempt →", read_skill_file("incident-triage", "../../data/triage_policy.md")["error"])

    providers = active_providers()
    step("Does each provider load the Skill when, and only when, the request needs it?")
    check_triage = load_check_triage()
    triage_only = dict(SKILLS)
    runs = pd.DataFrame(run_parallel(skill_run, [(p, label, triage_only, check_triage) for p in providers for label in REQUESTS]))
    show(runs, index=False, max_width=70)

    step("A second Skill: Lab 01's change-request extraction pattern")
    folder = write_change_request_skill()
    skill = parse_skill(folder)
    SKILLS[skill.name] = skill
    print(f"✅ {folder.relative_to(LAB_DIR)}/SKILL.md is a valid Skill: {skill.name}")
    if not providers:
        print("⚠️  No provider configured: the Skill was validated offline.")
        return
    document = (DATA_DIR / "change_requests" / "CHG-20932.txt").read_text(encoding="utf-8")
    for provider in providers:
        r = run_with_tools(provider, system_with_skills(SKILLS), f"Process this document:\n\n{document}", [READ_SKILL_FILE])
        loaded = {a["skill"] for _, a in r["calls"]}
        right = "change-request-extraction" in loaded and "incident-triage" not in loaded
        print(f"{'✅' if right else '⚠️ '} {provider}: loaded {sorted(loaded)} · {r['in']:,} input tokens")
