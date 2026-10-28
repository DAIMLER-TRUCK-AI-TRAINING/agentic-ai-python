"""Part 3 — the OpenSpec walkthrough: the template, how each section becomes code, and a deterministic spec linter."""

from __future__ import annotations

import re

import pandas as pd

from .common import DATA_DIR, LAB_DIR, show, step

TEMPLATE = LAB_DIR / "templates" / "openspec-template.md"
PARTIAL_SPEC = DATA_DIR / "openspec" / "enterprise-ops-assistant.partial.md"

SECTION_TO_CODE = pd.DataFrame(
    [
        ("1. Problem", "Success measures", "The eval set and its pass bars (Lab 05 Part 4; Module 5)"),
        ("2. Requirements + scenarios", "SHALL sentences; WHEN/THEN bullets", "System prompt rules; one test case per scenario"),
        ("3. Tool contracts", "Inputs, output, errors, side effects, permission", "JSON Schemas, tool descriptions, the dispatcher's error shapes"),
        ("4. Role boundaries", "MAY / MUST NOT", "System prompt, tool allow-list, read-only connections"),
        ("5. Escalation rules", "Measurable trigger, target, hand-over", "Code that enforces them after the model answers"),
        ("6. Output contract", "Schema and validation rules", "Pydantic model + structured output + validators"),
    ],
    columns=["Section", "What you write", "What it becomes"],
)

CHECKABLE = {
    "priority": re.compile(r"priority is (P[1-4])"),
    "assignment_group": re.compile(r"assignment group is (Service Desk|Network Ops|Finance|Platform Engineering|Security Operations)"),
    "escalate": re.compile(r"escalate is (true|false)"),
    "kb": re.compile(r"kb includes (KB-\d{4})"),
}
SECTIONS = ["## 1. Problem", "## 2. Requirements", "## 3. Tool contracts", "## 4. Role boundaries", "## 5. Escalation rules", "## 6. Output contract"]
TOOL_FIELDS = ["Purpose", "Inputs", "Output", "Errors", "Side effects", "Permission"]


def section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    end = text.find("\n## ", start + len(heading))
    return text[start: end if end > 0 else len(text)]


def table_rows(block: str) -> list[list[str]]:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in block.splitlines() if line.strip().startswith("|")]
    return rows[2:]  # skip the header and the --- line


def parse_scenarios(text: str) -> list[dict]:
    """Every scenario with its requirement, WHEN text and checkable THEN clauses."""
    scenarios = []
    for req in re.split(r"(?m)^### Requirement: ", section(text, "## 2. Requirements"))[1:]:
        req_name = req.split("\n", 1)[0].strip()
        for sc in re.split(r"(?m)^#### Scenario: ", req)[1:]:
            when = re.search(r"\*\*WHEN\*\*\s*(.+)", sc)
            then = re.search(r"\*\*THEN\*\*\s*(.+)", sc)
            expect = {k: p.findall(then.group(1)) for k, p in CHECKABLE.items()} if then else {}
            scenarios.append({
                "requirement": req_name, "scenario": sc.split("\n", 1)[0].strip(),
                "when": when.group(1).strip() if when else "", "then": then.group(1).strip() if then else "",
                "expect": {k: v for k, v in expect.items() if v},
            })
    return scenarios


def lint_spec(text: str) -> list[str]:
    """Everything a complete spec needs that this one lacks; an empty list means clean."""
    problems = [f"missing section {h!r}" for h in SECTIONS if h not in text]
    todos = text.count("TODO")
    if todos:
        problems.append(f"{todos} TODO marker(s) left")
    requirements = re.findall(r"(?m)^### Requirement: (.+)$", text)
    if len(requirements) < 4:
        problems.append(f"{len(requirements)} requirements; write at least 4")
    for req in re.split(r"(?m)^### Requirement: ", section(text, "## 2. Requirements"))[1:]:
        name = req.split("\n", 1)[0].strip()
        if "SHALL" not in req:
            problems.append(f"requirement {name!r} has no SHALL sentence")
        if "#### Scenario:" not in req:
            problems.append(f"requirement {name!r} has no scenario")
    for sc in parse_scenarios(text):
        if not sc["when"] or "TODO" in sc["when"]:
            problems.append(f"scenario {sc['scenario']!r} needs a WHEN")
        elif not sc["expect"]:
            problems.append(f"scenario {sc['scenario']!r}: THEN has no checkable clause (e.g. 'priority is P2')")
    for tool in re.split(r"(?m)^### Tool: ", section(text, "## 3. Tool contracts"))[1:]:
        name = tool.split("\n", 1)[0].strip()
        fields = {r[0]: r[1] for r in table_rows(tool) if len(r) >= 2}
        empty = [f for f in TOOL_FIELDS if not fields.get(f) or "TODO" in fields[f]]
        if empty:
            problems.append(f"tool {name} is missing {', '.join(empty)}")
    boundaries = [r for r in table_rows(section(text, "## 4. Role boundaries")) if "TODO" not in " ".join(r)]
    if len(boundaries) < 3:
        problems.append(f"{len(boundaries)} complete role-boundary rows; write at least 3")
    rules = [r for r in table_rows(section(text, "## 5. Escalation rules")) if len(r) >= 4 and "TODO" not in " ".join(r)]
    if len(rules) < 3:
        problems.append(f"{len(rules)} complete escalation rules; write at least 3")
    for r in rules:
        if not re.search(r"\d|P[1-4]|security|access|network|endpoint|platform|business_app|email_collab", r[1]):
            problems.append(f"escalation rule {r[0]}: the trigger is not measurable")
    return problems


def run() -> None:
    step("The template's skeleton")
    print("\n".join(line for line in TEMPLATE.read_text(encoding="utf-8").splitlines() if line.startswith("#")))

    step("How each section becomes code")
    show(SECTION_TO_CODE, index=False)

    step(f"The linter on the partial spec ({PARTIAL_SPEC.relative_to(LAB_DIR)})")
    for problem in lint_spec(PARTIAL_SPEC.read_text(encoding="utf-8")):
        print("•", problem)
