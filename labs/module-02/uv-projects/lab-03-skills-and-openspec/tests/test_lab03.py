"""Offline tests: the deterministic parts of Lab 03. No API key or network needed."""

import pytest

from lab03.part1_pattern import SKILLS_DIR, context_cost, skill_files
from lab03.part2_skills import (
    CHANGE_REQUEST_SKILL_MD,
    SKILLS,
    SkillError,
    parse_skill,
    read_skill_file,
    system_with_skills,
    write_change_request_skill,
)
from lab03.part3_openspec import PARTIAL_SPEC, lint_spec, parse_scenarios
from lab03.part4_scenarios import REFERENCE_SPEC, ScenarioOutcome, render_system_prompt, runnable_scenarios, scenario_failures


def test_the_skill_is_cheaper_than_the_pasted_prompt():
    assert all(path.endswith((".md", ".py")) for path, _ in skill_files())
    tokens = context_cost()["≈ tokens"].tolist()
    assert tokens[1] < tokens[2] < tokens[0], "level 1 must be the cheapest, and the full prompt the dearest"


def write_skill(tmp_path, folder: str, text: str):
    skill_dir = tmp_path / folder
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir


def test_parse_skill_accepts_the_example_and_the_reference_skill(tmp_path):
    assert parse_skill(SKILLS_DIR / "incident-triage").name == "incident-triage"
    assert parse_skill(write_change_request_skill(tmp_path)).name == "change-request-extraction"


@pytest.mark.parametrize(
    "name",
    ["Incident_Triage", "claude-helper", "change-request-extraction-but-a-different-folder", "x" * 65],
)
def test_parse_skill_rejects_bad_names(tmp_path, name):
    text = CHANGE_REQUEST_SKILL_MD.replace("name: change-request-extraction", f"name: {name}")
    with pytest.raises(SkillError):
        parse_skill(write_skill(tmp_path, "change-request-extraction", text))


def test_parse_skill_rejects_bad_frontmatter(tmp_path):
    with pytest.raises(SkillError, match="frontmatter"):
        parse_skill(write_skill(tmp_path, "no-frontmatter", "# Just a heading\n"))
    xml = CHANGE_REQUEST_SKILL_MD.replace("Use when", "Use for <ticket> text, when")
    with pytest.raises(SkillError, match="XML"):
        parse_skill(write_skill(tmp_path, "change-request-extraction", xml))


def test_read_skill_file_stays_inside_the_skill():
    SKILLS.clear()
    SKILLS["incident-triage"] = parse_skill(SKILLS_DIR / "incident-triage")
    assert read_skill_file("incident-triage", "SKILL.md")["content"].startswith("---")
    assert read_skill_file("incident-triage", "../../data/triage_policy.md")["error"] == "not_found_or_outside_skill"
    assert read_skill_file("nope", "SKILL.md")["error"] == "unknown_skill"
    assert "- incident-triage: " in system_with_skills(SKILLS)


def test_the_partial_spec_fails_the_linter_and_the_reference_passes():
    problems = lint_spec(PARTIAL_SPEC.read_text(encoding="utf-8"))
    assert any("TODO" in p for p in problems)
    assert lint_spec(REFERENCE_SPEC.read_text(encoding="utf-8")) == []


def test_reference_scenarios_are_checkable():
    spec = REFERENCE_SPEC.read_text(encoding="utf-8")
    scenarios = runnable_scenarios(spec)
    assert len(scenarios) == len(parse_scenarios(spec)) == 7
    system = render_system_prompt(spec)
    assert "## 2. Requirements" in system and "## 3. Tool contracts" not in system and "KB-0042" in system


def test_scenario_failures_compare_only_the_checkable_clauses():
    scenario = {"expect": {"priority": ["P2"], "escalate": ["true"], "kb": ["KB-0042"]}}
    out = ScenarioOutcome(
        category="network", impact="medium", urgency="high", priority="P2", assignment_group="Network Ops",
        kb_ids=["KB-0042"], escalate=True, user_reply="We are on it.",
    )
    assert scenario_failures(scenario, out) == []
    assert scenario_failures(scenario, out.model_copy(update={"priority": "P3"})) == ["priority: expected P2, got P3"]
