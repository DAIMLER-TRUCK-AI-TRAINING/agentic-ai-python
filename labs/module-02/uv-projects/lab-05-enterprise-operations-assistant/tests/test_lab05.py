"""Offline tests: the deterministic parts of Lab 05. No API key or network needed."""

import pandas as pd

from lab05.ops import EscalationRecord
from lab05.part1_brief import CASES, REFERENCE_SPEC, spec_path, system_prompt, use_spec
from lab05.part3_policy import EXAMPLES, allowed_groups, allowed_priorities, enforce_policy, record
from lab05.part4_evaluate import escalation_recall, grade, summarise
from lab05.part5_review import BIG_TICKET, LOG_LINES, REVIEW, condense_attachment, export_review, md_table


def test_the_brief():
    assert [c["case_id"] for c in CASES] == [f"EV-0{i}" for i in range(1, 9)]
    use_spec(REFERENCE_SPEC)
    assert spec_path() == REFERENCE_SPEC
    assert "<agent_spec>" in system_prompt() and "POL-OPS-004" in system_prompt()
    use_spec(None)


def test_allowed_values():
    assert allowed_priorities("network", "high", "high") == {"P1"}
    assert allowed_priorities("security", "low", "high") == {"P3", "P2"}
    assert allowed_groups("email_collab", "high") == {"Service Desk", "Platform Engineering"}
    assert allowed_groups("email_collab", "medium") == {"Service Desk"}


def test_enforce_policy_on_records_with_known_answers():
    r, _ = enforce_policy(EXAMPLES["security P3, one made-up KB ID"])
    assert (r.priority, r.assignment_group, r.escalate, r.escalation_target, r.kb_ids) == (
        "P3", "Security Operations", True, "Security Operations", ["KB-0733"])
    r, _ = enforce_policy(EXAMPLES["low/low ticket marked P1"])
    assert r.priority == "P4", "low impact x low urgency is P4, whatever the model says"
    r, _ = enforce_policy(EXAMPLES["email outage routed to Platform Engineering"])
    assert (r.assignment_group, r.escalation_target) == ("Platform Engineering", "Platform Engineering L2")
    _, notes = enforce_policy(EXAMPLES["reply promises a time"])
    assert any(n.startswith("warning") for n in notes)
    assert enforce_policy(EXAMPLES["nothing to correct"])[1] == []


def test_enforce_policy_does_not_modify_its_input():
    original = record(priority="P1")
    enforce_policy(original)
    assert original.priority == "P1"


def test_the_promised_time_pattern_ignores_kb_text():
    _, notes = enforce_policy(record(user_reply="Per KB-0042, the cache clears within 60 minutes of a password change."))
    assert not any(n.startswith("warning") for n in notes)


def test_grading_and_recall():
    case = next(c for c in CASES if c["expected"]["escalate"])
    e = case["expected"]
    perfect = record(priority=e["priority"], assignment_group=e["assignment_group"], escalate=True,
                     kb_ids=e["kb_any_of"][:1], category="network")
    assert grade(perfect, case)["all"]
    assert grade(None, case) == {k: False for k in ("valid", "priority", "group", "escalate", "kb", "all")}
    evaluation = pd.DataFrame([
        {"provider": "openai", "architecture": "agent", "case": case["case_id"], "valid": True, "priority": True, "group": True,
         "escalate": flag, "kb": True, "all": flag, "raw all": False, "corrections": 1, "seconds": 2.0, "input tokens": 1_000,
         "cost ($)": 0.001}
        for flag in (True, False)
    ])
    assert escalation_recall(evaluation) == 0.5
    assert summarise(evaluation).loc[("openai", "agent"), "corrections"] == 2


def test_condensed_attachment_keeps_the_errors():
    condensed = condense_attachment(BIG_TICKET)
    assert condensed.count("error=809") == 3 and len(condensed) < len(BIG_TICKET) / 50
    assert f"{len(LOG_LINES) + 1 - 10 - 3:,} routine lines removed" in condensed


def test_review_fills_every_placeholder(tmp_path):
    table = pd.DataFrame({"≈ tokens": {"system prompt": 3_000, "ticket": 100}})
    assert md_table(None) == "_not measured_" and md_table(table).startswith("| index |")
    path = export_review(team="Test team", providers=["openai"], architectures=["agent"],
                         tables={"eval_table": None, "budget_table": table, "stress_table": None, "waterfall_table": None},
                         output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "{{" not in text and "Test team" in text and "TODO" not in text
    assert set(REVIEW) >= {"decision", "next_steps"} and isinstance(record(), EscalationRecord)
