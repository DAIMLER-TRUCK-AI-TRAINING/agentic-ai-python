"""Offline tests: the deterministic parts of Lab 01. No API key or network needed."""

import pandas as pd
import pytest
from pydantic import ValidationError

from lab01.part1_baseline import BASELINE_SCORE, EXAMPLES, FIELDS, TEST, TriageLabel, score
from lab01.part2_techniques import CHAIN_OF_THOUGHT, FEW_SHOT, V1, ZERO_SHOT, TriageWithReasoning, examples_block
from lab01.part3_debug import FEW_SHOT_V2, ZERO_SHOT_V2, regressions
from lab01.part4_schema import DOCS, GOLD, RECORDED_REPLY, ChangeRequest, parse_reply
from lab01.part5_validate import (
    GOLD_20935,
    REVERSED_WINDOW,
    CheckedChangeRequest,
    business_rule_violations,
    field_check,
)


def test_the_labelled_set():
    assert (len(EXAMPLES), len(TEST)) == (6, 24)
    assert TEST.priority.value_counts().to_dict() == {"P4": 9, "P3": 6, "P2": 5, "P1": 4}


def test_scorer_and_label_contract():
    perfect = TEST[["ticket_id", *FIELDS]]
    assert score(perfect) == {"category": 1.0, "priority": 1.0, "assignment_group": 1.0, "all three": 1.0}
    with pytest.raises(ValidationError):
        TriageLabel(category="network", priority="P5", assignment_group="Network Ops")


def test_keyword_baseline_matches_the_guide():
    assert BASELINE_SCORE["all three"] == 0.25


def test_templates_render_and_differ_only_in_technique():
    system, user = FEW_SHOT.render("VPN is down")
    assert user == "<ticket>\nVPN is down\n</ticket>"
    assert system.startswith(ZERO_SHOT.system) and system.count("<example>") == len(EXAMPLES)
    assert CHAIN_OF_THOUGHT.schema is TriageWithReasoning and [t.version for t in V1] == ["v1"] * 3
    assert "<why>" in examples_block(EXAMPLES, with_reason=True) and "<why>" not in examples_block(EXAMPLES)
    assert "POL-OPS-004" in ZERO_SHOT_V2.system and FEW_SHOT_V2.system.startswith(ZERO_SHOT_V2.system)


def test_regressions_find_tickets_that_were_right_and_broke():
    gold = TEST[["ticket_id", *FIELDS]].assign(provider="openai", technique="few-shot")
    broken = gold.copy()
    broken.loc[0, "priority"] = "P1" if broken.loc[0, "priority"] != "P1" else "P4"
    found = regressions(gold, broken)
    assert list(found.ticket_id) == [TEST.ticket_id[0]]
    assert regressions(gold, gold).empty


def test_parse_reply_reports_why_a_reply_fails():
    assert parse_reply(RECORDED_REPLY)[0].startswith("invalid JSON")
    outcome, _ = parse_reply('{"change_id": "CHG-20932"}')
    assert outcome.startswith("schema error")
    record = {**GOLD["CHG-20931"], "title": "t", "rollback_plan": "re-flash"}
    outcome, cr = parse_reply(pd.Series(record).to_json())
    assert outcome == "valid" and isinstance(cr, ChangeRequest)


def test_five_documents_with_gold():
    assert sorted(DOCS) == sorted(GOLD) == [f"CHG-2093{i}" for i in range(1, 6)]


def test_the_validators_catch_what_the_schema_cannot():
    with pytest.raises(ValidationError, match="window_end must be after window_start"):
        CheckedChangeRequest.model_validate(REVERSED_WINDOW)
    assert business_rule_violations(GOLD_20935) == ["high-risk change without CAB approval"]


def test_field_check_compares_windows_as_instants():
    gold = GOLD["CHG-20931"]
    cr = ChangeRequest.model_validate({**gold, "title": "t", "rollback_plan": "re-flash" if gold["rollback_plan_present"] else None})
    assert all(field_check(cr, gold).values())
