"""Offline tests: the deterministic parts of Lab 04. No API key or network needed.

The database tests build labs/module-02/data/ops.db if it is missing (git-ignored, identical every time).
"""

import json

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from lab04.part2_tools import VALIDATOR_CASES, mock_search, query_ops_db, search_kb, validate_select
from lab04.part3_agent import answers_a, ground_truth_a
from lab04.part4_buffer import CONVO, probe_results
from lab04.part5_memory import keep_last_turns, token_table, without_old_thinking


@pytest.mark.parametrize("sql", VALIDATOR_CASES["allowed"])
def test_validator_allows_one_read_only_select(sql):
    assert validate_select(sql) == (True, "ok")


@pytest.mark.parametrize("sql", VALIDATOR_CASES["blocked"])
def test_validator_blocks_everything_else(sql):
    ok, reason = validate_select(sql)
    assert not ok and reason


def test_database_tool_rejects_writes_and_caps_rows():
    assert json.loads(query_ops_db.invoke({"sql": "DELETE FROM incidents"}))["error"] == "rejected"
    assert json.loads(query_ops_db.invoke({"sql": "SELECT COUNT(*) FROM incidents"}))["rows"][0][0] == 437
    capped = json.loads(query_ops_db.invoke({"sql": "SELECT * FROM incidents"}))
    assert len(capped["rows"]) == 50 and capped["truncated"] is True
    assert json.loads(query_ops_db.invoke({"sql": "SELECT nope FROM incidents"}))["error"] == "sql_error"


def test_question_a_ground_truth_matches_the_guide():
    truth = ground_truth_a()
    assert tuple(truth) == ("vpn-gateway-eu-01", 5)
    assert answers_a("vpn-gateway-eu-01 had 5 Network Ops incidents", truth)
    assert not answers_a("vpn-gateway-eu-02 had 4", truth)


def test_offline_search_tools():
    assert mock_search("SynthGate VPN rejects new password")[0]["url"].endswith("SG-2026-031")
    assert mock_search("zzz qqq") == []
    assert json.loads(search_kb.invoke({"query": "vpn password rejected"}))[0]["kb_id"] == "KB-0042"


def conversation():
    return [
        HumanMessage("a", id="1"), AIMessage("", id="2", tool_calls=[{"name": "search_kb", "args": {}, "id": "t1"}]),
        ToolMessage("r", tool_call_id="t1", id="3"), AIMessage("b", id="4"), HumanMessage("c", id="5"),
        AIMessage("d", id="6"), HumanMessage("e", id="7"),
    ]


def test_keep_last_turns_never_splits_a_tool_round():
    msgs = conversation()
    assert [m.id for m in keep_last_turns(msgs, 2)] == ["5", "6", "7"]
    assert [m.id for m in keep_last_turns(msgs, 1)] == ["7"]
    assert keep_last_turns(msgs, 3) == msgs and keep_last_turns(msgs, 9) == msgs


def test_old_thinking_is_stripped_and_the_current_turn_kept():
    thought = AIMessage(content=[{"type": "thinking", "thinking": "…", "signature": "sig"}, {"type": "text", "text": "hi"}], id="a1")
    cleaned = without_old_thinking([HumanMessage("q1", id="h1"), thought, HumanMessage("q2", id="h2"), thought.model_copy(update={"id": "a2"})])
    assert [b["type"] for b in cleaned[1].content] == ["text"] and len(cleaned[3].content) == 2


def test_probes_and_token_table():
    import pandas as pd

    turns = len(CONVO["turns"])
    facts = " ".join(s for p in CONVO["probes"] for s in p["must_contain"])
    frame = pd.DataFrame({"turn": range(1, turns + 1), "input tokens": [100 * n for n in range(1, turns + 1)], "reply": [facts] * turns})
    assert all(probe_results(frame).values())
    table = token_table({"buffer": frame, "window": frame.assign(**{"input tokens": 100})})
    assert table.loc["buffer", "total"] == 100 * turns * (turns + 1) // 2 and table.loc["window", "total"] == 100 * turns
