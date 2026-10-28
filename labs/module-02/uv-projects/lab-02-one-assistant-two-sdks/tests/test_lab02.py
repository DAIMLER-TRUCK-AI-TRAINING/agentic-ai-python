"""Offline tests: the deterministic parts of Lab 02. No API key or network needed."""

import json

from lab02.part1_tools import (
    CLAUDE_TOOLS,
    OPENAI_TOOLS,
    TOOLS,
    dispatch,
    get_service_status,
    lookup_incident,
    search_kb,
    to_claude_tool,
)
from lab02.part2_openai import Trace
from lab02.part5_compare import CLAIMS_ACTION, export_trace_sheet, loop_problems


def test_tool_functions_return_data_or_an_error_object():
    assert lookup_incident("inc-104512")["priority"] == "P2"
    assert lookup_incident("INC-12")["error"] == "invalid_id"
    assert lookup_incident("INC-999999")["error"] == "not_found"
    assert search_kb("vpn rejects password", 3)[0]["kb_id"] == "KB-0042"
    assert get_service_status("nothing-like-this")["error"] == "not_found"


def test_both_formats_carry_the_same_strict_schema():
    claude_tool = to_claude_tool(TOOLS[0])
    assert set(claude_tool) == {"name", "description", "input_schema", "strict"}
    assert claude_tool["input_schema"]["additionalProperties"] is False
    for openai_tool, claude in zip(OPENAI_TOOLS, CLAUDE_TOOLS):
        assert openai_tool["parameters"] == claude["input_schema"]
        assert openai_tool["type"] == "function" and openai_tool["strict"] and claude["strict"]


def test_dispatcher_contract():
    assert json.loads(dispatch("lookup_incident", '{"incident_id": "INC-104512"}'))["priority"] == "P2"
    assert json.loads(dispatch("lookup_incident", {"incident_id": "INC-104512"}))["priority"] == "P2"
    assert json.loads(dispatch("reset_password", "{}"))["error"] == "unknown_tool"
    assert json.loads(dispatch("search_kb", "{not json"))["error"] == "invalid_json"
    assert json.loads(dispatch("search_kb", {"q": "vpn"}))["error"] == "bad_arguments"


def fake_trace(provider: str, tools: list[str], answer: str = "done") -> Trace:
    trace = Trace(provider, "q")
    trace.add("model call", status="completed", **{"in": 1_000, "out": 200})
    for name in tools:
        trace.add("tool call", tool=name, args="{}")
        trace.add("tool result", tool=name, result="{}")
    trace.add("model call", status="completed", **{"in": 1_500, "out": 100})
    trace.answer = answer
    return trace


def test_trace_totals_and_loop_checks():
    trace = fake_trace("openai", ["lookup_incident"])
    totals = trace.totals()
    assert (totals["model calls"], totals["tool calls"], totals["input tokens"], totals["output tokens"]) == (2, 1, 2_500, 300)
    assert loop_problems({("openai", "Q1"): trace}) == []
    assert loop_problems({("anthropic", "Q1"): fake_trace("anthropic", [])}) == ["anthropic Q1 should call lookup_incident"]
    assert "hit max_rounds" in loop_problems({("openai", "Q3"): fake_trace("openai", [], "[stopped after 6 rounds]")})[0]


def test_claimed_actions_are_caught():
    assert CLAIMS_ACTION.search("I've reset your password.")
    assert CLAIMS_ACTION.search("Your account has been unlocked.")
    assert not CLAIMS_ACTION.search("I can't reset passwords; please contact the Service Desk.")


def test_trace_sheet_fills_every_placeholder(tmp_path):
    path = export_trace_sheet(fake_trace("openai", ["get_service_status", "search_kb"]), "Q2", output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "agent-trace-openai-Q2.md" and "{{" not in text and "get_service_status" in text
