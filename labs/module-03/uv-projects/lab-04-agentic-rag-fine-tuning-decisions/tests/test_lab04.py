"""Offline tests: the deterministic parts of Lab 04. No API key, model download or network needed."""

import asyncio
import threading
from dataclasses import replace

import pytest

from lab04.common import rk
from lab04.part1_tool import DOC_TYPES, search_knowledge
from lab04.part2_agent import TASKS
from lab04.part3_canvases import CANVASES, DECISIONS, check_part3
from lab04.part4_critique import CRITIC, Critique


def test_unknown_doc_type_names_the_valid_ones():
    """Checked before any search, so the model can correct itself on the next call."""
    result = search_knowledge("FC-BAT-2201", doc_type="manual")
    assert result.startswith("NO_RESULTS: unknown doc_type 'manual'")
    assert all(t in result for t in DOC_TYPES)


def test_the_docstring_is_a_complete_tool_description():
    doc = search_knowledge.__doc__
    assert all(f"({t})" in doc for t in DOC_TYPES)
    assert "do not use it for greetings" in doc and "JSON" in doc


def test_agent_tool_runs_the_function_on_the_event_loop():
    threads = []

    def probe(query: str) -> str:
        """Return the query."""
        threads.append(threading.current_thread())
        return query.upper()

    tool = rk.agent_tool(probe, "search_knowledge")
    output = asyncio.run(tool.acall(query="adblue"))
    assert tool.metadata.name == "search_knowledge" and output.content == "ADBLUE"
    assert threads == [threading.main_thread()], "the tool must not run in a worker thread"


def test_the_tasks_cover_small_talk_two_part_and_german():
    assert set(TASKS) == {"thanks", "two-part", "German"} and TASKS["thanks"][1] == set()


def test_the_reference_canvases_are_complete():
    check_part3()
    assert {c.scenario[0] for c in CANVASES} == {"A", "B", "C", "D"}
    assert {c.decision for c in CANVASES} == DECISIONS, "the four cards exercise all four options"


def test_an_unfinished_canvas_fails_the_check():
    with pytest.raises(AssertionError):
        check_part3([replace(CANVASES[0], decision="TODO"), *CANVASES[1:]])


def test_the_critique_contract():
    critique = Critique(agrees=False, strongest_counterargument="…", eval_to_run_first="A few-shot baseline.")
    assert set(critique.model_dump()) == {"agrees", "strongest_counterargument", "eval_to_run_first"}
    assert set(CRITIC.template_vars) == {"scenario", "gap", "decision", "why", "overturn_eval"}
