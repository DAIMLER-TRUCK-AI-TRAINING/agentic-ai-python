"""Part 3 — streaming responses: time to first token, and what actually arrives on the wire."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .common import OPENAI_MODEL, available, openai_client, show, step
from .part1_tools import OPENAI_TOOLS
from .part2_openai import QUESTIONS, SYSTEM, TRACES, run_openai


def event_types(question: str) -> Counter:
    """Count the event types of one streamed turn."""
    with openai_client().responses.stream(
        model=OPENAI_MODEL, instructions=SYSTEM, input=question, tools=OPENAI_TOOLS,
        reasoning={"effort": "low"}, max_output_tokens=4_000,
    ) as events:
        return Counter(event.type for event in events)


def run() -> None:
    if not available("openai"):
        print("⚠️  OpenAI is not selected or not configured: skipping this part (Part 4 streams on Claude).")
        return
    step("Q2 again with stream=True: tool rounds stream too, but only the final turn has text to show")
    TRACES["openai", "Q2 streamed"] = streamed = run_openai(QUESTIONS["Q2"], stream=True)
    print(f"\nfirst token after {streamed.first_token_s}s · whole answer after {streamed.steps[-1]['t (s)']}s")

    step("The event types of one streamed turn (Q1, which calls a tool)")
    show(pd.Series(event_types(QUESTIONS["Q1"]), name="events").to_frame())
