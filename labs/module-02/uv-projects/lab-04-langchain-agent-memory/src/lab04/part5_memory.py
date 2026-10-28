"""Part 5 — window memory vs summary memory, against the buffer, on the same conversation."""

from __future__ import annotations

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain.messages import AnyMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .common import chat_model, main_provider, show, step
from .part2_tools import summariser
from .part4_buffer import CHAT_SYSTEM, CHAT_TOOLS, CONVO, MEMORY_RUNS, buffer_run, probe_results, run_conversation

WINDOW_TURNS = 3
# Claude's thinking blocks are bound to the exact history they were produced in. Once older turns are
# dropped or summarised, a kept turn's thinking block no longer matches and the API returns a 400.
# The fix: strip thinking from every turn before the current one, deterministically, on every call.
# (OpenAI reasoning items are left alone: they belong to the function calls that follow them.)
CLAUDE_THINKING = {"thinking", "redacted_thinking"}


def keep_last_turns(messages: list[AnyMessage], k: int) -> list[AnyMessage]:
    """Return the messages from the k-th most recent HumanMessage to the end.

    A turn starts at a HumanMessage and includes every AI and tool message after it, so cutting at a
    HumanMessage never leaves a ToolMessage without the AIMessage that requested it (providers reject that).
    If there are k or fewer human turns, return the list unchanged.
    """
    starts = [i for i, m in enumerate(messages) if m.type == "human"]
    if len(starts) <= k:
        return list(messages)
    return list(messages[starts[-k]:])


def without_old_thinking(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Copy of `messages` with Claude thinking blocks removed from AI messages before the latest human message."""
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
    cleaned = []
    for i, m in enumerate(messages):
        if i < last_human and m.type == "ai" and isinstance(m.content, list):
            blocks = [b for b in m.content if not (isinstance(b, dict) and b.get("type") in CLAUDE_THINKING)]
            if len(blocks) != len(m.content):
                m = m.model_copy(update={"content": blocks})
        cleaned.append(m)
    return cleaned


@before_model
def window_memory(state, runtime):
    """Before each model call, drop everything older than the last WINDOW_TURNS turns."""
    kept = keep_last_turns(state["messages"], WINDOW_TURNS)
    if len(kept) == len(state["messages"]):
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *without_old_thinking(kept)]}


@before_model
def strip_old_thinking(state, runtime):
    """Runs after summarisation: same message IDs, so the reducer replaces the messages in place."""
    changed = [c for c, m in zip(without_old_thinking(state["messages"]), state["messages"]) if c is not m]
    return {"messages": changed} if changed else None


class TurnBoundarySummarization(SummarizationMiddleware):
    """Summarise only when a new user turn starts. The built-in trigger is checked before every model call,
    so it can fire in the middle of a tool round and rewrite the history under that round's thinking block."""

    def before_model(self, state, runtime):
        if state["messages"][-1].type != "human":
            return None
        return super().before_model(state, runtime)


def summary_memory():
    return TurnBoundarySummarization(model=summariser(), trigger=("tokens", 1_500), keep=("messages", 6))


def token_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Input tokens per turn, one row per strategy, plus the total."""
    tokens = pd.DataFrame({name: run["input tokens"].to_numpy() for name, run in runs.items()}, index=range(1, len(CONVO["turns"]) + 1))
    tokens.index.name = "turn"
    return tokens.T.assign(total=tokens.sum())


def run() -> None:
    llm = chat_model(main_provider())
    if llm is None:
        print(f"⚠️  {main_provider()} is not configured: skipping this part.")
        return
    step("Buffer (from Part 4), window of the last 3 turns, and summary memory on the same conversation")
    buffer_run(llm)
    MEMORY_RUNS["window"] = run_conversation(
        create_agent(llm, CHAT_TOOLS, system_prompt=CHAT_SYSTEM, middleware=[window_memory], checkpointer=InMemorySaver()),
        "window",
    )
    MEMORY_RUNS["summary"] = run_conversation(
        create_agent(llm, CHAT_TOOLS, system_prompt=CHAT_SYSTEM, middleware=[summary_memory(), strip_old_thinking],
                     checkpointer=InMemorySaver()),
        "summary",
    )
    print(f"Input tokens per turn ({main_provider()}):")
    show(token_table(MEMORY_RUNS))

    step("Recall: did the late turns still contain the early facts?")
    show(pd.DataFrame({name: probe_results(run) for name, run in MEMORY_RUNS.items()}))
    for name in ("window", "summary"):
        print(f"{name} · turn 9 reply: {MEMORY_RUNS[name].loc[8, 'reply'][:150]}")
    print("\n📝 Write-up: which strategy would you ship for the Service Desk chat, with which settings, and why?")
