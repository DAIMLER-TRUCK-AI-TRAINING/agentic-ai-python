"""Part 4 — conversation memory: stateless calls, then a checkpointed buffer over a 10-turn conversation."""

from __future__ import annotations

import json

import pandas as pd
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from .common import DATA_DIR, chat_model, main_provider, show, step
from .part2_tools import query_ops_db, search_kb

CONVO = json.loads((DATA_DIR / "support_conversation.json").read_text(encoding="utf-8"))
CHAT_SYSTEM = (
    "You are a Service Desk assistant chatting with one employee. Use search_kb and query_ops_db when they help. "
    "Answer in at most 60 words. You cannot change anything."
)
CHAT_TOOLS = [search_kb, query_ops_db]

# strategy name → one row per turn. Part 5 adds "window" and "summary" next to "buffer".
MEMORY_RUNS: dict[str, pd.DataFrame] = {}


def run_conversation(agent, thread_id: str) -> pd.DataFrame:
    """Run the whole conversation through one memory strategy and measure every turn."""
    config = {"configurable": {"thread_id": thread_id}}
    rows, seen_ids = [], set()
    for n, turn in enumerate(CONVO["turns"], start=1):
        out = agent.invoke({"messages": [{"role": "user", "content": turn}]}, config)
        # Memory middleware may remove old messages, so find this turn's model calls by message ID.
        ai = [m for m in out["messages"] if m.type == "ai" and m.usage_metadata and m.id not in seen_ids]
        seen_ids |= {m.id for m in out["messages"]}
        rows.append({
            "turn": n, "input tokens": sum(m.usage_metadata["input_tokens"] for m in ai),
            "messages in state": len(out["messages"]), "reply": out["messages"][-1].text,
        })
    return pd.DataFrame(rows)


def probe_results(frame: pd.DataFrame) -> dict:
    """Did the late turns still contain the early facts? (substring checks on the replies)"""
    return {
        f"turn {p['turn']}": all(s.lower() in frame.loc[p["turn"] - 1, "reply"].lower() for s in p["must_contain"])
        for p in CONVO["probes"]
    }


def buffer_run(llm) -> pd.DataFrame:
    """The buffer strategy, run once per process: Part 5 reuses it."""
    if "buffer" not in MEMORY_RUNS:
        agent = create_agent(llm, CHAT_TOOLS, system_prompt=CHAT_SYSTEM, checkpointer=InMemorySaver())
        MEMORY_RUNS["buffer"] = run_conversation(agent, "buffer")
    return MEMORY_RUNS["buffer"]


def run() -> None:
    llm = chat_model(main_provider())
    if llm is None:
        print(f"⚠️  {main_provider()} is not configured: skipping this part.")
        return
    step("Without memory, every call starts from nothing. With a checkpointer, a thread remembers")
    stateless = create_agent(llm, CHAT_TOOLS, system_prompt=CHAT_SYSTEM)
    stateless.invoke({"messages": [{"role": "user", "content": CONVO["turns"][0]}]})
    reply = stateless.invoke({"messages": [{"role": "user", "content": "What is my laptop's asset tag?"}]})
    print("stateless →", reply["messages"][-1].text[:160])

    with_memory = create_agent(llm, CHAT_TOOLS, system_prompt=CHAT_SYSTEM, checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "demo-1"}}
    with_memory.invoke({"messages": [{"role": "user", "content": CONVO["turns"][0]}]}, thread)
    reply = with_memory.invoke({"messages": [{"role": "user", "content": "What is my laptop's asset tag?"}]}, thread)
    print("buffer    →", reply["messages"][-1].text[:160])

    step(f"The {len(CONVO['turns'])}-turn conversation with buffer memory ({main_provider()})")
    buffer = buffer_run(llm)
    show(buffer, index=False, max_width=80)
    probes = probe_results(buffer)
    print("probes:", probes)
    grew = buffer["messages in state"].is_monotonic_increasing
    print(f"\n{'✅' if grew else '⚠️ '} the state grew every turn, and so did the input tokens")
    if not all(probes.values()):
        print("⚠️  the buffer run missed a probe. Read the reply; models sometimes paraphrase.")
