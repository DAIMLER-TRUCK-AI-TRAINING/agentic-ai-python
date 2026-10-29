"""Part 2 — a LlamaIndex FunctionAgent that decides whether, where and how often to search, against the 2-step pipeline."""

from __future__ import annotations

import asyncio

import pandas as pd
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult

from .common import check, needs_model, providers, rk, show, step
from .part1_tool import DOC_TYPES, USER, knowledge_base, search_knowledge

SYSTEM = (
    "You are the Daimler Truck after-sales assistant for workshop engineers. Search before you answer any "
    "question about vehicles, fault codes, repairs, warranty, policies or IT procedures, and cite doc_ids in "
    "square brackets. Answer only from what the search returned. Search at most 3 times per question."
)
# task -> (message, doc_ids the answer must cite)
TASKS = {
    "thanks": ("Thanks, that fixed it!", set()),
    "two-part": ("An ED-26 shows FC-BAT-2201 during DC fast charging. Above which temperature spread is the "
                 "charging power limited, and is a battery module replacement covered by warranty on a "
                 "5-year-old truck?", {"POL-WAR-007"}),
    "German": ("Ein OTA-Update hängt in der Werkstatt bei 60 %. Was soll ich tun?", {"TSB-2026-052"}),
}


async def run_agent(provider: str, task: str) -> dict:
    """One agent run with a trace of every search: the queries, and the doc_ids the tool returned."""
    agent = FunctionAgent(tools=[rk.agent_tool(search_knowledge, "search_knowledge")],  # runs on the event loop
                          llm=rk.llama_llm(provider), system_prompt=SYSTEM)
    searches, returned = [], set()
    with rk.Timer() as t:
        handler = agent.run(user_msg=task)
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                searches.append(event.tool_kwargs)
                returned |= rk.cited_ids(str(event.tool_output.content))
        answer = str(await handler)
    return {"searches": searches, "returned": returned, "answer": answer, "s": t.seconds}


async def agent_runs() -> pd.DataFrame:
    rows = []
    for provider in providers():
        for name, (task, expected) in TASKS.items():
            run = await run_agent(provider, task)
            cited = rk.cited_ids(run["answer"])
            rows.append({"provider": provider, "task": name, "searches": len(run["searches"]),
                         "queries": [s.get("query", "")[:40] + f" ({s.get('doc_type', 'any')})" for s in run["searches"]],
                         "cited": sorted(cited), "cited ⊆ returned": cited <= run["returned"],
                         "expected cited": expected <= cited, "s": run["s"], "answer": run["answer"]})
    return pd.DataFrame(rows)


def two_step_runs() -> pd.DataFrame:
    """The fixed pipeline on the same tasks: one search per message, whatever it says."""
    index, _ = knowledge_base()
    two_step = rk.make_assistant(rk.RagProfile("2-step", DOC_TYPES, "You are the Daimler Truck after-sales assistant."), USER, index)
    rows = []
    for name, (task, expected) in TASKS.items():
        with rk.Timer() as t:
            response = two_step.query(task)
        cited = rk.cited_ids(str(response))
        rows.append({"task": name, "searches": 1, "answered": rk.answered(response), "cited": sorted(cited),
                     "expected cited": expected <= cited, "s": t.seconds})
    return pd.DataFrame(rows)


def run() -> None:
    if not needs_model("the agent"):
        return
    knowledge_base()  # build the index before the agent's clock starts
    step(f"A FunctionAgent with one tool, and a trace of every search: {', '.join(providers())}")
    runs = asyncio.run(agent_runs())
    show(runs.drop(columns="answer"), index=False, max_width=60)

    step("The two-part answer")
    print(runs.query("task == 'two-part'").iloc[0]["answer"])

    step("The fixed 2-step pipeline on the same tasks")
    show(two_step_runs(), index=False)

    def check_part2():
        for _, row in runs.iterrows():
            assert row["cited ⊆ returned"], (row["provider"], row["task"], row["cited"])
        assert (runs.query("task == 'thanks'")["searches"] == 0).all(), "the agent searched for 'thanks'"
        assert runs.query("task == 'two-part'")["expected cited"].all(), "the two-part answer missed POL-WAR-007"

    check("Part 2 — agentic RAG", check_part2)
    print("\n📝 Decide: agent or 2-step pipeline for the workshop assistant? Cite the table.")
