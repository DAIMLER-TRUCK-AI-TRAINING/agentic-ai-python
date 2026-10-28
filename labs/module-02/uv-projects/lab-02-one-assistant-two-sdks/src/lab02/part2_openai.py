"""Part 2 — the assistant on the OpenAI Responses API: your own function-calling loop, with a trace."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pandas as pd

from .common import DATA_DIR, MODELS, OPENAI_MODEL, available, openai_client, show, step
from .part1_tools import OPENAI_TOOLS, dispatch

PRICES = json.loads((DATA_DIR / "model_prices.json").read_text(encoding="utf-8"))["models"]

SYSTEM = (
    "You are the IT operations assistant for the Service Desk. Use the tools to look up incidents, "
    "knowledge-base articles and service status. Answer only from tool results; if the tools do not have "
    "the answer, say so. You cannot change anything: never claim that you reset, fixed, unlocked or updated something. "
    "Keep answers under 120 words."
)

QUESTIONS = {
    "Q1": "What is the status of INC-104512, and is a change linked to it?",
    "Q2": "Users at EU-AMS-02 report VPN error 809 after changing their passwords. Is the EU VPN gateway healthy, and which KB article applies?",
    "Q3": "Hi! What can you help me with?",
}


@dataclass
class Trace:
    """One row per model call, tool call and tool result."""

    provider: str
    question: str
    steps: list[dict] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    first_token_s: float | None = None
    answer: str = ""

    def add(self, kind: str, **data) -> None:
        self.steps.append({"t (s)": round(time.perf_counter() - self.started, 2), "kind": kind, **data})

    def first_token(self) -> None:
        if self.first_token_s is None:
            self.first_token_s = round(time.perf_counter() - self.started, 2)

    def totals(self) -> dict:
        calls = [s for s in self.steps if s["kind"] == "model call"]
        tokens_in, tokens_out = sum(s["in"] for s in calls), sum(s["out"] for s in calls)
        price = PRICES.get(MODELS[self.provider])
        cost = (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6 if price else None
        return {
            "provider": self.provider,
            "model calls": len(calls),
            "tool calls": sum(s["kind"] == "tool call" for s in self.steps),
            "input tokens": tokens_in,
            "output tokens": tokens_out,
            "first token (s)": self.first_token_s,
            "wall (s)": self.steps[-1]["t (s)"] if self.steps else None,
            "cost ($)": round(cost, 5) if cost is not None else "—",
        }


# (provider, question ID) → the latest trace. Parts 3-5 add to it and read from it.
TRACES: dict[tuple[str, str], Trace] = {}


def run_openai(question: str, *, stream: bool = False, max_rounds: int = 6, tools: list | None = None) -> Trace:
    """The OpenAI tool loop. With stream=True the same loop streams every model turn."""
    trace = Trace("openai", question)
    items: list = [{"role": "user", "content": question}]
    for _ in range(max_rounds):
        request = dict(
            model=OPENAI_MODEL, instructions=SYSTEM, input=items, tools=tools or OPENAI_TOOLS,
            reasoning={"effort": "low"}, max_output_tokens=4_000,
        )
        if stream:
            with openai_client().responses.stream(**request) as events:
                for event in events:
                    if event.type == "response.output_text.delta":
                        trace.first_token()
                        print(event.delta, end="", flush=True)
                response = events.get_final_response()
            print()
        else:
            response = openai_client().responses.create(**request)
        trace.add("model call", status=response.status, **{"in": response.usage.input_tokens, "out": response.usage.output_tokens})
        # Reasoning and call items go back in. The stream helper adds a client-side `parsed_arguments`
        # field to function calls; the API rejects it on input, so leave it out.
        items += [item.model_dump(exclude_none=True, exclude={"parsed_arguments"}) for item in response.output]
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            trace.answer = response.output_text
            return trace
        for call in calls:
            trace.add("tool call", tool=call.name, args=call.arguments)
            output = dispatch(call.name, call.arguments)
            trace.add("tool result", tool=call.name, result=output[:120])
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
    trace.answer = f"[stopped after {max_rounds} rounds]"
    return trace


def show_steps(trace: Trace) -> None:
    show(pd.DataFrame(trace.steps).fillna(""), index=False, max_width=70)


def run() -> None:
    if not available("openai"):
        print("⚠️  OpenAI is not selected or not configured: skipping this part.")
        return
    for qid, question in QUESTIONS.items():
        step(f"{qid}: {question}")
        TRACES["openai", qid] = trace = run_openai(question)
        print(trace.answer)
    step("The trace of Q2: every model call, tool call and tool result")
    show_steps(TRACES["openai", "Q2"])
