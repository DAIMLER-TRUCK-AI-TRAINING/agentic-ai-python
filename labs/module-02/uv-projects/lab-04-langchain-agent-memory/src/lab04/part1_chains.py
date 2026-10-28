"""Part 1 — LangChain building blocks: prompt | model | parser, batching, and structured output on both providers."""

from __future__ import annotations

import time
from typing import Literal

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .common import DATA_DIR, chat_model, main_provider, other_provider, show, step

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a service-desk analyst. Write one line of at most 15 words for the incident queue."),
    ("human", "Ticket: {ticket}"),
])
TICKETS = pd.read_csv(DATA_DIR / "triage_tickets.csv").query("split == 'test'").text.tolist()[:6]
POLICY = (DATA_DIR / "triage_policy.md").read_text(encoding="utf-8")
TRIAGE_PROMPT = ChatPromptTemplate.from_messages([("system", "Triage the ticket under this policy:\n{policy}"), ("human", "{ticket}")])


class TriageLabel(BaseModel):
    category: Literal["access", "network", "endpoint", "email_collab", "business_app", "platform", "security"]
    priority: Literal["P1", "P2", "P3", "P4"]
    assignment_group: Literal["Service Desk", "Network Ops", "Finance", "Platform Engineering", "Security Operations"]


def run() -> None:
    llm = chat_model(main_provider())
    if llm is None:
        print(f"⚠️  {main_provider()} is not configured: skipping this part.")
        return
    step(f"prompt | model | parser: the smallest useful chain ({main_provider()})")
    one_liner = SUMMARY_PROMPT | llm | StrOutputParser()
    print(one_liner.invoke({"ticket": TICKETS[0]}))

    started = time.perf_counter()
    lines = one_liner.batch([{"ticket": t} for t in TICKETS], config={"max_concurrency": 6})
    print(f"\nbatch of {len(lines)} in {time.perf_counter() - started:.1f}s (concurrent calls, same chain):")
    print("\n".join(f"- {line}" for line in lines))

    step("Structured output through LangChain: the same chain, two models")
    rows = []
    for name in (main_provider(), other_provider()):
        model = chat_model(name)
        if model is None:
            continue
        chain = TRIAGE_PROMPT | model.with_structured_output(TriageLabel, method="json_schema")
        for ticket in TICKETS[:3]:
            label = chain.invoke({"policy": POLICY, "ticket": ticket})
            rows.append({"provider": name, "ticket": ticket[:60], **label.model_dump()})
    show(pd.DataFrame(rows), index=False)
    ok = isinstance(lines[0], str) and lines[0].strip() and all(r["priority"] in {"P1", "P2", "P3", "P4"} for r in rows)
    print(f"\n{'✅' if ok else '⚠️ '} the chains returned a string and validated TriageLabels")
