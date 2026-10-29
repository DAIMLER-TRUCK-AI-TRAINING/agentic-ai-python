"""Part 1 — the support-doc corpus as records, and the model alone on an internal question."""

from __future__ import annotations

import functools

import pandas as pd

from .common import llm, main_provider, needs_model, rk, show, step

QUESTION = "An LH-40 built in March 2026 shows FC-ADB-0412. Which bulletin applies, and which part replaces the dosing valve?"


@functools.cache
def records() -> list[dict]:
    """One record per Markdown file in data/corpus/; the front matter becomes metadata."""
    return rk.load_corpus()


@functools.cache
def chat_model(provider: str):
    """LangChain chat model; low reasoning effort on OpenAI's own API."""
    return llm.get_chat_model(provider, **({"reasoning": {"effort": "low"}} if provider == "openai" else {}))


def run() -> None:
    step("One record per Markdown file; the front matter becomes metadata")
    columns = ("doc_id", "doc_type", "product", "audience", "language", "title")
    show(pd.DataFrame([{k: r["metadata"][k] for k in columns} for r in records()]), index=False, max_width=60)

    if needs_model("the question without retrieval"):
        step(f"The model alone ({main_provider()}): it has never seen these internal documents")
        print(chat_model(main_provider()).invoke(QUESTION).text)
