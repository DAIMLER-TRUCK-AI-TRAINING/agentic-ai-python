"""Part 3 — retrieve, augment, generate: an LCEL chain whose citations are checked in code."""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .common import check, comparison_provider, main_provider, needs_model, rk, show, step
from .part1_corpus import QUESTION, chat_model
from .part2_faiss import retrieve

QUESTIONS = {
    "paraphrase": "The exhaust fluid light is flashing. Can the driver keep going?",
    "German": "Was bedeutet der Fehlercode FC-BRK-0705?",
    "unanswerable": "What is the list price of a new LH-40 tractor?",
}

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the Daimler Truck after-sales assistant. Answer only from the sources and cite the "
               "source id in square brackets after each claim, for example [KB-0610]. If the sources do not "
               "answer the question, say so."),
    ("human", "<sources>\n{sources}\n</sources>\n\nQuestion: {question}"),
])


def format_sources(hits: list[Document]) -> str:
    """Augment: one <source> per document, its chunks in document order, so no id appears twice."""
    by_doc: dict[str, list[Document]] = {}
    for h in hits:
        by_doc.setdefault(h.metadata["doc_id"], []).append(h)
    return "\n\n".join(
        f'<source id="{doc_id}" title="{parts[0].metadata["title"]}">\n'
        + "\n[...]\n".join(p.page_content for p in sorted(parts, key=lambda p: p.metadata["start_index"]))
        + "\n</source>"
        for doc_id, parts in by_doc.items()
    )


def rag_chain(model):
    return {"sources": RunnableLambda(retrieve) | format_sources, "question": RunnablePassthrough()} | RAG_PROMPT | model | StrOutputParser()


def run() -> None:
    if not needs_model("the RAG chain"):
        return
    step(f"The chain on {main_provider()}: retrieve | format_sources | prompt | model | parser")
    answer = rag_chain(chat_model(main_provider())).invoke(QUESTION)
    retrieved = {h.metadata["doc_id"] for h in retrieve(QUESTION)}
    print(answer, "\n")
    print(f"cited: {sorted(rk.cited_ids(answer))} · retrieved: {sorted(retrieved)}")

    step("Three more questions: a paraphrase, a German question, and one the corpus cannot answer")
    runs = []
    for kind, question in QUESTIONS.items():
        text = rag_chain(chat_model(main_provider())).invoke(question)
        runs.append({"kind": kind, "retrieved": sorted({h.metadata["doc_id"] for h in retrieve(question)}),
                     "cited": sorted(rk.cited_ids(text)), "answer": text[:150]})
    show(pd.DataFrame(runs), index=False, max_width=70)

    if partner := comparison_provider():
        step(f"The same chain on {partner}: only the model object changes")
        other = rag_chain(chat_model(partner)).invoke(QUESTION)
        print(f"{other}\n\ncited: {sorted(rk.cited_ids(other))}")

    def check_part3():
        assert rk.cited_ids(answer), "the answer should cite at least one source"
        assert rk.cited_ids(answer) <= retrieved, f"cited ids that were not retrieved: {rk.cited_ids(answer) - retrieved}"
        for row in runs:
            assert set(row["cited"]) <= set(row["retrieved"]), row

    check("Part 3 — grounded answers with valid citations", check_part3)
