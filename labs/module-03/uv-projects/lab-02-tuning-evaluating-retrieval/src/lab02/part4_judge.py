"""Part 4 — answer five questions with the hybrid retriever, then grade relevance with a small-tier judge on both providers."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from pydantic import BaseModel, Field

from .common import check, needs_model, providers, rk, show, step
from .part1_metrics import K, QUERIES
from .part2_chunking import indexes

QA_PROMPT = PromptTemplate(
    "Answer only from the sources and cite the doc_id in square brackets, for example [KB-0610]. "
    "If the sources do not answer the question, say so.\n\n<sources>\n{context_str}\n</sources>\n\n"
    "Question: {query_str}\nAnswer: "
)
EVAL_SET = [q for q in QUERIES if q["qid"] in {"q05", "q08", "q10", "u01", "u02"}]


class Relevance(BaseModel):
    score: Literal[1, 2, 3, 4, 5] = Field(description="5 = fully addresses the question, 1 = not at all")
    refused: bool = Field(description="True if the answer says the information is not available")
    reason: str = Field(description="One sentence")


JUDGE = PromptTemplate(
    "You grade answer relevance: does the ANSWER address the QUESTION that was asked? Ignore whether it is "
    "true; groundedness is graded separately.\n\nQUESTION: {question}\n\nANSWER: {answer}"
)


def passed(q: dict, verdict: Relevance) -> bool:
    """Unanswerable questions pass on a refusal, answerable ones on a score of 4 or 5."""
    return verdict.refused if not q["relevant"] else verdict.score >= 4


def run() -> None:
    if not needs_model("the answers and the judge"):
        return
    step("Answer five questions with the hybrid retriever (two of them unanswerable)")
    index, nodes = indexes()["markdown"]
    engine = RetrieverQueryEngine.from_args(rk.hybrid_retriever(index, nodes, k=K, pool=10), text_qa_template=QA_PROMPT)
    answers = {q["qid"]: str(engine.query(q["query"])) for q in EVAL_SET}
    for q in EVAL_SET:
        print(f"{q['qid']} {q['query']}\n    -> {' '.join(answers[q['qid']].split())[:160]}\n")

    step(f"A small-tier judge with a structured verdict: {', '.join(providers())}")
    judged = []
    for provider in providers():
        judge = rk.llama_llm(provider, small=True)
        for q in EVAL_SET:
            verdict = judge.structured_predict(Relevance, JUDGE, question=q["query"], answer=answers[q["qid"]])
            judged.append({"judge": provider, "qid": q["qid"], "answerable": bool(q["relevant"]),
                           "score": verdict.score, "refused": verdict.refused, "pass": passed(q, verdict)})
    judged = pd.DataFrame(judged)
    table = judged.pivot_table(index=["qid", "answerable"], columns="judge", values=["score", "pass"], aggfunc="first")
    table.columns = [f"{judge} {value}" for value, judge in table.columns]
    show(table)

    def check_part4():
        assert len(judged) == len(EVAL_SET) * judged["judge"].nunique()
        assert set(judged["answerable"]) == {True, False}

    check("Part 4 — answer relevance judged", check_part4)
    print("\n📝 Fill in templates/retrieval-eval-rubric.md: where do the judges disagree, and why?")
