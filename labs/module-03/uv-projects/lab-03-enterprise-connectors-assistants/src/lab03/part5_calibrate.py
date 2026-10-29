"""Part 5 — calibrate the score floor on labelled questions, then run the same assistants on the other provider."""

from __future__ import annotations

import pandas as pd
from llama_index.core import QueryBundle

from .common import comparison_provider, main_provider, needs_model, rk, show, step
from .part3_access import access_filters, knowledge_base
from .part4_assistants import CASES, POLICY_SEARCH, run_cases


def best_scores() -> pd.DataFrame:
    """The best re-ranked score per question, answerable or not (retrieval and re-ranking only, no LLM)."""
    index, _, _ = knowledge_base()
    rerank = rk.reranker(1)
    rows = []
    for name, (profile, user, questions) in CASES.items():
        retriever = index.as_retriever(similarity_top_k=2 * profile.k, filters=access_filters(user, list(profile.doc_types)))
        for question, answerable in questions:
            best = rerank.postprocess_nodes(retriever.retrieve(question), query_bundle=QueryBundle(question))[0]
            rows.append({"assistant": name, "question": question[:60], "answerable": answerable,
                         "best score": round(float(best.score), 2), "best doc": best.node.metadata["doc_id"]})
    return pd.DataFrame(rows).sort_values("best score")


def run() -> None:
    step("The best re-ranked score per question, answerable vs not: where should the floor go?")
    scores = best_scores()
    lowest_ok = scores.loc[scores["answerable"], "best score"].min()
    highest_no = scores.loc[~scores["answerable"], "best score"].max()
    print(f"lowest answerable: {lowest_ok} · highest unanswerable: {highest_no} · floor in use: {POLICY_SEARCH.floor}")
    show(scores, index=False, max_width=60)
    print(f"\n{'✅' if highest_no < POLICY_SEARCH.floor < lowest_ok else '❌'} the floor separates answerable from unanswerable questions")

    if not needs_model("the provider comparison") or not (partner := comparison_provider()):
        return
    step(f"The same assistants on {partner}: retrieval is identical, only the writer changes")
    both = pd.concat([run_cases(main_provider()), run_cases(partner)])
    table = both.pivot_table(index=["assistant", "question"], columns="provider", values=["answered", "s"], aggfunc="first")
    table.columns = [f"{provider} {value}" for value, provider in table.columns]
    show(table.reset_index(), index=False, max_width=60)
    print("\n" + both[both["provider"] == partner].iloc[1]["answer"])
    print("\n📝 Write-up: which floor would you ship, and what margin does your calibration leave?")
