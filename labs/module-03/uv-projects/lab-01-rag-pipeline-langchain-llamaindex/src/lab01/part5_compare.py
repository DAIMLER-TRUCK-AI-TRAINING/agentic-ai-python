"""Part 5 — the two pipelines side by side: which documents does each retrieve for the same questions?"""

from __future__ import annotations

import pandas as pd

from .common import rk, show, step
from .part1_corpus import QUESTION
from .part2_faiss import retrieve
from .part3_chain import QUESTIONS
from .part4_llamaindex import llama_pipeline


def run() -> None:
    step("Same questions, both pipelines (retrieval only, no model call)")
    _, _, index = llama_pipeline()
    li_retriever = index.as_retriever(similarity_top_k=4)
    rows = []
    for question in [QUESTION, *QUESTIONS.values()]:
        lc_ids = [h.metadata["doc_id"] for h in retrieve(question)]
        li_ids = rk.doc_ids(li_retriever.retrieve(question))
        rows.append({"question": question[:55], "LangChain + FAISS": lc_ids, "LlamaIndex + Chroma": li_ids,
                     "shared docs": len(set(lc_ids) & set(li_ids))})
    show(pd.DataFrame(rows), index=False, max_width=60)
    print("\n📝 Fill in the observation table in the notebook: chunk unit, retriever, metadata, prompt, best fit.")
