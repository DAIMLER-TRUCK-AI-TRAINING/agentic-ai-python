"""Part 3 — dense, BM25, hybrid (reciprocal rank fusion) and hybrid + cross-encoder re-rank, per query."""

from __future__ import annotations

import functools

import pandas as pd
from llama_index.core import QueryBundle

from .common import check, rk, show, step
from .part1_metrics import K, QUERIES, evaluate, retrieval_metrics
from .part2_chunking import indexes


@functools.cache
def retrievers() -> dict:
    """name -> retrieve_ids(query), all over the markdown-section nodes."""
    index, nodes = indexes()["markdown"]
    dense = index.as_retriever(similarity_top_k=K)
    bm25 = rk.bm25_retriever(nodes, K)
    hybrid = rk.hybrid_retriever(index, nodes, k=K, pool=10)  # reciprocal rank fusion, no LLM
    wide = rk.hybrid_retriever(index, nodes, k=10, pool=10)
    rerank = rk.reranker(K)  # RERANK_MODEL, a local cross-encoder

    def hybrid_rerank_ids(query: str) -> list[str]:
        return rk.doc_ids(rerank.postprocess_nodes(wide.retrieve(query), query_bundle=QueryBundle(query)))

    return {
        "dense": lambda q: rk.doc_ids(dense.retrieve(q)),
        "bm25": lambda q: rk.doc_ids(bm25.retrieve(q)),
        "hybrid": lambda q: rk.doc_ids(hybrid.retrieve(q)),
        "hybrid + rerank": hybrid_rerank_ids,
    }


def run() -> None:
    step("Four retrievers over the markdown-section nodes")
    show(pd.DataFrame({name: evaluate(fn) for name, fn in retrievers().items()}).T)

    step("The averages hide the story: which queries does each retriever miss?")
    hits = pd.DataFrame({
        name: {q["qid"]: retrieval_metrics(fn(q["query"]), set(q["relevant"]), K)["hit_rate"] for q in QUERIES if q["relevant"]}
        for name, fn in retrievers().items()
    }).astype(int)
    misses = hits[hits.min(axis=1) < 1].join(pd.DataFrame(QUERIES).set_index("qid")[["query", "kind"]])
    show(misses, max_width=60)

    def check_part3():
        assert hits["hybrid"].sum() >= hits["dense"].sum(), hits.sum()
        assert evaluate(retrievers()["hybrid"])["recall"] >= 0.9

    check("Part 3 — hybrid retrieval", check_part3)
