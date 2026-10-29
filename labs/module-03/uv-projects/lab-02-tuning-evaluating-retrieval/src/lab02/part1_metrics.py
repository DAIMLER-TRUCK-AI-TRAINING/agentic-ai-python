"""Part 1 — the labelled queries and the four retrieval metrics, with the harness that averages them."""

from __future__ import annotations

import json

import pandas as pd

from .common import DATA_DIR, check, show, step

QUERIES = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
K = 4


def retrieval_metrics(ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Metrics for the top k of `ids` (doc_ids in rank order; one entry per retrieved chunk, so an id can repeat).

    - precision: relevant entries in the top k / k
    - recall:    distinct relevant documents found in the top k / number of relevant documents
    - hit_rate:  1.0 if any relevant document is in the top k, else 0.0
    - mrr:       1 / rank of the first relevant entry (ranks start at 1), or 0.0 if none
    """
    ids = ids[:k]
    first = next((rank for rank, i in enumerate(ids, 1) if i in relevant), None)
    return {
        "precision": sum(i in relevant for i in ids) / k,
        "recall": len(relevant & set(ids)) / len(relevant),
        "hit_rate": float(first is not None),
        "mrr": 1 / first if first else 0.0,
    }


def evaluate(retrieve_ids, queries=QUERIES, k: int = K) -> dict[str, float]:
    """Mean metrics over the answerable labelled queries; `retrieve_ids(query)` returns doc_ids in rank order."""
    rows = [retrieval_metrics(retrieve_ids(q["query"]), set(q["relevant"]), k) for q in queries if q["relevant"]]
    return {m: round(sum(r[m] for r in rows) / len(rows), 2) for m in rows[0]}


def check_part1():
    m = retrieval_metrics(["A", "B", "A", "C"], {"A", "D"}, 4)
    assert m == {"precision": 0.5, "recall": 0.5, "hit_rate": 1.0, "mrr": 1.0}, m
    m = retrieval_metrics(["B", "C", "D", "E", "A"], {"D"}, 4)
    assert m["mrr"] == 1 / 3 and m["hit_rate"] == 1.0 and m["precision"] == 0.25, m
    assert retrieval_metrics(["B", "C"], {"A"}, 4) == {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0}


def run() -> None:
    step("The labelled set: answerable queries by kind, plus unanswerable ones")
    print(pd.Series([q["kind"] for q in QUERIES]).value_counts().to_dict())
    show(pd.DataFrame(QUERIES).head(8), index=False, max_width=70)

    step("The four metrics on cases you can work out by hand")
    check("Part 1 — retrieval metrics", check_part1)
