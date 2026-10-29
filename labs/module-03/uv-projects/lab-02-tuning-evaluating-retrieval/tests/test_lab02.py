"""Offline tests: the deterministic parts of Lab 02. No API key, model download or network needed."""

import itertools

from llama_index.core.node_parser import MarkdownNodeParser

from lab02.common import rk
from lab02.part1_metrics import QUERIES, check_part1, evaluate, retrieval_metrics
from lab02.part4_judge import Relevance, passed


def test_the_labelled_set():
    kinds = [q["kind"] for q in QUERIES]
    assert len(QUERIES) == 27 and kinds.count("unanswerable") == 3
    assert all(q["relevant"] for q in QUERIES if q["kind"] != "unanswerable")
    assert all(not q["relevant"] for q in QUERIES if q["kind"] == "unanswerable")


def test_metrics_on_hand_worked_cases():
    check_part1()  # raises AssertionError on a wrong metric


def test_metrics_match_the_rag_kit_reference():
    ids = ["A", "B", "A", "C", "D"]
    for relevant in ({"A"}, {"B", "C"}, {"E"}, {"A", "D", "E"}):
        for k in (1, 2, 4, 5):
            assert retrieval_metrics(ids, relevant, k) == rk.retrieval_metrics(ids, relevant, k)


def test_an_oracle_retriever_scores_one():
    by_query = {q["query"]: q["relevant"] for q in QUERIES}
    perfect = evaluate(lambda query: list(itertools.islice(itertools.cycle(by_query[query]), 4)))
    assert perfect == {"precision": 1.0, "recall": 1.0, "hit_rate": 1.0, "mrr": 1.0}


def test_bm25_finds_the_german_fault_code():
    """Keyword search needs no embedding model, so the guide's BM25 row is reproducible offline."""
    nodes = MarkdownNodeParser().get_nodes_from_documents(rk.to_llama_documents(rk.load_corpus()))
    assert len(nodes) == 89
    bm25 = rk.bm25_retriever(nodes, 4)
    q10 = next(q for q in QUERIES if q["qid"] == "q10")
    assert set(q10["relevant"]) & set(rk.doc_ids(bm25.retrieve(q10["query"])))
    assert evaluate(lambda query: rk.doc_ids(bm25.retrieve(query))) == {"precision": 0.79, "recall": 0.96, "hit_rate": 1.0, "mrr": 0.96}


def test_refusals_are_scored_with_their_own_rule():
    answerable, unanswerable = {"relevant": ["KB-0610"]}, {"relevant": []}
    assert passed(answerable, Relevance(score=4, refused=False, reason="."))
    assert not passed(answerable, Relevance(score=3, refused=False, reason="."))
    assert passed(unanswerable, Relevance(score=2, refused=True, reason="."))
    assert not passed(unanswerable, Relevance(score=5, refused=False, reason="."))
