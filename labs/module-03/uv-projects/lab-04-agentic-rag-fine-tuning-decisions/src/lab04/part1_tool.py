"""Part 1 — the retriever as a tool: hybrid search + re-rank + floor inside the caller's access filters, as JSON."""

from __future__ import annotations

import functools
import json

from llama_index.core import QueryBundle

from .common import check, llm, rk, setup_llama, step

DOC_TYPES = ("bulletin", "kb", "faq", "runbook", "policy", "contract", "engineering")
USER = rk.EU_AFTERSALES  # the authenticated caller: an EU after-sales engineer


@functools.cache
def knowledge_base() -> tuple:
    """(index, nodes): corpus + SharePoint + Confluence by Markdown section, on Chroma."""
    setup_llama()
    return rk.build_knowledge_base(f"lab04-{llm.embedding_slug()}")


@functools.cache
def _search_state() -> tuple:
    index, nodes = knowledge_base()
    return index, [n for n in nodes if rk.visible_to(USER, n.metadata)], rk.reranker(4)


def hybrid_search(query: str, doc_type: str = "any", floor: float = -1.0):
    """Up to 4 re-ranked nodes above the floor, for one doc_type or "any" (Labs 02-03)."""
    index, allowed, rerank = _search_state()
    types = None if doc_type == "any" else [doc_type]
    pool = [n for n in allowed if types is None or n.metadata["doc_type"] in types]
    candidates = rk.hybrid_retriever(index, pool, k=10, pool=10, filters=rk.access_filters(USER, types)).retrieve(query)
    return [h for h in rerank.postprocess_nodes(candidates, query_bundle=QueryBundle(query)) if h.score >= floor]


def search_knowledge(query: str, doc_type: str = "any") -> str:
    """Search Daimler Truck workshop bulletins (bulletin), repair guides (kb), customer FAQs (faq),
    IT runbooks (runbook), policies (policy), contracts (contract) and engineering specs (engineering).
    Use it for any question about vehicles, fault codes, repairs, warranty, policies or IT procedures;
    do not use it for greetings or thanks. doc_type narrows the search to one of those types, or "any".
    Returns up to 4 passages as JSON with doc_id, title and text."""
    if doc_type != "any" and doc_type not in DOC_TYPES:
        return f"NO_RESULTS: unknown doc_type {doc_type!r}. Use one of {', '.join(DOC_TYPES)} or 'any'."
    hits = hybrid_search(query, doc_type)
    if not hits:
        return "NO_RESULTS: nothing relevant found. Rephrase the query, or try doc_type='any'."
    return json.dumps([{"doc_id": h.node.metadata["doc_id"], "title": h.node.metadata["title"],
                        "text": h.node.get_content()} for h in hits], ensure_ascii=False)


def check_part1():
    passages = json.loads(search_knowledge("FC-BAT-2201 threshold", doc_type="engineering"))
    assert passages and {p["doc_id"] for p in passages} == {"ENG-SPEC-0417"}, passages
    assert {"doc_id", "title", "text"} <= set(passages[0])
    assert search_knowledge("FC-BAT-2201", doc_type="manual").startswith("NO_RESULTS")
    assert search_knowledge("What is the canteen menu on Friday?").startswith("NO_RESULTS")


def run() -> None:
    step("The knowledge base and the caller")
    _, nodes = knowledge_base()
    print(f"{len(nodes)} nodes · user {USER.name} sees {sum(rk.visible_to(USER, n.metadata) for n in nodes)} of them")

    step("The tool contract: JSON with stable ids, a doc_type filter, and misses that say what to do next")
    for query, doc_type in [("FC-BAT-2201 threshold", "engineering"), ("FC-BAT-2201", "manual"), ("What is the canteen menu on Friday?", "any")]:
        result = search_knowledge(query, doc_type)
        shown = result if result.startswith("NO_RESULTS") else f"{[p['doc_id'] for p in json.loads(result)]}"
        print(f"search_knowledge({query!r}, {doc_type!r}) -> {shown}")
    check("Part 1 — search_knowledge tool", check_part1)
