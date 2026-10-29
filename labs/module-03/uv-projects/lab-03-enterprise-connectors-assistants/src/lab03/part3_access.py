"""Part 3 — index with access metadata on Pinecone or Chroma, and prove that every user sees only their documents."""

from __future__ import annotations

import functools
import os

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

from .common import check, llm, rk, setup_llama, step
from .part2_sync import records

QUESTION = "How long is the warranty on driveline components, and who approves goodwill above the dealer limit?"
USERS = (rk.EU_DEALER, rk.US_WARRANTY, rk.FLEET_CUSTOMER)


def access_filters(user: rk.User, doc_types: list[str] | None = None) -> MetadataFilters:
    """AND of: region IN (user.region, "ALL"); audience IN user.audiences;
    OR over the user's groups plus "public" of acl_<group> == 1; and doc_type IN doc_types when given."""
    groups = [MetadataFilter(key=f"acl_{g}", value=1) for g in (*user.groups, "public")]
    filters = [
        MetadataFilter(key="region", value=[user.region, "ALL"], operator=FilterOperator.IN),
        MetadataFilter(key="audience", value=list(user.audiences), operator=FilterOperator.IN),
        MetadataFilters(filters=groups, condition=FilterCondition.OR),
    ]
    if doc_types:
        filters.append(MetadataFilter(key="doc_type", value=list(doc_types), operator=FilterOperator.IN))
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


@functools.cache
def knowledge_base() -> tuple:
    """(index, nodes, backend): Markdown-section nodes on Pinecone when PINECONE_API_KEY is set, else Chroma."""
    setup_llama()
    backend = rk.vector_backend()
    nodes = MarkdownNodeParser().get_nodes_from_documents(rk.to_llama_documents(records()))
    with rk.Timer() as t:
        index = rk.build_index(nodes, f"m03-{llm.embedding_slug()}", backend=backend, namespace="knowledge")
    print(f"{len(nodes)} nodes indexed on {backend} in {t.seconds}s"
          + ("" if backend == "pinecone" else " (no PINECONE_API_KEY: the offline fallback)"))
    if backend == "pinecone" and os.getenv("PINECONE_REGION", "us-east-1").startswith("us-"):
        print("⚠️  Residency: this index is in a US region but holds EU documents. POL-DAT-004 §3 requires an EU region.")
    return index, nodes, backend


def run() -> None:
    step("Build the knowledge base on the chosen backend")
    index, _, _ = knowledge_base()

    step("One question, three users: who sees which documents?")
    trimmed = {}
    for user in USERS:
        hits = index.as_retriever(similarity_top_k=6, filters=access_filters(user)).retrieve(QUESTION)
        trimmed[user.name] = hits
        print(f"{user.name:22} {sorted(set(rk.doc_ids(hits)))}")

    def check_part3():
        for user in USERS:
            for hit in trimmed[user.name]:
                assert rk.visible_to(user, hit.node.metadata), (user.name, hit.node.metadata["doc_id"])
        dealer, analyst = rk.doc_ids(trimmed["eu_dealer_advisor"]), rk.doc_ids(trimmed["us_warranty_analyst"])
        assert "POL-WAR-001" in dealer and "POL-GDW-003" not in dealer
        assert "POL-WAR-002" in analyst and "POL-WAR-001" not in analyst

    check("Part 3 — access trimming", check_part3)
