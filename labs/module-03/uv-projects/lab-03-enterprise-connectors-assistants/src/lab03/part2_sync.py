"""Part 2 — drop retired versions at ingest, then an ingestion pipeline with a docstore for edits and deletes."""

from __future__ import annotations

import functools

from llama_index.core import Settings
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore

from .common import check, llm, rk, setup_llama, step
from .part1_connectors import confluence, sharepoint


@functools.cache
def records() -> list[dict]:
    """The unified knowledge base: corpus + SharePoint + Confluence, published versions only."""
    everything = rk.load_corpus() + sharepoint() + confluence()
    return [r for r in everything if r["metadata"].get("status", "published") == "published"]


def sync_runs(docs: list, vector_store, embed_model) -> dict[str, int]:
    """Nodes written by each run: first, unchanged, one document edited, one document deleted."""
    pipeline = IngestionPipeline(
        transformations=[MarkdownNodeParser(), embed_model],  # chunk, then embed
        vector_store=vector_store, docstore=SimpleDocumentStore(), docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    runs = {"first run": len(pipeline.run(documents=docs)), "unchanged rerun": len(pipeline.run(documents=docs))}
    docs[0].set_content(docs[0].text + "\n\nUpdate 2026-10: also inspect the dosing line for kinks.")
    runs["one document edited"] = len(pipeline.run(documents=docs))
    runs["one document deleted"] = len(pipeline.run(documents=docs[1:]))
    return runs


def run() -> None:
    step("Drop retired versions at ingest")
    retired = [r["metadata"]["doc_id"] + "@" + r["metadata"]["version"] for r in sharepoint() if r["metadata"]["status"] != "published"]
    print(f"{len(records())} published documents · excluded as retired: {retired}")

    step("An ingestion pipeline with a docstore: id_ = doc_id is the key for upserts and deletes")
    setup_llama()
    docs = rk.to_llama_documents(records())
    store = rk.chroma_vector_store(f"lab03-sync-{llm.embedding_slug()}")
    runs = sync_runs(docs, store, Settings.embed_model)
    print({**runs, "vectors in the store now": store._collection.count()})

    def check_part2():
        assert retired == ["POL-WAR-001@2025"], retired
        assert runs["unchanged rerun"] == 0, "an unchanged document must not be re-embedded"
        assert 0 < runs["one document edited"] < runs["first run"], "only the edited document is re-embedded"
        assert store._collection.count() == runs["first run"] - runs["one document edited"], "the deleted document's vectors must go"

    check("Part 2 — retired versions dropped, edits and deletes synced", check_part2)
