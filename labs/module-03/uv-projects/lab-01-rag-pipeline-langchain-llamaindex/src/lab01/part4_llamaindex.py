"""Part 4 — the same pipeline in LlamaIndex: the same records and embeddings, chunked by tokens, on Chroma."""

from __future__ import annotations

import functools

from llama_index.core import PromptTemplate, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from .common import check, llm, main_provider, needs_model, rk, setup_llama, step
from .part1_corpus import QUESTION, records

# The default LlamaIndex prompt does not ask for citations, so pass one.
QA_PROMPT = PromptTemplate(
    "You are the Daimler Truck after-sales assistant. Answer only from the sources and cite the doc_id in "
    "square brackets after each claim, for example [KB-0610]. If the sources do not answer, say so.\n\n"
    "<sources>\n{context_str}\n</sources>\n\nQuestion: {query_str}\nAnswer: "
)


@functools.cache
def llama_pipeline():
    """(nodes, Chroma store, index): SentenceSplitter counts TOKENS, not characters."""
    setup_llama()  # Settings.llm and Settings.embed_model from .env
    li_docs = rk.to_llama_documents(records())  # id_ = doc_id; only the title is embedded with each chunk
    nodes = SentenceSplitter(chunk_size=160, chunk_overlap=20).get_nodes_from_documents(li_docs)
    store = rk.chroma_vector_store(f"lab01-{llm.embedding_slug()}")  # outputs/uv-projects/chroma_db/
    index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store))
    return nodes, store, index


def run() -> None:
    step("Same records, same embedding model, 160-token chunks in a Chroma collection")
    nodes, store, index = llama_pipeline()
    print(f"{len(records())} documents -> {len(nodes)} nodes · Chroma collection holds {store._collection.count()} vectors")
    if not needs_model("the LlamaIndex query engine"):
        return

    step(f"A query engine = retriever + prompt + LLM ({main_provider()})")
    response = index.as_query_engine(similarity_top_k=4, text_qa_template=QA_PROMPT).query(QUESTION)
    li_retrieved = {n.node.metadata["doc_id"] for n in response.source_nodes}
    print(response, "\n")
    print("sources:", [(n.node.metadata["doc_id"], round(n.score, 2)) for n in response.source_nodes])

    def check_part4():
        assert store._collection.count() == len(nodes)
        cited = rk.cited_ids(str(response))
        assert cited and cited <= li_retrieved, (cited, li_retrieved)

    check("Part 4 — LlamaIndex pipeline on Chroma", check_part4)
