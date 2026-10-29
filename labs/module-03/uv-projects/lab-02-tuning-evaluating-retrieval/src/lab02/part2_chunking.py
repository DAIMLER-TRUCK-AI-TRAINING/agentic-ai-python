"""Part 2 — four node parsers on the same documents and embeddings, measured with the Part 1 harness."""

from __future__ import annotations

import functools

import pandas as pd
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser, SentenceSplitter

from .common import llm, rk, setup_llama, show, step
from .part1_metrics import K, evaluate


@functools.cache
def splitters() -> dict:
    """SentenceSplitter sizes are tokens; the semantic splitter cuts where meaning shifts (it embeds sentences).

    Pass the embedding model to the semantic splitter: from_defaults() falls back to OpenAIEmbedding(),
    not Settings.embed_model, which would send the documents to OpenAI and fail without its key."""
    setup_llama()
    return {
        "fixed-128": SentenceSplitter(chunk_size=128, chunk_overlap=16),
        "fixed-512": SentenceSplitter(chunk_size=512, chunk_overlap=50),
        "semantic": SemanticSplitterNodeParser.from_defaults(embed_model=Settings.embed_model, breakpoint_percentile_threshold=90),
        "markdown": MarkdownNodeParser(),  # structure-aware: one node per heading section
    }


@functools.cache
def indexes() -> dict[str, tuple]:
    """strategy -> (VectorStoreIndex, nodes), one Chroma collection per strategy."""
    li_docs = rk.to_llama_documents(rk.load_corpus())
    built = {}
    for name, splitter in splitters().items():
        nodes = splitter.get_nodes_from_documents(li_docs)
        built[name] = (rk.build_index(nodes, f"lab02-{name}-{llm.embedding_slug()}"), nodes)
    return built


def run() -> None:
    step(f"Same documents, same embeddings, dense top-{K}: only the node parser changes")
    rows = []
    for name, (index, nodes) in indexes().items():
        retriever = index.as_retriever(similarity_top_k=K)
        rows.append({"strategy": name, "nodes": len(nodes),
                     "median chars": int(pd.Series([len(n.get_content()) for n in nodes]).median()),
                     **evaluate(lambda q, r=retriever: rk.doc_ids(r.retrieve(q)))})
    show(pd.DataFrame(rows).set_index("strategy"))

    q = "How long does a DC fast charge from 20 to 80 percent take on an ED-26?"
    step(f"Read what each strategy hands the model: {q!r}")
    for name, (index, _) in indexes().items():
        top = index.as_retriever(similarity_top_k=1).retrieve(q)[0]
        print(f"--- {name} · {top.node.metadata['doc_id']} · {len(top.node.get_content())} chars\n{top.node.get_content()[:300]}\n")
    print("📝 Copy the table into templates/chunking-workbook.md and note which strategy you would ship.")
