"""Part 2 — naive RAG with LangChain: chunk by characters, embed locally, index in FAISS."""

from __future__ import annotations

import functools
import json

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .common import LAB_DIR, check, llm, rk, step
from .part1_corpus import QUESTION, records


@functools.cache
def chunks() -> list[Document]:
    """LangChain's splitter counts CHARACTERS; add_start_index gives every chunk a stable offset."""
    docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in records()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80, add_start_index=True)
    return splitter.split_documents(docs)


@functools.cache
def embeddings():
    """EMBEDDING_PROVIDER: local Sentence Transformers by default."""
    return llm.get_embeddings()


@functools.cache
def faiss_index() -> faiss.Index:
    """Embed and index every chunk. Unit-length vectors, so inner product = cosine similarity.

    FAISS stores vectors only: the chunks and their metadata are saved beside the index."""
    vectors = np.array(embeddings().embed_documents([c.page_content for c in chunks()]), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; fine up to ~1M vectors
    index.add(vectors)
    rk.FAISS_DIR.mkdir(parents=True, exist_ok=True)  # git-ignored
    faiss.write_index(index, str(rk.FAISS_DIR / "lab01.faiss"))
    (rk.FAISS_DIR / "lab01-chunks.json").write_text(
        json.dumps([{"text": c.page_content, "metadata": c.metadata} for c in chunks()]), encoding="utf-8"
    )
    return index


def retrieve(question: str, k: int = 4) -> list[Document]:
    """Top-k chunks for a question: embed it, search FAISS, map row numbers back to chunks."""
    query = np.array([embeddings().embed_query(question)], dtype="float32")
    scores, rows = faiss_index().search(query, k)
    return [Document(page_content=chunks()[i].page_content, metadata={**chunks()[i].metadata, "score": float(s)})
            for s, i in zip(scores[0], rows[0])]


def run() -> None:
    step("Chunk: 600 characters with 80 overlap")
    sizes = [len(c.page_content) for c in chunks()]
    print(f"{len(records())} documents -> {len(chunks())} chunks · median {int(np.median(sizes))} characters · max {max(sizes)}")

    step("Embed and index in FAISS, then retrieve for the Part 1 question")
    index = faiss_index()
    print(f"FAISS: {index.ntotal} vectors of dimension {index.d} · saved to {rk.FAISS_DIR.relative_to(LAB_DIR)}/")
    for hit in retrieve(QUESTION):
        print(f"{hit.metadata['score']:.2f}  {hit.metadata['doc_id']:13} {hit.page_content[:70]!r}")

    def check_part2():
        assert index.ntotal == len(chunks())
        top = [h.metadata["doc_id"] for h in retrieve(QUESTION)]
        assert {"TSB-2026-031", "KB-0610"} & set(top), top

    check("Part 2 — chunked, embedded and indexed in FAISS", check_part2)
