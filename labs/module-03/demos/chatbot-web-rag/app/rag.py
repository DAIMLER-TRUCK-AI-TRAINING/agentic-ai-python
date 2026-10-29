"""
RAG layer — the capability this Module 6 demo adds on top of the Module 5 chatbot.

Pipeline (Guide 06 §4-§6):
    load docs -> chunk -> embed -> store in Chroma -> retrieve top-k -> ground

The knowledge base lives in ./data/*.md. On first use we chunk every document,
embed the chunks with OpenAI (so documents and queries share one vector space),
and index them in a persistent Chroma collection. Each chat turn for a
RAG-enabled persona embeds the user's question, pulls the top-k most similar
chunks, and hands them to the model as grounding context.

The build is lazy: nothing touches OpenAI or Chroma until the first retrieval,
so the app imports cleanly (and the smoke tests run) without an API key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import chromadb
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_store"
COLLECTION = "knowledge_base"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

_client: OpenAI | None = None
_collection = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Embed with OpenAI so documents and queries live in one vector space."""
    resp = _openai().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def chunk_markdown(text: str, source: str) -> list[dict]:
    """
    Split a markdown doc into retrievable chunks, one per heading section.

    Each chunk keeps its heading so a retrieved passage carries enough context
    to stand alone. This is deliberately simple — production pipelines use
    token-aware sliding windows with overlap.
    """
    chunks: list[dict] = []
    sections = re.split(r"\n(?=#{1,6}\s)", text.strip())
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading = re.match(r"#{1,6}\s+(.*)", section)
        title = heading.group(1).strip() if heading else source
        chunks.append({"text": section, "source": source, "title": title})
    return chunks


def _load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def get_collection():
    """Build the Chroma collection once (lazily) and cache it for the process."""
    global _collection
    if _collection is not None:
        return _collection

    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in {c.name for c in db.list_collections()}:
        col = db.get_collection(COLLECTION)
        if col.count() > 0:  # already indexed on a previous run
            _collection = col
            return col
        db.delete_collection(COLLECTION)

    col = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    chunks = _load_chunks()
    if chunks:
        col.add(
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=embed([c["text"] for c in chunks]),
            metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
        )
    _collection = col
    return col


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Embed the question and return the top-k most similar chunks with scores."""
    col = get_collection()
    if col.count() == 0:
        return []
    q_vec = embed([question])[0]
    res = col.query(
        query_embeddings=[q_vec],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "title": meta.get("title", ""),
            "score": round(1 - dist, 3),  # cosine distance -> similarity
        })
    return hits


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks as a numbered grounding block for the prompt."""
    return "\n\n".join(
        f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )
