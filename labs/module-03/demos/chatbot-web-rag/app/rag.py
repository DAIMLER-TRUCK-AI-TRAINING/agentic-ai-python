"""
RAG layer — the capability this Module 3 demo adds on top of the Module 2 chatbot.

Pipeline (Module 3 guide §2.1):
    load docs -> chunk -> embed -> store in Chroma -> retrieve top-k -> ground

The knowledge base lives in ./data: the course's synthetic service bulletins,
workshop fault-code guides, fleet-customer FAQs and IT runbooks (a copy of
labs/module-03/data/corpus). On first use we chunk every document, embed the
chunks with the LangChain embedding model from app/llm.py (so documents and
queries share one vector space), and index them in a persistent Chroma
collection. Each chat turn for a RAG-enabled persona embeds the user's question,
pulls the top-k most similar chunks, and hands them to the model as grounding
context.

The collection is named after the embedding model, and it remembers a
fingerprint of ./data: switching EMBEDDING_PROVIDER or editing a document builds
a fresh index on the next retrieval instead of mixing vectors or serving stale text.

The build is lazy: nothing touches a model or Chroma until the first retrieval,
so the app imports cleanly (and the smoke tests run) without an API key.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from threading import Lock

import chromadb

from .llm import embedding_slug, get_embeddings

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_store"
COLLECTION_PREFIX = "kb"

_lock = Lock()
_collection = None


def split_front_matter(text: str) -> tuple[dict, str]:
    """Split a leading '---' block of 'key: value' lines from a markdown document."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not match:
        return {}, text
    meta: dict = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def chunk_markdown(text: str, source: str) -> list[dict]:
    """
    Split a markdown doc into retrievable chunks, one per heading section.

    The front matter is metadata, not content, so it is dropped. A section such
    as "## After the repair" does not say which fault code it belongs to, so
    every chunk below the title starts with the document title, and a
    heading with no text of its own gets no chunk. This is
    deliberately simple — production pipelines use token-aware sliding windows
    with overlap.
    """
    meta, body = split_front_matter(text)
    h1 = re.search(r"^#\s+(.+)$", body, re.M)
    doc_title = meta.get("title") or (h1.group(1).strip() if h1 else source)

    chunks: list[dict] = []
    sections = re.split(r"\n(?=#{1,6}\s)", body.strip())
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading = re.match(r"#{1,6}\s+(.*)", section)
        if heading and not section[heading.end():].strip():
            continue  # a heading with no text: the chunks below it carry the title
        title = heading.group(1).strip() if heading else source
        if not section.startswith("# "):
            section = f"{doc_title}\n\n{section}"
        chunks.append({"text": section, "source": source, "title": title})
    return chunks


def _doc_paths() -> list[Path]:
    return sorted(DATA_DIR.rglob("*.md"))


def _load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in _doc_paths():
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def _fingerprint(paths: list[Path]) -> str:
    """Changes whenever a document in data/ is added, removed or edited."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(DATA_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def collection_name() -> str:
    return f"{COLLECTION_PREFIX}-{embedding_slug()}"


def get_collection():
    """Build the Chroma collection once (lazily) and cache it for the process."""
    global _collection
    # Two first questions at once would otherwise both build the index
    with _lock:
        if _collection is None:
            _collection = _open_or_build()
        return _collection


def _open_or_build():
    name = collection_name()
    fingerprint = _fingerprint(_doc_paths())
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if name in {c.name for c in db.list_collections()}:
        col = db.get_collection(name)
        if (col.metadata or {}).get("fingerprint") == fingerprint and col.count() > 0:
            return col  # indexed on a previous run, from the same documents
        db.delete_collection(name)

    col = db.create_collection(
        name,
        metadata={"fingerprint": fingerprint},
        configuration={"hnsw": {"space": "cosine"}},
    )
    chunks = _load_chunks()
    if chunks:
        print(f"[rag] indexing {len(_doc_paths())} documents → {len(chunks)} chunks into '{name}'")
        col.add(
            ids=[f"chunk-{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=get_embeddings().embed_documents([c["text"] for c in chunks]),
            metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
        )
    return col


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Embed the question and return the top-k most similar chunks with scores."""
    col = get_collection()
    if col.count() == 0:
        return []
    q_vec = get_embeddings().embed_query(question)
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
