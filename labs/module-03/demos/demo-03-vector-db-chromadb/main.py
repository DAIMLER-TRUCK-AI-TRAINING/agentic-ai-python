"""
Demo 03: Vector Database + RAG with ChromaDB
Graduates from the in-memory search of Demo 02 to a real, persistent vector
database (Chroma), then closes the loop into Retrieval-Augmented Generation:
retrieve the most relevant chunks and have the LLM answer ONLY from them.

Pipeline (Guide 06 §4-§6):
    chunk -> embed -> store in Chroma -> query -> retrieve top-k -> generate
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Our knowledge base. In production these are chunks of real documents.
KNOWLEDGE_BASE = [
    ("doc1", "The TimeFlow X1 smartwatch has a battery life of 7 days on a single charge."),
    ("doc2", "The TimeFlow X1 is water resistant to 50 meters and safe for swimming."),
    ("doc3", "TimeFlow watches sync with both iOS and Android via the TimeFlow app."),
    ("doc4", "The TimeFlow X1 includes built-in GPS, a heart-rate sensor, and an SpO2 monitor."),
    ("doc5", "Replacement bands for the TimeFlow X1 come in silicone, leather, and metal."),
    ("doc6", "The TimeFlow X1 ships with a 1-year limited warranty against manufacturing defects."),
]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed with OpenAI so documents and queries share one vector space."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def build_collection():
    """Create (or reset) a persistent Chroma collection and load our vectors."""
    db = chromadb.PersistentClient(path="./chroma_store")
    # Start clean each run so the demo is reproducible.
    if "timeflow" in [c.name for c in db.list_collections()]:
        db.delete_collection("timeflow")
    col = db.create_collection("timeflow")

    ids = [doc_id for doc_id, _ in KNOWLEDGE_BASE]
    texts = [text for _, text in KNOWLEDGE_BASE]
    logger.info(f"Embedding + indexing {len(texts)} docs into Chroma...")
    col.add(ids=ids, documents=texts, embeddings=embed(texts))
    return col


def retrieve(col, question: str, top_k: int = 2) -> list[str]:
    """Embed the question and pull the top-k most similar chunks from Chroma."""
    q_vec = embed([question])[0]
    res = col.query(query_embeddings=[q_vec], n_results=top_k)
    return res["documents"][0]


def answer(question: str, context_chunks: list[str]) -> str:
    """RAG generation: answer strictly from the retrieved context."""
    context = "\n".join(f"- {c}" for c in context_chunks)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content":
                "You answer questions about the TimeFlow X1 watch using ONLY the "
                "provided context. If the answer is not in the context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content


def main() -> None:
    col = build_collection()
    logger.info("Collection ready.\n")

    for question in [
        "Can I swim with the watch?",
        "How long does the battery last?",
        "What is the price?",  # not in the knowledge base — should say "don't know"
    ]:
        logger.info(f"Q: {question}")
        chunks = retrieve(col, question)
        logger.info("  retrieved:")
        for c in chunks:
            logger.info(f"    • {c}")
        logger.info(f"  A: {answer(question, chunks)}\n")


if __name__ == "__main__":
    main()
