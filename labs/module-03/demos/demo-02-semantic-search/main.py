"""
Demo 02: In-Memory Semantic Search
A complete (tiny) semantic-search engine with no vector database — just a list
of vectors and brute-force cosine ranking. This makes the search pipeline from
Guide 06 §5 concrete: index once, then answer queries by top-k similarity.

Run interactively: type questions, get the closest passages back.
"""

import os
import logging
import math
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# A small "knowledge base". In a real system these are chunks of your documents.
DOCUMENTS = [
    "Our return policy allows refunds within 30 days of purchase with a receipt.",
    "Standard shipping takes 3-5 business days; express shipping is next-day.",
    "The wireless headphones offer 20 hours of battery life on a full charge.",
    "To reset your password, click 'Forgot password' on the login screen.",
    "Premium members get free shipping and early access to seasonal sales.",
    "The laptop comes with a 2-year manufacturer warranty covering hardware defects.",
    "Customer support is available 24/7 via live chat and email.",
    "Gift cards never expire and can be used across all our store locations.",
]


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def build_index(docs: list[str]) -> list[tuple[str, list[float]]]:
    """INDEXING phase — embed every document once, keep (text, vector) pairs."""
    logger.info(f"Indexing {len(docs)} documents with {EMBED_MODEL}...")
    vectors = embed(docs)
    return list(zip(docs, vectors))


def search(index, query: str, top_k: int = 3) -> list[tuple[float, str]]:
    """QUERYING phase — embed query, score against the index, return top-k."""
    query_vec = embed([query])[0]
    scored = [(cosine(query_vec, vec), text) for text, vec in index]
    scored.sort(reverse=True)
    return scored[:top_k]


def main() -> None:
    index = build_index(DOCUMENTS)
    logger.info("Index ready. Ask a question (or press Enter to quit).\n")

    # A couple of seeded queries so the demo shows results even without typing.
    seeded = ["how long is the battery?", "can I get my money back?"]
    for q in seeded:
        logger.info(f"Query: {q!r}")
        for score, text in search(index, q):
            logger.info(f"  [{score:.3f}]  {text}")
        logger.info("")

    # Interactive loop.
    try:
        while True:
            q = input("> ").strip()
            if not q:
                break
            for score, text in search(index, q):
                logger.info(f"  [{score:.3f}]  {text}")
            logger.info("")
    except (EOFError, KeyboardInterrupt):
        pass
    logger.info("Done.")


if __name__ == "__main__":
    main()
