"""
Demo 01: Embeddings Basics
Turn text into vectors with the OpenAI Embeddings API, then rank a small set
of sentences by how semantically close they are to a query — using cosine
similarity. This is the foundation of every semantic-search system.
"""

import os
import logging
import math
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 1. Environment setup
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

# 2. Client + model. text-embedding-3-small is cheap, fast, and 1536-dim.
client = OpenAI()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# A tiny corpus to rank against the query.
CORPUS = [
    "The cat slept on the warm windowsill all afternoon.",
    "A feline curled up in the sunshine by the glass.",
    "Quarterly revenue grew 12% driven by cloud services.",
    "Our finance team reported strong earnings this quarter.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
]

QUERY = "a sleepy kitten in the sun"


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input string (order preserved)."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: the angle between two vectors, in [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def main() -> None:
    logger.info(f"Embedding model: {EMBED_MODEL}")
    logger.info(f"Query: {QUERY!r}\n")

    # 3. Embed the query and the corpus (one API call for the batch).
    query_vec = embed([QUERY])[0]
    corpus_vecs = embed(CORPUS)
    logger.info(f"Each vector has {len(query_vec)} dimensions.\n")

    # 4. Score every sentence against the query, then rank high → low.
    scored = sorted(
        ((cosine(query_vec, vec), text) for vec, text in zip(corpus_vecs, CORPUS)),
        reverse=True,
    )

    logger.info("Ranked by semantic similarity to the query:\n")
    for rank, (score, text) in enumerate(scored, start=1):
        logger.info(f"  {rank}. [{score:.3f}]  {text}")

    logger.info(
        "\nNotice: the paraphrase ('A feline curled up...') scores high even though "
        "it shares almost no words with the query. That is semantic — not keyword — matching."
    )


if __name__ == "__main__":
    main()
