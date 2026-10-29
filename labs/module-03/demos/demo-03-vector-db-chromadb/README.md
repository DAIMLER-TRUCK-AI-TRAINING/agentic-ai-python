# Demo 03 — Vector Database + RAG with ChromaDB

Graduates from Demo 02's in-memory list to a **real, persistent vector database** ([Chroma](https://www.trychroma.com/)), then closes the loop into **Retrieval-Augmented Generation**: retrieve the most relevant chunks, then let the LLM answer *only* from them.

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §4 (Vector Databases) and §6.4 (From Retrieval to RAG).

---

## Features

* **Persistent store** — Chroma's `PersistentClient` saves vectors to disk
* **Explicit OpenAI embeddings** — documents and queries share one vector space
* **top-k retrieval** — Chroma handles (approximate) nearest-neighbor search
* **Grounded generation** — the LLM answers strictly from retrieved context
* **Honest "don't know"** — a question outside the knowledge base is declined

---

## Setup

```bash
uv sync
```

Rename `.envbackup` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
EMBED_MODEL=text-embedding-3-small   # optional
CHAT_MODEL=gpt-4o-mini               # optional
```

---

## Usage

```bash
uv run main.py
```

The script builds a fresh collection each run (so it's reproducible) and stores it under `./chroma_store/` (git-ignored).

---

## Example Output

```text
Embedding + indexing 6 docs into Chroma...
Collection ready.

Q: Can I swim with the watch?
  retrieved:
    • The TimeFlow X1 is water resistant to 50 meters and safe for swimming.
    • The TimeFlow X1 has a battery life of 7 days on a single charge.
  A: Yes — the TimeFlow X1 is water resistant to 50 meters and safe for swimming.

Q: What is the price?
  retrieved:
    • ...
  A: I don't know — the price isn't in the provided information.
```

---

## Key Concepts

| Step | Concept | Description |
| ---- | ------- | ----------- |
| 1 | Persistence | Vectors survive restarts (`PersistentClient`) |
| 2 | Indexing | `collection.add(ids, documents, embeddings)` |
| 3 | ANN query | `collection.query(query_embeddings, n_results)` |
| 4 | RAG | Inject retrieved chunks as grounding context |
| 5 | Faithfulness | System prompt forbids answering beyond context |

---

## Try This

* Add a `price` fact to `KNOWLEDGE_BASE` and re-ask — watch the "don't know" turn into a real answer.
* Bump `top_k` and observe more (sometimes noisier) context.
* Swap Chroma for `pgvector` or Qdrant — the pipeline shape is identical.
