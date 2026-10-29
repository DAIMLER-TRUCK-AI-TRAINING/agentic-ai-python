# Demo 02 — In-Memory Semantic Search

A complete (tiny) semantic-search engine with **no vector database** — just a list of `(text, vector)` pairs and brute-force cosine ranking. It makes the two-phase pipeline from the guide concrete: **index once, then answer queries by top-k similarity**.

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §5 (Semantic Search Architecture) and §6 (Information Retrieval).

---

## Features

* **Indexing phase** — embed a small knowledge base once
* **Querying phase** — embed the query with the *same* model, rank by cosine
* **top-k retrieval** — return the most relevant passages
* **Interactive** — type your own questions in a REPL loop
* **Zero dependencies beyond the SDK** — no DB, no NumPy

---

## Setup

```bash
uv sync
```

Rename `.envbackup` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
```

---

## Usage

```bash
uv run main.py
```

The demo first runs two seeded queries, then drops into an interactive prompt. Type a question and press Enter; press Enter on an empty line to quit.

---

## Example Output

```text
Indexing 8 documents with text-embedding-3-small...
Index ready. Ask a question (or press Enter to quit).

Query: 'how long is the battery?'
  [0.58]  The wireless headphones offer 20 hours of battery life on a full charge.
  [0.21]  The laptop comes with a 2-year manufacturer warranty covering hardware defects.
  [0.16]  Premium members get free shipping and early access to seasonal sales.

> what about shipping speed
  [0.55]  Standard shipping takes 3-5 business days; express shipping is next-day.
  ...
```

---

## Key Concepts

| Step | Concept | Description |
| ---- | ------- | ----------- |
| 1 | Indexing | Embed documents once, store vectors in memory |
| 2 | Same-model rule | Query embedded with the identical model |
| 3 | Brute-force search | Compare query to every vector (fine at this scale) |
| 4 | top-k | Return the k closest passages |

---

## Where This Goes Next

This is brute-force `O(n)` search — perfect for a few thousand vectors. At millions, you need **Approximate Nearest Neighbor** search inside a vector database — see **Demo 03 (ChromaDB)**.
