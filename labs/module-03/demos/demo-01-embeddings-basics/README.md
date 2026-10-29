# Demo 01 — Embeddings Basics

Turn text into vectors with the **OpenAI Embeddings API**, then rank a small set of sentences by semantic similarity to a query using **cosine similarity**. This is the atom every semantic-search and RAG system is built from.

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §2 (Embeddings) and §3 (Similarity).

---

## Features

* **Embeddings API** — `text-embedding-3-small` (1536-dim, cheap, fast)
* **Batch embedding** — query + corpus in a single API call
* **Cosine similarity** — pure-Python, no NumPy needed
* **Semantic ranking** — shows paraphrases beating keyword overlap

---

## Setup

```bash
uv sync
```

Configure environment variables — rename `.envbackup` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
EMBED_MODEL=text-embedding-3-small   # optional
```

---

## Usage

```bash
uv run main.py
```

---

## Example Output

```text
Embedding model: text-embedding-3-small
Query: 'a sleepy kitten in the sun'

Each vector has 1536 dimensions.

Ranked by semantic similarity to the query:

  1. [0.61]  A feline curled up in the sunshine by the glass.
  2. [0.55]  The cat slept on the warm windowsill all afternoon.
  3. [0.18]  Photosynthesis converts sunlight into chemical energy in plants.
  4. [0.09]  Our finance team reported strong earnings this quarter.
  5. [0.07]  Quarterly revenue grew 12% driven by cloud services.
```

(Scores vary slightly run to run, but the *ranking* is stable.)

---

## Key Concepts

| Step | Concept | Description |
| ---- | ------- | ----------- |
| 1 | Embedding | Text → fixed-length vector capturing meaning |
| 2 | Batching | Embed many strings in one request |
| 3 | Cosine similarity | Angle between vectors, in [-1, 1] |
| 4 | Semantic ranking | Sort corpus by closeness to the query |

---

## Try This

* Change `QUERY` to something abstract ("financial results") and watch the ranking flip.
* Swap `EMBED_MODEL` to `text-embedding-3-large` and compare scores.
* Add your own sentences to `CORPUS`.
