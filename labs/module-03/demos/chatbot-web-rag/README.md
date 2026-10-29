# RAG Chatbot

The **Module 5 streaming chatbot, enhanced with Retrieval-Augmented Generation.**
It takes everything from `module-05/chatbot-web` — personas, memory strategies,
token streaming, guardrails, session telemetry — and adds a retrieval layer so a
persona can answer from _your documents_ instead of the model's parametric
memory.

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §4 (Vector Databases)
> and §6.4 (From Retrieval to RAG). This is the capstone that wires Demos 01–03
> (embeddings → semantic search → Chroma) into a working chat application.

---

## What's new vs. Module 5

| Module 5 chatbot                | This demo adds                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Persona system prompts          | A RAG-grounded persona, **Sage**, that answers only from the knowledge base                                                         |
| Sliding / full / summary memory | A **retrieval step** before each RAG turn: embed the question → top-k chunks from Chroma → inject as grounding context              |
| Token streaming + telemetry     | A **Retrieved context panel** showing exactly which chunks (and similarity scores) grounded the answer, plus inline `[n]` citations |
| Input/output guardrails         | Same guardrails, unchanged                                                                                                          |

Non-RAG personas (Nova, Aurora, Bean, Atlas) still work exactly as before — flip
between them to see grounded vs. ungrounded behavior side by side.

## How the RAG layer works

```
data/*.md ──chunk──▶ embed (OpenAI) ──▶ Chroma collection (persistent)
                                              │
user question ──embed──▶ top-k nearest ───────┘──▶ inject as context ──▶ LLM answers
```

- `app/rag.py` — loads `data/*.md`, chunks by markdown heading, embeds with
  OpenAI, and indexes into a persistent Chroma collection. The build is **lazy**:
  nothing hits OpenAI or Chroma until the first retrieval, so the app imports
  (and the tests run) without a key.
- `app/main.py` — for a RAG persona, each turn retrieves the top-k chunks,
  streams a `sources` event to the UI, then inserts the chunks as a system
  message instructing the model to answer **only** from that context.

The shipped knowledge base is a fictional **TimeFlow X1 smartwatch** (specs, app,
warranty, accessories) — the same product line as Demo 03, so the continuity is
intentional.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- OpenAI API key with access to `gpt-4o-mini` and `text-embedding-3-small`

## Setup

```bash
# from labs/module-06/chatbot-web-rag/

uv sync

cp .env.example .env
# Edit .env — set OPENAI_API_KEY
```

> Never commit `.env`. It is already in `.gitignore`.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>. The first message to **Sage** builds the vector
store (chunk + embed + index); it's cached under `./chroma_store/` for later runs.

## Try it

1. Keep the default persona **Sage** (note the green **RAG** pill).
2. Ask a grounded question — _"Can I swim with the watch?"_ The **Retrieved
   context** panel fills with the chunks that were pulled, each with a similarity
   score, and the reply cites them as `[n]`.
3. Ask something outside the docs — _"What's the price?"_ Retrieval still runs,
   but the model honestly says it doesn't know.
4. Switch to **Nova** (no RAG pill) and ask anything — no retrieval happens; the
   answer comes straight from the model. Switch back to feel the difference.
5. Edit a file in `data/`, delete `chroma_store/`, and restart — the new content
   is indexed and instantly retrievable.

## Project Layout

```
chatbot-web-rag/
├── .env.example                  # API key + model template
├── pyproject.toml                # Project metadata & deps (adds chromadb)
├── data/                         # Knowledge base — markdown docs to index
│   ├── timeflow-x1-specs.md
│   ├── timeflow-app.md
│   ├── warranty-and-returns.md
│   └── accessories.md
├── app/
│   ├── main.py                   # FastAPI app + streaming + RAG injection
│   ├── rag.py                    # NEW — chunk, embed, Chroma store, retrieve
│   ├── personas.py               # Adds RAG-grounded "Sage"
│   ├── memory.py                 # Sliding/summary/full strategies (unchanged)
│   ├── sessions.py               # In-memory session store (unchanged)
│   └── guardrails.py             # Input + output filters (unchanged)
├── tests/
│   └── test_smoke.py             # Wiring + chunking tests (no key needed)
└── static/
    └── index.html                # UI with a Retrieved-context panel
```

## API Reference

The `chat` endpoint streams Server-Sent Events. RAG adds one new event type:

| event       | fields                                                         |
| ----------- | -------------------------------------------------------------- |
| `sources`   | hits: `[{n, source, title, score, preview}]` — **new for RAG** |
| `meta`      | persona, strategy, rag, messages_in_prompt, input_tokens_est   |
| `delta`     | delta (string of new tokens)                                   |
| `guardrail` | stage (`input` or `output`), reason                            |
| `done`      | input_tokens, output_tokens, turn_count, totals                |
| `error`     | error                                                          |

## Test

```bash
uv run pytest
```

The smoke tests cover routing, session lifecycle, persona/RAG metadata, the
input guardrail, and the markdown-chunking logic — all without an API key.

## Troubleshooting

**`AuthenticationError`** — Set `OPENAI_API_KEY` in `.env`.

**Stale answers after editing `data/`** — The collection is cached on disk.
Delete `./chroma_store/` and restart to re-index.

**`ModuleNotFoundError: app`** — Run from inside `chatbot-web-rag/`, not the
repo root.

**`uvicorn: command not found`** — Use `uv run uvicorn ...` (not bare `uvicorn`).

## Beyond This Demo

- **Smarter chunking** — token-aware sliding windows with overlap instead of
  one-chunk-per-heading.
- **Hybrid retrieval** — combine vector search with keyword (BM25) re-ranking.
- **Citations with offsets** — link `[n]` to the exact source span.
- **Per-persona knowledge bases** — give each bot its own collection.
- **Swap the store** — the pipeline shape is identical for `pgvector` or Qdrant.
