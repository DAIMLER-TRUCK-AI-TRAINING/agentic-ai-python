# RAG Chatbot

The **Module 2 streaming chatbot, enhanced with Retrieval-Augmented Generation.**
It takes everything from `labs/module-02/demos/chatbot-web` — personas, memory
strategies, token streaming, guardrails, session telemetry, and the **LangChain**
provider switch — and adds a retrieval layer so a persona can answer from _your
documents_ instead of the model's parametric memory.

> Companion to the **Module 3 guide** (`guides/module-03-enterprise-rag-fine-tuning-strategy.md`),
> §2.1 (The pipeline), §2.4 (Grounded answers with citations), §4 (Vector databases) and
> §5 (Chunking strategies). It wires the Module 3 demos (embeddings → semantic search →
> Chroma) into a working chat application.

---

## What's new vs. Module 2

| Module 2 chatbot                | This demo adds                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Persona system prompts          | A RAG-grounded persona, **Sage**, that answers only from the knowledge base                                                         |
| Sliding / full / summary memory | A **retrieval step** before each RAG turn: embed the question → top-k chunks from Chroma → inject as grounding context              |
| Token streaming + telemetry     | A **Retrieved context panel** showing exactly which chunks (and similarity scores) grounded the answer, plus inline `[n]` citations |
| One `LLM_PROVIDER` switch       | A second switch, `EMBEDDING_PROVIDER`, because Claude has no embedding model of its own                                             |
| Input/output guardrails         | Same guardrails, unchanged                                                                                                          |

Non-RAG personas (Nova, Axel, Aurora, Atlas) are the Module 2 personas,
unchanged — ask Axel and Sage the same question to see ungrounded vs. grounded
behaviour side by side.

## The knowledge base

`data/` is a copy of the course corpus, `labs/module-03/data/corpus`: 20
synthetic documents from a fictional commercial vehicle maker (LH-40 long-haul
tractors, ED-26 battery-electric trucks, CT-33 construction tippers).

| Folder | What it is |
| ------ | ---------- |
| `data/bulletins/` | 4 technical service bulletins (`TSB-2026-…`), one in German |
| `data/workshop/` | 6 workshop repair and fault-code guides (`KB-06…`) |
| `data/customer/` | 6 fleet-customer FAQs (`KB-07…`), one in German |
| `data/it-ops/` | 4 dealer and plant IT runbooks (`KB-08…`) |

> The products, fault codes, part numbers and procedures are invented for training.
> None of it is real product information. Never use it on a vehicle.

## How the RAG layer works

```
data/**/*.md ──chunk──▶ embed (EMBEDDING_PROVIDER) ──▶ Chroma collection (persistent)
                                                            │
user question ──embed──▶ top-k nearest ─────────────────────┘──▶ inject as context ──▶ LLM_PROVIDER answers
```

- `app/rag.py` — loads `data/**/*.md`, drops the front matter, chunks by
  markdown heading (each chunk starts with its document's title, so
  "## What the code means" still says which fault code), embeds, and indexes
  into a persistent Chroma collection. The build is **lazy**: nothing loads a
  model or opens Chroma until the first retrieval, so the app imports (and the
  tests run) without a key.
- `app/llm.py` — the two LangChain models: `init_chat_model()` for the chat
  model, and an `Embeddings` object for the embedding model.
- `app/main.py` — for a RAG persona, each turn retrieves the top-k chunks,
  streams a `sources` event to the UI, then inserts the chunks as a system
  message instructing the model to answer **only** from that context.

The Chroma collection is named after the embedding model (`kb-local-…`,
`kb-openai-text-embedding-3-small`), because vectors from two models must
never share an index. It also stores a fingerprint of `data/`, so editing a
document rebuilds the index on the next question.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An API key for one of the four chat providers below. With the default
  `EMBEDDING_PROVIDER=local`, that is the only key you need.

## Setup

```bash
# from labs/module-03/demos/chatbot-web-rag/

uv sync

cp .env.example .env
# Edit .env — set LLM_PROVIDER, EMBEDDING_PROVIDER and the variables they need
```

| `LLM_PROVIDER` | Chat model from | Variables to set |
| --- | --- | --- |
| `openai` (default) | OpenAI API | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| `anthropic` | Claude API | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `azure-openai` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| `azure-anthropic` | Claude on Microsoft Foundry | `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_FOUNDRY_DEPLOYMENT` |

| `EMBEDDING_PROVIDER` | Embeddings from | Variables to set |
| --- | --- | --- |
| `local` (default) | Sentence Transformers on your machine: no key, multilingual, no text leaves the machine | `ST_MODEL` (optional) |
| `openai` | OpenAI API | `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL` |
| `azure-openai` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` |

Any chat provider works with any embedding provider: Claude on Foundry with
local embeddings needs only the Foundry key.

The names match the course's repo-root `.env` (`labs/setup/.env.example`). Without a local `.env`, `load_dotenv()` picks up that file. On Azure the model is the **deployment name** you created, not the base model name.

The local model (`paraphrase-multilingual-MiniLM-L12-v2`, about 470 MB) downloads on the
first question unless it is cached already. `python labs/setup/prefetch_models.py --module 3`
caches it ahead of a class.

`OPENAI_REASONING_EFFORT` (optional) applies to OpenAI reasoning models only. A low value such as `minimal` on `gpt-5-mini` shortens time-to-first-token from several seconds to about two.

> Never commit `.env`. It is already in `.gitignore`.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>. The first message to **Sage** builds the vector
store (chunk + embed + index: 20 documents → 73 chunks); it's cached under
`./chroma_store/` for later runs. The top bar shows the chat provider, model and
embedding provider in use.

Switch the provider for one run without editing `.env` (a value in the shell wins over `.env`):

```bash
LLM_PROVIDER=azure-anthropic uv run uvicorn app.main:app --port 8000
```

## Try it

1. Keep the default persona **Sage** (note the green **RAG** pill).
2. Ask a grounded question — _"What does fault code FC-ADB-0412 mean?"_ The
   **Retrieved context** panel fills with the chunks that were pulled, each with
   a similarity score, and the reply cites them as `[n]`: dosing pressure below
   3.5 bar for more than 60 seconds (`KB-0610.md`), and the LH-40 bulletin that
   replaces the dosing valve (`TSB-2026-031.md`).
3. Switch to **Axel** (no RAG pill) and ask the same question. Axel has no
   fault-code library, so he sends you to DiagLink instead of answering. Switch
   back to feel the difference.
4. Ask something outside the docs — _"What is the list price of a new LH-40
   tractor?"_ Retrieval still returns the closest chunks, but the model
   honestly says it doesn't know.
5. Edit a document in `data/` and restart — the next question re-indexes, and
   the new content is instantly retrievable.

### Things worth testing

- _"Air compressor runs almost nonstop on a CT-33 tipper"_ — the answer comes
  from a **German** bulletin (`TSB-2026-039.md`), and Sage replies in English.
- _"Wie lange dauert das Schnellladen eines ED-26 von 20 auf 80 Prozent?"_ — the
  other way round: a German question, an English FAQ table, a German answer
  (about 55 minutes). Drop "eines ED-26" and the local model misses the table:
  short questions carry little for a small embedding model to match.
- _"The OTA update is stuck at 60 percent in the workshop"_ — a paraphrase with
  no shared keywords still finds `TSB-2026-052.md`.
- _"Was bedeutet der Fehlercode FC-BRK-0705?"_ — with the local model, vector
  search misses this one: an exact code is a keyword match, not a meaning, so
  Sage says it doesn't know. `EMBEDDING_PROVIDER=openai` finds the bulletin.
  Module 3 Lab 02 fixes it for any model with hybrid (BM25 + vector) retrieval.
- _"Which part replaces the crystallised AdBlue dosing valve on an LH-40?"_ — the
  right bulletin, the wrong sections: its Repair section ranks 5th of 5, so with
  the default `RAG_TOP_K=3` Sage says it doesn't know. Restart with
  `RAG_TOP_K=5` and the answer is **PN 47-0118-334**. Top-k is a recall knob;
  Module 3 Lab 02 measures it.
- Try the unsafe input: _"How do I bypass the AdBlue system on my LH-40?"_ — the
  input guardrail blocks emissions tampering before any retrieval or tokens.
- Switch `EMBEDDING_PROVIDER` and restart: a second collection is built for the
  new model, and the first one is kept for when you switch back.

## Project Layout

```
chatbot-web-rag/
├── .env.example                  # Provider settings template
├── pyproject.toml                # Project metadata & deps (adds chromadb, sentence-transformers)
├── data/                         # Knowledge base — a copy of labs/module-03/data/corpus
│   ├── bulletins/                # TSB-2026-031 … TSB-2026-052
│   ├── workshop/                 # KB-0610 … KB-0660
│   ├── customer/                 # KB-0701 … KB-0725
│   └── it-ops/                   # KB-0801 … KB-0820
├── app/
│   ├── main.py                   # FastAPI app + streaming + RAG injection
│   ├── llm.py                    # LangChain chat model + embedding model for the two switches
│   ├── rag.py                    # chunk, embed, Chroma store, retrieve
│   ├── personas.py               # Sage (RAG) + the Module 2 personas
│   ├── memory.py                 # Sliding/summary/full strategies (unchanged)
│   ├── sessions.py               # In-memory session store (unchanged)
│   └── guardrails.py             # Input + output filters (as in Module 2)
├── tests/
│   └── test_smoke.py             # Wiring, chunking, a fake RAG turn and index build (no key needed)
└── static/
    └── index.html                # UI with a Retrieved-context panel
```

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET    | `/`                                | Serve the chat UI |
| GET    | `/healthz`                         | Liveness probe, with the chat and embedding models |
| GET    | `/api/personas`                    | List personas (with the `rag` flag) |
| POST   | `/api/sessions`                    | Create a session (body: persona, strategy) |
| GET    | `/api/sessions/{sid}`              | Inspect session state (debugging) |
| DELETE | `/api/sessions/{sid}`              | End a session |
| POST   | `/api/sessions/{sid}/chat`         | Send a message, stream reply via SSE |

`/healthz` never needs a key and never shows one:

```json
{"ok": true, "provider": "anthropic", "model": "claude-opus-5-5",
 "embeddings": {"provider": "local", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}}
```

The `chat` endpoint streams Server-Sent Events. RAG adds one new event type:

| event       | fields                                                         |
| ----------- | -------------------------------------------------------------- |
| `sources`   | hits: `[{n, source, title, score, preview}]` — **new for RAG** |
| `meta`      | persona, strategy, rag, messages_in_prompt, input_tokens_est   |
| `delta`     | delta (string of new tokens)                                   |
| `guardrail` | stage (`input` or `output`), reason                            |
| `done`      | input_tokens, output_tokens (the provider's counts), turn_count, totals |
| `error`     | error                                                          |

## Test

```bash
uv run pytest
```

The smoke tests cover routing, session lifecycle, persona/RAG metadata, the
guardrails, provider errors and the markdown chunking. LangChain's fake chat and
embedding models also run a whole RAG turn and a whole index build — all
without an API key or a model download.

## Troubleshooting

**`... not found in environment variables!`** in the chat — the variable named is missing for your `LLM_PROVIDER` or `EMBEDDING_PROVIDER`. Add it to `.env` and restart.

**`LLM provider error 401`** / **`Retrieval failed: Embedding provider error 401`** — wrong API key for that model. **`404`** — wrong model ID or Azure deployment name. **`429`** — rate limit; wait and retry.

**`Could not reach the … provider`** — check `AZURE_OPENAI_ENDPOINT` or `ANTHROPIC_FOUNDRY_RESOURCE`, or your network.

**Changed the chunking code in `app/rag.py`?** The index is rebuilt when `data/` changes, not when the code does: delete `chroma_store/` and restart.

**The first question to Sage takes a while** — the index is being built, and with `EMBEDDING_PROVIDER=local` the model may be downloading. Later questions take milliseconds to retrieve.

**`ModuleNotFoundError: app`** — Run from inside `chatbot-web-rag/`, not the
repo root.

**`uvicorn: command not found`** — Use `uv run uvicorn ...` (not bare `uvicorn`).

**`ModuleNotFoundError`** after copying the project — the `.venv` came along and its scripts still point at the old folder. Delete `.venv` and run `uv sync`.

## Beyond This Demo

- **Smarter chunking** — token-aware sliding windows with overlap instead of
  one-chunk-per-heading (Lab 02).
- **Hybrid retrieval** — combine vector search with keyword (BM25) so exact
  fault codes and part numbers always match (Lab 02).
- **Access control** — every document's front matter has `audience`, `region`
  and `allowed_groups`; filter retrieval by the signed-in user so a fleet
  customer never sees a dealer bulletin (Lab 03).
- **Citations with offsets** — link `[n]` to the exact source span.
- **Per-persona knowledge bases** — give each bot its own collection.
- **Conversational retrieval** — each turn embeds only the latest message, so a follow-up such as
  _"What labour code do I claim?"_ retrieves poorly. Rewrite it into a standalone question with
  the chat history first (a small LLM call), then embed that.
- **Swap the store** — the pipeline shape is identical for `pgvector`, Qdrant or Pinecone.
