# Lab 01 demo — Building a RAG pipeline with LangChain and LlamaIndex

A uv project that runs [Lab 01](../../lab-01-rag-pipeline-langchain-llamaindex.md) from the
terminal. It holds the same code as the solution notebook, one module per part, with the notebook's
check cells printed as ✅ / ❌ and the deterministic contracts as offline tests. The provider factory
(`labs/setup/llm.py`) and the shared Module 3 code (`labs/module-03/rag_kit.py`) are imported in
place, exactly as the notebooks import them. The notebooks stay the main lab path.

## Run it

```bash
# (terminal) - from labs/module-03/uv-projects/lab-01-rag-pipeline-langchain-llamaindex
uv sync                          # creates .venv/ with the pinned packages
uv run lab01 status
uv run lab01 faiss               # one part, no keys needed
uv run lab01 all                 # every part, in order
uv run lab01 all --provider anthropic
uv run pytest                    # the offline tests
```

The provider comes from `LLM_PROVIDER` (`openai`, `anthropic`, `azure-openai` or
`azure-anthropic`); the comparison steps also run its partner (OpenAI ↔ Claude, or Azure OpenAI ↔
Claude on Foundry) when it is configured. `--provider` runs on that one only, with no comparison.

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `corpus` | Part 1: the 20 documents and their metadata, then the model alone on the LH-40 question | one call | the table prints; the call is skipped |
| `faiss` | Part 2: 600-character chunks, local embeddings, FAISS, the top 4 for the question | no | runs |
| `chain` | Part 3: the LCEL chain, citations checked in code, three more questions, the partner provider | yes | skipped with a note |
| `llamaindex` | Part 4: 160-token nodes on Chroma and a query engine with a citing prompt | yes | the index is built; the query is skipped |
| `compare` | Part 5: the documents each pipeline retrieves for the same four questions | no | runs |

Later steps reuse what earlier steps of the same run built (the chunks, the FAISS index, the Chroma
collection). Run a later step on its own and it builds what it needs first.

## What it needs

- **uv** 0.9 or later, and the repo-root `.env` (`cp labs/setup/.env.example .env`) with the
  `LLM_PROVIDER` you want and its key and model.
- **The local embedding model, cached:** `uv run python ../../../setup/prefetch_models.py --module 3`.
  With `EMBEDDING_PROVIDER=openai` or `azure-openai` the embeddings come from that API instead.

The FAISS index and the Chroma collection go to `labs/module-03/outputs/uv-projects/lab01/` (git-ignored),
not to the notebooks' `faiss_index/` and `chroma_db/`, so a terminal run never touches a store a
notebook kernel has open.

## Expected output (deterministic)

`uv run lab01 faiss` with the local embedding model:

```
20 documents -> 55 chunks · median 472 characters · max 597
FAISS: 55 vectors of dimension 384 · saved to outputs/uv-projects/lab01/faiss_index/
0.61  TSB-2026-031  '# AdBlue dosing valve crystallisation on LH-40 tractors with DTE-13 en'
0.55  KB-0610       'LH-40 tractors with the DTE-13 engine built in weeks 01 to 13 of 2026 '
0.51  TSB-2026-031  '## Repair\n\n1. Read the fault memory with DiagLink and confirm FC-ADB-0'
0.49  TSB-2026-039  '# Druckluftkompressor mit erhöhter Einschaltdauer bei CT-33 Baustellen'
✅ Part 2 — chunked, embedded and indexed in FAISS
```

The answers in `chain` and `llamaindex` come from a model and vary; the lab guide shows
representative runs.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab01` command
src/lab01/common.py       .env, the provider choice, llm.py and rag_kit.py in place, checks, print helpers (identical in all four projects)
src/lab01/part1_…part5_…  one module per lab part, each with a run() function
src/lab01/cli.py          the `lab01` command
tests/test_lab01.py       offline tests of the deterministic logic
```
