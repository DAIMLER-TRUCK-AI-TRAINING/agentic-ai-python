# Module 3 labs as uv projects

The four Module 3 labs as small, standalone [uv](https://docs.astral.sh/uv/) projects: the same
code as the solution notebooks, run from the terminal one lab part at a time. Use them for a live
demo, or if you prefer scripts to Jupyter. The notebooks in `../notebooks/` and `../solutions/` stay
the main lab path, and nothing here changes them.

| Project | Lab | Command | Steps |
| ------- | --- | ------- | ----- |
| [`lab-01-rag-pipeline-langchain-llamaindex/`](lab-01-rag-pipeline-langchain-llamaindex/README.md) | [Lab 01](../lab-01-rag-pipeline-langchain-llamaindex.md) | `uv run lab01` | `corpus` · `faiss` · `chain` · `llamaindex` · `compare` |
| [`lab-02-tuning-evaluating-retrieval/`](lab-02-tuning-evaluating-retrieval/README.md) | [Lab 02](../lab-02-tuning-evaluating-retrieval.md) | `uv run lab02` | `metrics` · `chunking` · `hybrid` · `judge` |
| [`lab-03-enterprise-connectors-assistants/`](lab-03-enterprise-connectors-assistants/README.md) | [Lab 03](../lab-03-enterprise-connectors-assistants.md) | `uv run lab03` | `connectors` · `sync` · `access` · `assistants` · `calibrate` |
| [`lab-04-agentic-rag-fine-tuning-decisions/`](lab-04-agentic-rag-fine-tuning-decisions/README.md) | [Lab 04](../lab-04-agentic-rag-fine-tuning-decisions.md) | `uv run lab04` | `tool` · `agent` · `canvases` · `critique` |

Every command also takes `all` (every step in order), `status` (which `.env` was loaded, which
providers, embeddings and stores are in use, never the key values) and
`--provider openai|anthropic|azure-openai|azure-anthropic`. Without `--provider`, each lab runs on
`LLM_PROVIDER` and its comparison steps also run the partner (OpenAI ↔ Claude, or Azure OpenAI ↔
Claude on Foundry) when it is configured; with `--provider`, on that one only. A command exits with
status 1 if any check failed.

## Quick start

```bash
# (terminal) - from the repo root, once
cp labs/setup/.env.example .env          # then add LLM_PROVIDER and its key and model (and its partner's)

# (terminal) - then, per lab
cd labs/module-03/uv-projects/lab-01-rag-pipeline-langchain-llamaindex
uv sync                                  # .venv/ inside the project, from uv.lock
uv run python ../../../setup/prefetch_models.py --module 3   # once: the local embedding and re-ranking models
uv run lab01 status
uv run lab01 all
uv run pytest                            # offline tests, no keys needed
```

You need uv 0.9 or later. Each project pins Python 3.12 in `.python-version`; uv downloads it if
you do not have it. No system packages are needed. On Linux and Windows the projects install
CPU-only PyTorch from the PyTorch index, which avoids the much larger CUDA wheels; macOS uses PyPI.

## How they relate to the notebooks

| | Notebooks | uv projects |
| --- | --- | --- |
| Code | starter with `TODO`s, plus the solution | the solution, one module per lab part |
| Checks | check cells print ✅ / ⏳ / ⚠️ | each step prints the same checks as ✅ / ❌; offline `pytest` tests cover the deterministic contracts |
| Packages | the course venv, `labs/setup/requirements.txt` | one `.venv/` per project, with the same pins in `pyproject.toml` and a committed `uv.lock` |
| Provider and RAG helpers | `labs/setup/llm.py` and `labs/module-03/rag_kit.py` | the same two files, imported in place by `common.py` |
| Settings and data | repo-root `.env`, `labs/module-03/data/` | the same files, read in place |
| Stores | `chroma_db/` and `faiss_index/` in `labs/module-03/` | `labs/module-03/outputs/uv-projects/lab0N/`, one folder per project, so a run never opens a store that a notebook kernel or another project has open |
| Offline | the starter's TODOs fall back to `rag_kit`; Pinecone falls back to Chroma; connector exports instead of live tenants | the same fallbacks; without a provider key, the model steps are skipped with a note and the local steps (chunking, indexing, retrieval, re-ranking, sync, access trimming) still run |

Each project is self-contained, so you can `cd` into one and run it without the others. They share
nothing but the data and the two in-place files: `src/lab0N/common.py` (the `.env` loading, the
provider choice, the checks and the print helpers) is the same file in all four. Only Lab 01
installs LangChain and FAISS; Lab 03 adds the Pinecone client.

## What a full run costs

Measured with `gpt-5-mini` (`LLM_PROVIDER=openai`) and `claude-haiku-5-5` as its partner,
`EMBEDDING_PROVIDER=local`, `all` on both providers, four projects at once. Embeddings and
re-ranking run locally and cost nothing.

| Command | Model calls (both providers, approximate) | Wall time (measured) |
| ------- | ----------------------------------------- | -------------------- |
| `uv run lab01 all` | ≈ 7 | 1 min |
| `uv run lab02 all` | ≈ 15, 10 of them on the small tier | 1 min 25 s |
| `uv run lab03 all` | ≈ 12 (the score floor skips the model for unanswerable questions) | 1 min 40 s |
| `uv run lab04 all` | ≈ 22, 8 of them on the small tier; the agent's count varies per run | 1 min 40 s |

## Cleaning up

```bash
# (terminal) - from labs/module-03/uv-projects
rm -rf */.venv                           # uv sync recreates them from uv.lock
rm -rf ../outputs/uv-projects            # the projects' Chroma collections and FAISS index
```
