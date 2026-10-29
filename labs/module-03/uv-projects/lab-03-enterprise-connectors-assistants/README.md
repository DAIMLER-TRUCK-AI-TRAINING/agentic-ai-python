# Lab 03 demo — Enterprise connectors and use-case assistants

A uv project that runs [Lab 03](../../lab-03-enterprise-connectors-assistants.md) from the terminal.
It holds the same code as the solution notebook (the `load_confluence_export`, `access_filters` and
`make_assistant` TODOs filled in), one module per part, with the notebook's check cells printed as
✅ / ❌ and the deterministic contracts as offline tests. `labs/setup/llm.py` and
`labs/module-03/rag_kit.py` are imported in place, as the notebooks import them. The notebooks stay
the main lab path.

## Run it

```bash
# (terminal) - from labs/module-03/uv-projects/lab-03-enterprise-connectors-assistants
uv sync
uv run lab03 status              # also shows whether Pinecone or Chroma is used
uv run lab03 access              # one part, no keys needed
uv run lab03 all                 # every part, in order
uv run lab03 all --provider anthropic
uv run pytest                    # the offline tests
```

The assistants run on `LLM_PROVIDER`; Part 5 also runs them on its partner. `--provider` runs on
that one only.

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `connectors` | Part 1: live readers or the exports; SharePoint columns and permissions; the Confluence loader | no | runs |
| `sync` | Part 2: retired versions dropped; an ingestion pipeline re-embeds only edits and propagates deletes | no | runs |
| `access` | Part 3: the knowledge base on Pinecone or Chroma, and one question for three users | no | runs |
| `assistants` | Part 4: the policy-search and customer-support assistants on 8 questions, 2 of them unanswerable | yes | skipped with a note |
| `calibrate` | Part 5: the best re-ranked score per question against the floor, then the partner provider | partly | the score table still runs |

`calibrate` reuses Part 4's answers from the same run; on its own it runs Part 4's questions first.

## What it needs

- **uv** 0.9 or later, the repo-root `.env`, and the local models cached:
  `uv run python ../../../setup/prefetch_models.py --module 3`.
- **Pinecone (optional):** with `PINECONE_API_KEY` set, `access` indexes on Pinecone (namespace
  `knowledge`, index `PINECONE_INDEX` or `m03-…`) and warns that a US region breaks POL-DAT-004.
  Without it, the same steps run on Chroma in `labs/module-03/outputs/uv-projects/lab03/chroma_db/`.
- **No connector credentials:** the SharePoint and Confluence exports in `data/connectors/` are read
  in place. The live readers are an instructor demo in the notebook.

## Expected output (deterministic)

Steps 1–3 and the score table do not depend on a model (trimmed):

```
30 published documents · excluded as retired: ['POL-WAR-001@2025']
{'first run': 145, 'unchanged rerun': 0, 'one document edited': 6, 'one document deleted': 0, 'vectors in the store now': 139}
✅ Part 2 — retired versions dropped, edits and deletes synced

eu_dealer_advisor      ['POL-WAR-001']
us_warranty_analyst    ['POL-GDW-003', 'POL-WAR-002']
✅ Part 3 — access trimming

lowest answerable: 1.56 · highest unanswerable: -5.87 · floor in use: -1.0
✅ the floor separates answerable from unanswerable questions
```

The offline tests reproduce the sync counts and the access trimming on a real Chroma store with
LlamaIndex's `MockEmbedding`, because neither depends on what the vectors mean.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt, plus Pinecone) · the `lab03` command
src/lab03/common.py       .env, the provider choice, llm.py and rag_kit.py in place, checks, print helpers (identical in all four projects)
src/lab03/part1_…part5_…  one module per lab part, each with a run() function
src/lab03/cli.py          the `lab03` command
tests/test_lab03.py       offline tests: the connector, the filters, sync and trimming on Chroma
```
