# Lab 02 demo — Tuning and evaluating retrieval

A uv project that runs [Lab 02](../../lab-02-tuning-evaluating-retrieval.md) from the terminal. It
holds the same code as the solution notebook (your `retrieval_metrics` TODO filled in), one module
per part, with the notebook's check cells printed as ✅ / ❌ and the deterministic contracts as
offline tests. `labs/setup/llm.py` and `labs/module-03/rag_kit.py` are imported in place, as the
notebooks import them. The notebooks stay the main lab path.

## Run it

```bash
# (terminal) - from labs/module-03/uv-projects/lab-02-tuning-evaluating-retrieval
uv sync
uv run lab02 status
uv run lab02 hybrid              # one part, no keys needed
uv run lab02 all                 # every part, in order
uv run lab02 judge --provider anthropic
uv run pytest                    # the offline tests
```

The answers come from `LLM_PROVIDER`; the judge runs on the small tier of that provider and its
partner. `--provider` runs on that one only.

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `metrics` | Part 1: the 27 labelled queries and the four metrics on hand-worked cases | no | runs |
| `chunking` | Part 2: fixed-128, fixed-512, semantic and markdown nodes, measured; the chunks behind the numbers | no | runs |
| `hybrid` | Part 3: dense, BM25, hybrid and hybrid + re-rank, then the per-query misses | no | runs |
| `judge` | Part 4: five answers from the hybrid retriever, graded by a small-tier judge on both providers | yes | skipped with a note |

`hybrid` and `judge` reuse the indexes `chunking` built in the same run; on their own they build them first.

## What it needs

uv 0.9 or later, the repo-root `.env`, and the local embedding and re-ranking models cached:
`uv run python ../../../setup/prefetch_models.py --module 3`. The four Chroma collections go to
`labs/module-03/outputs/uv-projects/lab02/chroma_db/` (git-ignored) and are rebuilt on every run.

## Expected output (deterministic)

With the local embedding model, `chunking` and `hybrid` print the lab guide's tables exactly (trimmed):

```
           nodes  median chars  precision  recall  hit_rate  mrr
fixed-128  72      331          0.66       0.92    0.96      0.84
fixed-512  20     1073          0.27       0.96    0.96      0.86
semantic   57      321          0.57       0.90    0.92      0.82
markdown   89      212          0.80       0.92    0.96      0.84

                 precision  recall  hit_rate  mrr
dense            0.80       0.92    0.96      0.84
bm25             0.79       0.96    1.00      0.96
hybrid           0.83       0.98    1.00      0.95
hybrid + rerank  0.77       0.90    0.96      0.96

     dense  bm25  hybrid  hybrid + rerank query                                                    kind
q10  0      1     1       1               Was bedeutet der Fehlercode FC-BRK-0705?                 identifier
q11  1      1     1       0               Wie lange dauert das Schnellladen von 20 auf 80 Prozent? cross-lingual
✅ Part 3 — hybrid retrieval
```

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab02` command
src/lab02/common.py       .env, the provider choice, llm.py and rag_kit.py in place, checks, print helpers (identical in all four projects)
src/lab02/part1_…part4_…  one module per lab part, each with a run() function
src/lab02/cli.py          the `lab02` command
tests/test_lab02.py       offline tests: the metrics, the labelled set, BM25's row of the table
```
