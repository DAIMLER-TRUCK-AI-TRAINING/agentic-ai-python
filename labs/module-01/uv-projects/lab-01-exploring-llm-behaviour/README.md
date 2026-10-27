# Lab 01 demo — Exploring LLM behaviour with OpenAI and Claude

A uv project that runs [Lab 01](../../lab-01-exploring-llm-behaviour.md) from the terminal. It
holds the same code as the solution notebook, split into one module per part, with the notebook's
check cells turned into offline tests. The notebooks stay the main lab path; use this project for a
live demo, or if you prefer scripts to Jupyter.

## Run it

```bash
# (terminal) - from labs/module-01/uv-projects/lab-01-exploring-llm-behaviour
uv sync                          # creates .venv/ with the pinned packages
uv run lab01 status              # which .env was loaded, which providers are ready
uv run lab01 temperature         # one part
uv run lab01 all                 # every part, in order
uv run lab01 all --provider anthropic   # one provider only
uv run pytest                    # the offline tests
```

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `first-calls` | Part 1: the first call to each provider and the shape of each response | yes | skipped with a note |
| `temperature` | Part 2: SmolLM2 temperature table and sampled continuations, then the deliberate `temperature` failures, the `OPENAI_SMALL_MODEL` sweep and the default-sampling comparison | partly | the local model still runs |
| `hallucination` | Part 3: probes H1–H5, DOI and version checks, grounding with a verbatim quote checked in code | yes | lists the probe set only |
| `context` | Part 4: limits, tokens per language, truncation, needle in a haystack, the pre-flight guard | partly | o200k_base token counts and the guard still run |

`--provider openai|anthropic` limits a run to one provider: useful when you demo one SDK at a time
or want to halve the spend. The `all` step makes roughly 60 requests, about 40 of them small; the
needle and counting step in Part 4 sends four prompts of about 14,400 tokens per provider, and the
step prints its own estimate before it sends them.

## What it needs

- **uv** 0.9 or later ([install](https://docs.astral.sh/uv/getting-started/installation/)). uv reads
  `.python-version` and uses Python 3.12, downloading it if needed.
- **The repo-root `.env`** (`cp labs/setup/.env.example .env`). The project finds it by walking up
  from the current folder, as the notebooks do. It uses `OPENAI_API_KEY`, `OPENAI_MODEL`,
  `OPENAI_SMALL_MODEL`, `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`.
- **SmolLM2-135M and o200k_base, cached.** If you have not already run the prefetch script from the
  course venv, run it with this project's environment:
  `uv run python ../../../setup/prefetch_models.py --module 1`.
  Without the model, Part 2 falls back to an illustrative logits vector and says so.

The project reads `labs/module-01/data/` in place; it does not copy it.

## Expected output (deterministic)

These come from the pinned SmolLM2, fixed seeds and o200k_base, so your numbers should match
(trimmed):

```
T=0.2: entropy  0.99 nats · top-1 probability 0.574
T=1.5: entropy  6.84 nats · top-1 probability 0.030
 greedy:  1 distinct of  1 · e.g. ' reset your password.\n\nIf you'
  T=0.3: 18 distinct of 20 · …
  T=1.5: 20 distinct of 20 · …
600 lines ≈ 14,417 tokens per prompt (o200k_base estimate)
🛑 offline: needs 16,417 tokens (prompt + 2,000 reserved for output) but the budget is 8,000
```

Everything a hosted model writes varies from run to run: record your own results in the guide's
observation tables.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · `lab01` command · CPU-only torch off macOS
src/lab01/common.py       .env, both SDK clients, print helpers (identical in the three Module 1 projects)
src/lab01/part1_…part4_…  one module per lab part, each with a run() function
src/lab01/cli.py          the `lab01` command
tests/test_lab01.py       offline tests of the deterministic logic
```
