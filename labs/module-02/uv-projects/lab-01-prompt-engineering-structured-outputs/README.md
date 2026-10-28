# Lab 01 demo — Prompt engineering and structured outputs

A uv project that runs [Lab 01](../../lab-01-prompt-engineering-structured-outputs.md) from the
terminal. It holds the same code as the solution notebook, split into one module per part, with the
notebook's check cells turned into offline tests. The notebooks stay the main lab path; use this
project for a live demo, or if you prefer scripts to Jupyter.

## Run it

```bash
# (terminal) - from labs/module-02/uv-projects/lab-01-prompt-engineering-structured-outputs
uv sync                          # creates .venv/ with the pinned packages
uv run lab01 status              # which .env was loaded, which providers are ready
uv run lab01 techniques          # one part
uv run lab01 all                 # every part, in order
uv run lab01 all --provider anthropic   # one provider only
uv run pytest                    # the offline tests
```

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `baseline` | Part 1: the labelled set, the label contract, the scorer and the keyword baseline | no | runs |
| `techniques` | Part 2: zero-shot, few-shot and chain-of-thought on the 24 test tickets, per provider | yes | skipped with a note |
| `debug` | Part 3: error analysis, v2 with the policy, the regression check, token counts and prompt caching | yes | skipped with a note |
| `schema` | Part 4: JSON in the prompt vs an enforced schema on five change requests, and the deliberate failures | partly | the recorded-reply failure still runs |
| `validate` | Part 5: the validators, one retry with the errors, routing to a person | partly | the validators run on the gold records |

`debug` analyses the v1 runs from `techniques`. In `all` it reuses them; on its own it runs them
first (another 72 calls per provider). Parts 2 and 3 together make 240 classification calls on
two providers, eight at a time. The optional v3 template from the notebook's Part 3 is a learner
exercise and is not in this project.

## What it needs

- **uv** 0.9 or later ([install](https://docs.astral.sh/uv/getting-started/installation/)). uv reads
  `.python-version` and uses Python 3.12, downloading it if needed.
- **The repo-root `.env`** (`cp labs/setup/.env.example .env`). The project finds it by walking up
  from the current folder, as the notebooks do. It uses `OPENAI_API_KEY`, `OPENAI_MODEL`,
  `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`.

The project reads `labs/module-02/data/` in place; it does not copy it.

## Expected output (deterministic)

The labelled set, the keyword baseline and the validators use only `data/` and the code, so these
match the lab guide:

```
6 examples · 24 test tickets · languages {'en': 22, 'fr': 1, 'es': 1}
keyword baseline: {'category': 0.958, 'priority': 0.333, 'assignment_group': 0.875, 'all three': 0.25}
recorded reply → invalid JSON (Expecting value)
layer 1 → Value error, window_end must be after window_start
layer 2 → ['high-risk change without CAB approval']
```

Everything a model writes varies from run to run: record your own results in the guide's tables.
On the current Claude models the assistant-prefill probe in `schema` returns a 400; that is the
lesson, not a bug.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab01` command
src/lab01/common.py       .env, both SDK clients, the LangChain factory, print helpers (identical in all five projects)
src/lab01/part1_…part5_…  one module per lab part, each with a run() function
src/lab01/cli.py          the `lab01` command
tests/test_lab01.py       offline tests of the deterministic logic
```
