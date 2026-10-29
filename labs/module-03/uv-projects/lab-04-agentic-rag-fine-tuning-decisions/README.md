# Lab 04 demo — Agentic RAG and fine-tuning decisions

A uv project that runs [Lab 04](../../lab-04-agentic-rag-fine-tuning-decisions.md) from the
terminal. It holds the same code as the solution notebook (the `search_knowledge` tool and the four
reference canvases), one module per part, with the notebook's check cells printed as ✅ / ❌ and the
deterministic contracts as offline tests. `labs/setup/llm.py` and `labs/module-03/rag_kit.py` are
imported in place, as the notebooks import them. The notebooks stay the main lab path; in the lab
itself, your team fills in the canvases.

## Run it

```bash
# (terminal) - from labs/module-03/uv-projects/lab-04-agentic-rag-fine-tuning-decisions
uv sync
uv run lab04 status
uv run lab04 tool                # one part, no keys needed
uv run lab04 all                 # every part, in order
uv run lab04 agent --provider anthropic
uv run pytest                    # the offline tests
```

The agent and the critic run on `LLM_PROVIDER` and its partner (the critic on the small tier).
`--provider` runs on that one only.

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `tool` | Part 1: the knowledge base, and the `search_knowledge` contract on a hit, an unknown type and a miss | no | runs |
| `agent` | Part 2: a `FunctionAgent` on three tasks per provider, with a trace of every search; then the 2-step pipeline | yes | skipped with a note |
| `canvases` | Part 3: the four reference canvases, checked | no | runs |
| `critique` | Part 4: a structured `Critique` of each canvas from each provider's small tier | yes | skipped with a note |

The agent calls `search_knowledge` through `rk.agent_tool()`, which runs the tool on the event loop:
the local PyTorch re-ranker is not safe in LlamaIndex's worker thread on every platform (the lab
guide's "Kernel died" trap).

## What it needs

uv 0.9 or later, the repo-root `.env`, and the local models cached:
`uv run python ../../../setup/prefetch_models.py --module 3`. The knowledge base goes to
`labs/module-03/outputs/uv-projects/lab04/chroma_db/` (git-ignored) and is rebuilt on every run.

## Expected output (deterministic)

`uv run lab04 tool` with the local models:

```
145 nodes · user eu_aftersales_engineer sees 131 of them
search_knowledge('FC-BAT-2201 threshold', 'engineering') -> ['ENG-SPEC-0417']
search_knowledge('FC-BAT-2201', 'manual') -> NO_RESULTS: unknown doc_type 'manual'. Use one of bulletin, kb, faq, runbook, policy, contract, engineering or 'any'.
search_knowledge('What is the canteen menu on Friday?', 'any') -> NO_RESULTS: nothing relevant found. Rephrase the query, or try doc_type='any'.
✅ Part 1 — search_knowledge tool
```

How many searches the agent makes varies by provider and run; the check asserts what matters (no
search for small talk, every cited ID came from a search, the two-part task found the warranty).

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab04` command
src/lab04/common.py       .env, the provider choice, llm.py and rag_kit.py in place, checks, print helpers (identical in all four projects)
src/lab04/part1_…part4_…  one module per lab part, each with a run() function
src/lab04/cli.py          the `lab04` command
tests/test_lab04.py       offline tests: the tool contract, agent_tool on the event loop, the canvases, the critique schema
```
