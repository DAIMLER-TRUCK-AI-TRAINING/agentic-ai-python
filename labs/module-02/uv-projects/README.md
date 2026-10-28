# Module 2 labs as uv projects

The five Module 2 labs as small, standalone [uv](https://docs.astral.sh/uv/) projects: the same
code as the solution notebooks, run from the terminal one lab part at a time. Use them for a live
demo, or if you prefer scripts to Jupyter. The notebooks in `../notebooks/` and `../solutions/` stay
the main lab path, and nothing here changes them.

| Project | Lab | Command | Steps |
| ------- | --- | ------- | ----- |
| [`lab-01-prompt-engineering-structured-outputs/`](lab-01-prompt-engineering-structured-outputs/README.md) | [Lab 01](../lab-01-prompt-engineering-structured-outputs.md) | `uv run lab01` | `baseline` · `techniques` · `debug` · `schema` · `validate` |
| [`lab-02-one-assistant-two-sdks/`](lab-02-one-assistant-two-sdks/README.md) | [Lab 02](../lab-02-one-assistant-two-sdks.md) | `uv run lab02` | `tools` · `openai` · `streaming` · `claude` · `compare` |
| [`lab-03-skills-and-openspec/`](lab-03-skills-and-openspec/README.md) | [Lab 03](../lab-03-skills-and-openspec.md) | `uv run lab03` | `pattern` · `skills` · `openspec` · `scenarios` |
| [`lab-04-langchain-agent-memory/`](lab-04-langchain-agent-memory/README.md) | [Lab 04](../lab-04-langchain-agent-memory.md) | `uv run lab04` | `chains` · `tools` · `agent` · `buffer` · `memory` |
| [`lab-05-enterprise-operations-assistant/`](lab-05-enterprise-operations-assistant/README.md) | [Lab 05](../lab-05-enterprise-operations-assistant.md) | `uv run lab05` | `brief` · `build` · `policy` · `evaluate` · `review` |

Every command also takes `all` (every step in order), `status` (which `.env` was loaded and which
providers are ready, never the key values) and `--provider openai|anthropic`:

- **Labs 01–03** call both SDKs side by side, as the notebooks do; `--provider` calls one only.
- **Labs 04–05** run on `LLM_PROVIDER` and use the other provider for their comparison steps;
  `--provider` runs on that one only, with no comparison.

## Quick start

```bash
# (terminal) - from the repo root, once
cp labs/setup/.env.example .env          # then add your OpenAI and Anthropic keys, and LLM_PROVIDER

# (terminal) - then, per lab
cd labs/module-02/uv-projects/lab-01-prompt-engineering-structured-outputs
uv sync                                  # .venv/ inside the project, from uv.lock
uv run lab01 status
uv run lab01 all
uv run pytest                            # offline tests, no keys needed
```

You need uv 0.9 or later. Each project pins Python 3.12 in `.python-version`; uv downloads it if
you do not have it. No system packages are needed for Module 2.

## How they relate to the notebooks

| | Notebooks | uv projects |
| --- | --- | --- |
| Code | starter with `TODO`s, plus the solution | the solution, one module per lab part |
| Checks | check cells print ✅ / ⏳ / ⚠️ | each step prints the same checks as ✅ / ⚠️; offline `pytest` tests cover the deterministic contracts |
| Packages | the course venv, `labs/setup/requirements.txt` | one `.venv/` per project, with the same pins in `pyproject.toml` and a committed `uv.lock` |
| Provider (Labs 04–05) | `labs/setup/llm.py` | the same factory, inside `common.py` |
| Settings and data | repo-root `.env`, `labs/module-02/data/` | the same files, read in place; Lab 05 imports `../ops_assistant/` in place, as the notebook and Module 5 do |
| Outputs | `labs/module-02/outputs/` | `labs/module-02/outputs/uv-projects/`, so a run never overwrites a trace sheet, Skill or review you edited |
| Plots | Lab 04's token chart | tables only |
| Offline | each provider is skipped, with a note; mock web search without `TAVILY_API_KEY`; a local JSONL trace without LangSmith | the same fallbacks |

Each project is self-contained, so you can `cd` into one and run it without the others. They share
nothing but the data: `src/lab0N/common.py` (the `.env` loading, both SDK clients, the LangChain
model factory and the print helpers) is the same file in all five. Labs 01–03 install the two SDKs
only; Labs 04–05 add LangChain, LangGraph and `langchain-tavily`.

Later steps reuse what earlier steps of the same run produced (Lab 01's v1 runs, Lab 02's traces,
Lab 04's buffer run, Lab 05's evaluation). Run a later step on its own and it makes what it needs
first, and says so.

## What a full run costs

Measured with `gpt-5-mini` and `claude-haiku-5-5`, `all` on both providers, five projects at once.
The big spenders are Lab 01 Parts 2–3 (240 classification calls) and Lab 05 Part 4 (32 assistant
runs); Lab 05's raw-log stress run alone sends 150k–250k input tokens.

| Command | Model calls (both providers, approximate) | Wall time (measured) |
| ------- | ----------------------------------------- | -------------------- |
| `uv run lab01 all` | ≈ 290 | 1 min 50 s |
| `uv run lab02 all` | ≈ 30 | 1 min |
| `uv run lab03 all` | ≈ 30 | 30 s |
| `uv run lab04 all` | ≈ 90 | 3 min 30 s |
| `uv run lab05 all` | ≈ 95 | 2 min |

## Cleaning up

```bash
# (terminal) - from labs/module-02/uv-projects
rm -rf */.venv                           # uv sync recreates them from uv.lock
rm -rf ../outputs/uv-projects            # trace sheets, the generated Skill, the Lab 04 trace, the review
rm -f ../data/ops.db                     # build_ops_db.py recreates it identically
```
