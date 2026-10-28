# Lab 04 demo — Building a tool-enabled LangChain agent with memory

A uv project that runs [Lab 04](../../lab-04-langchain-agent-memory.md) from the terminal. It holds
the same code as the solution notebook, including the SQL validator and the window-memory function
the lab asks you to write, one module per part, with the notebook's check cells turned into offline
tests. The notebooks stay the main lab path.

## Run it

```bash
# (terminal) - from labs/module-02/uv-projects/lab-04-langchain-agent-memory
uv sync
uv run lab04 status              # the provider it will run on, and the comparison provider
uv run lab04 tools               # one part, no keys needed
uv run lab04 all                 # every part, in order
uv run lab04 all --provider anthropic   # Claude only, no comparison runs
uv run pytest                    # the offline tests
```

From this lab on, the provider comes from `LLM_PROVIDER` in `.env`, as in the notebook (which uses
`labs/setup/llm.py`; this project has the same factory in `common.py`). Steps that compare
providers also run the other one when it is configured. `--provider openai|anthropic` runs on that
provider only, whatever `LLM_PROVIDER` says.

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `chains` | Part 1: prompt \| model \| parser, `batch`, and structured output through LangChain on both providers | yes | skipped with a note |
| `tools` | Part 2: web search (Tavily or the offline mock), `search_kb`, the SQL validator, `query_ops_db`, `summarise` | no | runs |
| `agent` | Part 3: `create_agent` with limits on questions A–D, the same code on the other provider, the trace | yes | the SQL ground truth only |
| `buffer` | Part 4: stateless vs checkpointed, then the 10-turn conversation with buffer memory | yes | skipped with a note |
| `memory` | Part 5: window and summary memory against the buffer: tokens per turn and recall | yes | skipped with a note |

`memory` reuses the buffer run from `buffer` in the same run; on its own it runs the buffer first.
The agent's trace goes to `labs/module-02/outputs/uv-projects/lab04-trace.jsonl`, unless
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set, in which case LangSmith records it instead.

## What it needs

- uv 0.9 or later and the repo-root `.env` (`cp labs/setup/.env.example .env`), with `LLM_PROVIDER`,
  both keys and both model IDs.
- Optional: `OPENAI_SMALL_MODEL` / `ANTHROPIC_SMALL_MODEL` for the `summarise` tool and summary
  memory (otherwise the main model summarises), `TAVILY_API_KEY` for live web search, and LangSmith.
- The SQLite database: the first step that needs it builds `labs/module-02/data/ops.db` with
  `data/build_ops_db.py` (git-ignored, identical on every build).

## Expected output (deterministic)

The validator, the database and the mock search use only `data/` and the code:

```
✅ blocked 'SELECT 1; DELETE FROM incidents'                                  → only one statement is allowed
✅ blocked 'WITH x AS (DELETE FROM incidents RETURNING *) SELECT * FROM x'    → keyword DELETE is not allowed
DELETE → {"error": "rejected", "reason": "only SELECT statements are allowed"}
COUNT  → {"columns": ["COUNT(*)"], "rows": [[437]], "truncated": false}
Ground truth for A, straight from SQL: vpn-gateway-eu-01 with 5 incidents
```

On Claude, window and summary memory work because the project strips old thinking blocks and
summarises only at turn boundaries (the guide's Part 5 explains why). Remove either fix and the run
stops with `400 … Invalid signature in thinking block`.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab04` command
src/lab04/common.py       .env, both SDK clients, the LangChain factory, print helpers (identical in all five projects)
src/lab04/part1_…part5_…  one module per lab part, each with a run() function
src/lab04/cli.py          the `lab04` command
tests/test_lab04.py       offline tests of the deterministic logic (they build data/ops.db if it is missing)
```
