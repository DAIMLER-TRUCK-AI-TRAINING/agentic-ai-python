# Lab 02 demo — One assistant, two SDKs: OpenAI function calling vs Claude tool use

A uv project that runs [Lab 02](../../lab-02-one-assistant-two-sdks.md) from the terminal. It holds
the same code as the solution notebook, including the two functions the lab asks you to write
(`to_claude_tool` and `dispatch`), one module per part, with the notebook's check cells turned into
offline tests. The notebooks stay the main lab path.

## Run it

```bash
# (terminal) - from labs/module-02/uv-projects/lab-02-one-assistant-two-sdks
uv sync
uv run lab02 status
uv run lab02 tools               # one part, no keys needed
uv run lab02 all                 # every part, in order
uv run lab02 all --provider openai      # one SDK at a time, for a live demo
uv run pytest                    # the offline tests
```

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `tools` | Part 1: the tool functions, one definition in both formats, the dispatcher's error contract | no | runs |
| `openai` | Part 2: your own Responses API tool loop on Q1–Q3, and the trace of Q2 | yes (OpenAI) | skipped with a note |
| `streaming` | Part 3: Q2 streamed, time to first token, and the event types of one streamed turn | yes (OpenAI) | skipped with a note |
| `claude` | Part 4: the Messages API tool loop on Q1–Q3, Q2 streamed, and the SDK's tool runner (beta) | yes (Claude) | skipped with a note |
| `compare` | Part 5: the runs side by side, deliberate failures A–C, and the agent trace sheets | yes | skipped with a note |

`compare` uses the traces from the earlier steps of the same run; on its own, it runs Q1–Q3 first.
It writes one trace sheet per provider, filled from `templates/agent-trace-sheet.md`, to
`labs/module-02/outputs/uv-projects/agent-trace-<provider>-Q2.md` (git-ignored). Section 4,
Review, is left for you to fill in by hand.

## What it needs

uv 0.9 or later and the repo-root `.env` (`cp labs/setup/.env.example .env`). It reads
`OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`; the cost column comes
from `data/model_prices.json` and shows `—` for a model that file does not list.

## Expected output (deterministic)

The dispatcher's contract does not depend on a model, so `tools` prints exactly this (trimmed):

```
dispatch('reset_password', '{}') → {"error": "unknown_tool", "tool": "reset_password"}
dispatch('search_kb', '{not json') → {"error": "invalid_json", "detail": "Expecting property name enclosed in double quotes"}
dispatch('search_kb', {'q': 'vpn'}) → {"error": "bad_arguments", "detail": "search_kb() got an unexpected keyword argument 'q'"}
```

Deliberate failure C (a `strict` tool without `"additionalProperties": false`) returns a 400 from
both APIs, with each provider's own wording. Everything else a model writes varies from run to run.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab02` command
src/lab02/common.py       .env, both SDK clients, the LangChain factory, print helpers (identical in all five projects)
src/lab02/part1_tools.py  the tool contracts and the dispatcher both loops share
src/lab02/part2_…part5_…  one module per lab part, each with a run() function
src/lab02/cli.py          the `lab02` command
tests/test_lab02.py       offline tests of the deterministic logic
```
