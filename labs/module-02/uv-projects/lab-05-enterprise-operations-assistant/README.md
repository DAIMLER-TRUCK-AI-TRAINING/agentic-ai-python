# Lab 05 demo — Mini-project: the Enterprise Operations Assistant

A uv project that runs [Lab 05](../../lab-05-enterprise-operations-assistant.md) from the terminal.
It holds the same code as the solution notebook: both architectures, the policy layer, the
evaluation and the reference review, one module per part, with the notebook's check cells turned
into offline tests. The read-only tools and the schemas come from `labs/module-02/ops_assistant/`,
imported in place, exactly as the notebook and Module 5 Lab 03 import them. The notebooks stay the
main lab path; in the lab itself, your team chooses the architecture and writes the review.

## Run it

```bash
# (terminal) - from labs/module-02/uv-projects/lab-05-enterprise-operations-assistant
uv sync
uv run lab05 status
uv run lab05 policy              # one part, no keys needed
uv run lab05 all                 # every part, in order
uv run lab05 evaluate --architecture workflow   # one architecture
uv run lab05 review --team "Blue team"
uv run pytest                    # the offline tests
```

The provider comes from `LLM_PROVIDER` (as in Lab 04); the evaluation also runs the other provider
when it is configured. `--provider openai|anthropic` runs on that one only.

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `brief` | Part 1: the spec rendered as the system prompt, and the eight labelled cases | no | runs |
| `build` | Part 2: the agent (`create_agent`) and the workflow (three fixed steps), smoke-tested on EV-02 | yes | skipped with a note |
| `policy` | Part 3: `enforce_policy` on hand-made records with known answers | no | runs |
| `evaluate` | Part 4: 8 cases × each architecture × each provider, graded on labels; raw vs policy-checked | yes | skipped with a note |
| `review` | Part 5: the token budget and time waterfall, two stress tests, the review, and the packaged `ops_assistant` | partly | the stress table and the review still run |

| Option | Default | Effect |
| ------ | ------- | ------ |
| `--architecture agent\|workflow\|both` | `both` | what `build`, `evaluate` and `review` run |
| `--spec PATH` | your `outputs/enterprise-ops-assistant.spec.md` if it has no `TODO`s, else the reference spec | the spec rendered as the system prompt |
| `--team NAME` | `Reference solution` | the team name on the review |

`review` uses the evaluation from `evaluate` in the same run; on its own it runs the evaluation
first. It writes `labs/module-02/outputs/uv-projects/architecture-review.md` (git-ignored) from
`templates/architecture-review.md`, with your measured tables and the reference write-up. Your
team's review from the notebook, in `outputs/`, is never overwritten.

## What it needs

uv 0.9 or later and the repo-root `.env` (`cp labs/setup/.env.example .env`), with `LLM_PROVIDER`,
both keys and both model IDs. The first step that needs the database builds `data/ops.db`.

## Expected output (deterministic)

The policy layer and the stress inputs do not depend on a model (trimmed):

```
low/low ticket marked P1                    P1 → P4  Service Desk          True     Service Desk L2 …  priority P1 -> P4
query_ops_db: SELECT * FROM incidents, uncapped  437        21142
query_ops_db: same query, 50-row cap              50         2461
ticket with pasted log, raw                     2400        47446
ticket with pasted log, condensed                 14          301
```

The token figures are offline estimates (about 4 characters per token). The providers' own counts
for the raw log are several times higher; `review` prints them next to the outcome of each run.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab05` command
src/lab05/common.py       .env, both SDK clients, the LangChain factory, print helpers (identical in all five projects)
src/lab05/ops.py          imports ../../ops_assistant/ in place: tools, schemas, render_system_prompt
src/lab05/part1_…part5_…  one module per lab part, each with a run() function
src/lab05/cli.py          the `lab05` command
tests/test_lab05.py       offline tests of the deterministic logic
```
