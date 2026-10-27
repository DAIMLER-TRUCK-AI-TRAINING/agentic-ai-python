# Lab 03 demo — Mapping use cases and selecting models

A uv project that runs [Lab 03](../../lab-03-use-cases-model-selection.md) from the terminal. It
holds the same code as the solution notebook, including the worked mappings for UC-1 to UC-4, one
module per part, with the notebook's check cells turned into offline tests. The notebooks stay the
main lab path; in the lab itself, the team writes the mappings and the scoring functions.

## Run it

```bash
# (terminal) - from labs/module-01/uv-projects/lab-03-use-cases-model-selection
uv sync                          # small: no PyTorch in this one
uv run lab03 status
uv run lab03 cost                # one part
uv run lab03 decide --scenario UC-3 --team "Blue team"
uv run lab03 all                 # every part; decide uses --scenario (default UC-1)
uv run pytest                    # the offline tests
```

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `scenarios` | Part 1: the four scenarios, the agent types, the model catalogue | no | runs |
| `mapping` | Part 2: the mappings, the consistency checks, and a `MappingReview` from both providers | partly | the review is skipped |
| `cost` | Part 3: token counts per counter, the UC-1 tier probe, the monthly cost table | partly | o200k_base counts and the cost table still run |
| `decide` | Part 4: ratings, the weighted matrix, sensitivity, and the filled comparison sheet | no | runs |

`decide` writes `labs/module-01/outputs/model-comparison-<scenario>.md` from
`templates/model-comparison-sheet.md` (the `outputs/` folder is git-ignored). The tie-breaker,
trade-off, review date and Responsible AI rows stay as `TODO` lines for the team to write.

## Expected output (deterministic)

The cost model and the matrix use only `data/` and the code, so these match the guide:

```
UC-1: claude-haiku-5-5 and gpt-6-luna tie at 4.25 ($13.19 a month each)
      → "Tie at 4.25 between …: apply the tie-breaker you agreed before scoring."
UC-3: gpt-6.1-terra 4.25 · gpt-oss-120b 3.75 · claude-sonnet-5-5 3.25
```

## What it needs

uv 0.9 or later and the repo-root `.env` (`cp labs/setup/.env.example .env`). It reads
`OPENAI_MODEL`, `OPENAI_SMALL_MODEL`, `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_MODEL`; the tier probe
skips any that are unset. For the offline o200k_base counts, cache the encoding once:
`uv run python ../../../setup/prefetch_models.py --module 1` (it needs `huggingface_hub`, so run it
from the Lab 01 or Lab 02 project, or the course venv).
