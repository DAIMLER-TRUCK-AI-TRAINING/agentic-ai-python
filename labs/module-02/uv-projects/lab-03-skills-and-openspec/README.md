# Lab 03 demo — Designing agents with Skills and OpenSpec

A uv project that runs [Lab 03](../../lab-03-skills-and-openspec.md) from the terminal. It holds
the same code as the solution notebook: the Skill loader, the reference answer to "write your own
Skill", and the reference spec in place of the one your team writes. Each part is one module, and
the notebook's check cells are offline tests. The notebooks stay the main lab path; in the lab
itself, your team writes the loader, the Skill and the spec.

## Run it

```bash
# (terminal) - from labs/module-02/uv-projects/lab-03-skills-and-openspec
uv sync
uv run lab03 status
uv run lab03 openspec            # one part, no keys needed
uv run lab03 all                 # every part, in order
uv run lab03 scenarios --spec ../../outputs/enterprise-ops-assistant.spec.md   # test your team's spec
uv run pytest                    # the offline tests
```

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `pattern` | Part 1: the example Skill's files, and pasted-prompt vs Skill-level token estimates | no | runs |
| `skills` | Part 2: the loader and its rules, the sandboxed `read_skill_file` tool, when each provider loads the Skill, and a second Skill | partly | the Skills are validated offline |
| `openspec` | Part 3: the template's skeleton, how each section becomes code, the linter on the partial spec | no | runs |
| `scenarios` | Part 4: lint the spec, render it as a system prompt, run every scenario as a test on both providers | partly | the spec is linted offline |

`scenarios` tests `solutions/enterprise-ops-assistant.spec.md` unless you pass `--spec`; it never
copies over or edits your team's file. `skills` writes its reference Skill to
`labs/module-02/outputs/uv-projects/skills/change-request-extraction/SKILL.md` (git-ignored), beside
the one you write from the notebook in `outputs/skills/`, not over it.

## What it needs

uv 0.9 or later and the repo-root `.env` (`cp labs/setup/.env.example .env`). It reads
`OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`. It reads `skills/`,
`templates/`, `data/` and `solutions/` under `labs/module-02/` in place.

## Expected output (deterministic)

The token estimates and the linter use only the files and the code, so these match the guide
(trimmed):

```
Lab 01 few-shot v2: policy + 6 examples, in every request                 1442
Skill level 1: name + description, in every request                         74
Skill level 2: SKILL.md body, only when triaging                           508
Escape attempt → not_found_or_outside_skill
• 39 TODO marker(s) left
• tool `get_service_status` is missing Purpose, Inputs, Output, Errors, Side effects, Permission
• 1 complete escalation rules; write at least 3
✅ spec lint: clean
7 runnable scenario(s) · system prompt ≈ 2,307 tokens
```

The first linter lines are for the partial spec (`openspec`); the clean result is for the reference
spec (`scenarios`). Which Skill a model loads, and which scenarios it passes, varies from run to run.

## Layout

```
pyproject.toml            pins (same as labs/setup/requirements.txt) · the `lab03` command
src/lab03/common.py       .env, both SDK clients, the LangChain factory, print helpers (identical in all five projects)
src/lab03/tool_loop.py    a compact tool loop for both providers (Lab 02 built it step by step)
src/lab03/part1_…part4_…  one module per lab part, each with a run() function
src/lab03/cli.py          the `lab03` command
tests/test_lab03.py       offline tests of the deterministic logic
```
