# Skills for the Module 2 labs

An Agent Skill is a folder with a `SKILL.md` file: YAML frontmatter (`name`, `description`) plus
instructions, and optionally reference files and scripts. [Lab 03](../lab-03-skills-and-openspec.md)
loads Skills from this folder and from `outputs/skills/` (yours), with progressive disclosure:

| Level | What | In context |
| ----- | ---- | ---------- |
| 1 | `name` and `description` | Every request, in the system prompt |
| 2 | The `SKILL.md` body | When the model reads it with `read_skill_file` |
| 3 | `reference/*.md`, `scripts/*.py` | Reference files when the body points there; scripts are run by your code, and only their result is used |

| Skill | Purpose |
| ----- | ------- |
| [`incident-triage/`](incident-triage/SKILL.md) | Triage an incident under POL-OPS-004; `scripts/check_triage.py` validates the result (`python check_triage.py '<json>'`, or import `problems()`) |

Rules the Lab 03 loader enforces (from the Agent Skills documentation): `name` is 1–64 lowercase
letters, digits or hyphens, contains neither `anthropic` nor `claude`, and matches the folder name;
`description` is 1–1,024 characters and says what the Skill does **and** when to use it; neither field
contains XML tags. Treat a Skill like code you install: read every file before you use one you did not write.
