# <Agent name> — agent spec

> **Module 2 · Lab 03 artefact (OpenSpec template).** Copy this file, replace every `<…>` and every
> `TODO`, and keep it next to the code. The walkthrough order is the outline's: **problem → spec →
> tool contracts → role boundaries → escalation rules**. Requirements and scenarios use
> [OpenSpec](https://github.com/Fission-AI/OpenSpec)'s syntax (`### Requirement:` with SHALL, and
> `#### Scenario:` with **WHEN** / **THEN** bullets), so a spec written here can move into an
> `openspec/specs/` folder later. You do not need the OpenSpec CLI for this lab.

| Field | Value |
| ----- | ----- |
| Spec ID | <SPEC-OPS-nnn> |
| Owner team | <team that answers for the agent in production> |
| Version | <0.1> |
| Status | <draft · reviewed · approved> |
| Related policies | <POL-…> |

## 1. Problem

<Two to four sentences: who has the problem, what it costs today (volume, time, errors), and what
"better" looks like in numbers.>

**In scope:** <the inputs the agent handles>

**Out of scope:** <what the agent must refuse or hand over>

**Success measures:** <measurable targets: accuracy on a labelled set, escalation precision, latency, cost per request>

## 2. Requirements

Each requirement uses SHALL and has at least one scenario. Scenarios become test cases.

### Requirement: <name>

The agent SHALL <observable behaviour>.

#### Scenario: <name>

- **WHEN** <input or situation>
- **THEN** <expected, checkable outcome>

## 3. Tool contracts

One block per tool. A tool contract is an API contract: inputs, outputs, errors, side effects and
who may call it.

### Tool: `<tool_name>`

| Field | Value |
| ----- | ----- |
| Purpose | <one sentence; this becomes the tool description the model reads> |
| Inputs | <name: type, format, constraints> |
| Output | <shape, size cap> |
| Errors | <what the tool returns when it fails; it never raises into the model> |
| Side effects | <none (read-only) · writes to …> |
| Permission | <who may trigger it; approval needed?> |

## 4. Role boundaries

| The agent MAY | The agent MUST NOT |
| ------------- | ------------------ |
| <…> | <…> |

## 5. Escalation rules

Each rule has a measurable trigger, a target and the information handed over.

| ID | Trigger (measurable) | Escalate to | Hand-over must include |
| -- | -------------------- | ----------- | ---------------------- |
| E1 | <…> | <…> | <…> |

## 6. Output contract

<The structured output the agent returns: schema name, fields, and the validation rules that code
enforces after the model answers.>

## 7. Open questions

- <decisions still to make, with an owner>
