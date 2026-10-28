# Architecture review — Enterprise Operations Assistant

> **Module 2 · Lab 05 artefact.** The Lab 05 notebook fills the measured sections from your runs and
> writes the result to `labs/module-02/outputs/architecture-review.md`. Write the decisions and
> mitigations yourself: a review is only as good as the evidence each row cites.

| Team | Date | Provider(s) | Architecture shipped |
| ---- | ---- | ----------- | -------------------- |
| {{team}} | {{date}} | {{providers}} | {{architecture}} |

## 1. Evaluation result (Lab 05 Part 4)

{{eval_table}}

**Decision:** {{decision}}

## 2. Where the context window goes (first model call of one run)

{{budget_table}}

## 3. Stress tests

{{stress_table}}

## 4. Where the time goes (one traced run)

{{waterfall_table}}

## 5. Risks and bottlenecks

| # | Risk | Evidence (numbers from §2–4) | Mitigation | Owner |
| - | ---- | ---------------------------- | ---------- | ----- |
| R1 | Tool output floods the context (an unbounded query or search result) | {{r1_evidence}} | {{r1_mitigation}} | {{r1_owner}} |
| R2 | A pasted attachment or log exceeds the budget | {{r2_evidence}} | {{r2_mitigation}} | {{r2_owner}} |
| R3 | The fixed prompt (spec + policy + tool schemas) grows with every rule added | {{r3_evidence}} | {{r3_mitigation}} | {{r3_owner}} |
| R4 | Latency: the slowest step on the critical path | {{r4_evidence}} | {{r4_mitigation}} | {{r4_owner}} |
| R5 | Your own: | {{r5_evidence}} | {{r5_mitigation}} | {{r5_owner}} |

## 6. What Module 3 and Module 4 should change

{{next_steps}}
