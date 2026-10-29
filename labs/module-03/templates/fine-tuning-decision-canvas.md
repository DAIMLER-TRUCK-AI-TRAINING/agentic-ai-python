# Fine-tuning decision canvas (Lab 04 Part 3)

> One canvas per scenario card. Start with the first question: is the gap **knowledge** (the model
> does not know your facts) or **behaviour** (it knows enough but does not act the way you need)?

**Scenario:** `__________________`  ·  **Team:** `__________________`

| Box | Your answer |
| --- | ----------- |
| 1. The gap, found by which evaluation? | knowledge / behaviour: |
| 2. Data available (documents, labelled examples, how many) | |
| 3. Latency budget (p95) and volume per day | |
| 4. Cost per month you can accept | |
| 5. Update frequency of the facts or the behaviour | |
| 6. Provenance: must answers cite a source? Must you delete data on request? | |
| 7. Where can it run? (OpenAI fine-tuning is being wound down; Claude cannot be fine-tuned; open-weight models you host) | |
| 8. **Decision**: prompt engineering / RAG / fine-tuning / hybrid | |
| 9. Why, in 2–3 sentences, using boxes 2–6 | |
| 10. **The evaluation that would overturn it** (metric, data set, threshold) | |

## Quick rules

| If… | Then lean to… |
| --- | ------------- |
| Facts change weekly, users need the source, access control applies | RAG |
| The corpus is small and stable (fits in a cached prompt) | Prompt stuffing + caching |
| Tone, format or rules are wrong, and a template with examples fixes it | Prompt engineering |
| A fixed label set, many thousands of labelled examples, and latency or cost force a small model | Fine-tune an open-weight model |
| Retrieval is right but answers still misread the domain | Hybrid: RAG + fine-tuning (RAFT-style) |
