# Retrieval evaluation rubric (Lab 02 Part 4, reused in Lab 03)

> Fill in one column per configuration you evaluated. Set the thresholds per use case **before**
> you look at the numbers, and keep this file next to the code: it is the release gate.

**Use case:** `__________________`  ·  **Labelled set:** `data/test_queries.json` (24 answerable, 3 unanswerable)

| Check | Threshold | Config A: ________ | Config B: ________ | If it fails |
| ----- | --------- | ------------------ | ------------------ | ----------- |
| Recall@4 on answerable queries | ≥ 0.95 | | | Do not ship; fix retrieval first (hybrid, chunking) |
| MRR | ≥ 0.85 | | | Re-rank, or add the title to the embedded text |
| Identifier queries found (fault codes, part numbers) | 100 % | | | Add keyword search (BM25) |
| Cross-lingual queries found (German ↔ English) | 100 % | | | Multilingual embeddings; check the re-ranker's languages |
| Unanswerable queries refused | 100 % | | | Score floor; stricter grounding prompt |
| Answer relevance (LLM judge, mean of answerable) | ≥ 4 / 5 | | | Prompt, chunk size, k |
| Judges agree (both providers) | same pass/fail | | | Calibrate the judge prompt on hand-labelled answers |
| Cited IDs were retrieved | 100 % | | | Validate citations in code before showing the answer |
| Latency per query (retrieval only) | ≤ 500 ms | | | Smaller pool, cheaper re-ranker, cache embeddings |
| Regression vs the last release | no drop beyond one query | | | Block the change |

## Verdict

> Ship config: ________ · Main risk left: ________ · Next measurement: ________
