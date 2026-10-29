# Chunking workbook (Lab 02 Part 2)

> Copy this file to `outputs/` and fill it in. One row per strategy, measured with the Lab 02
> harness (24 labelled queries, dense retrieval, k = 4). Note the embedding model you used.

**Embedding model:** `__________________`  ·  **Corpus:** `data/corpus/` (20 documents)

| Strategy | Settings | Nodes | Median chars | Precision@4 | Recall@4 | Hit rate@4 | MRR | Read one chunk: could you answer from it? |
| -------- | -------- | ----- | ------------ | ----------- | -------- | ---------- | --- | ----------------------------------------- |
| Fixed | `SentenceSplitter(128, 16)` | | | | | | | |
| Fixed | `SentenceSplitter(512, 50)` | | | | | | | |
| Semantic | `SemanticSplitterNodeParser(90th percentile)` | | | | | | | |
| Structure-aware | `MarkdownNodeParser()` | | | | | | | |
| Hierarchical *(stretch)* | `HierarchicalNodeParser([512, 128])` + `AutoMergingRetriever` | | | | | | | |

## Questions

1. Which strategy has the best precision? Is that because its chunks are better, or because they
   are smaller (four small chunks of the right document fill the top 4)?
2. Which strategy keeps a table (for example the AC/DC charging table in `KB-0701`) with its heading?
3. Recall barely moves between strategies on 24 queries. How many queries would you need before a
   difference of one query (0.04) means something?
4. Which strategy would you ship for the workshop corpus, and what would you re-measure when the
   corpus grows from 20 to 3,000 documents?

## Decision

> Strategy: ________ · Settings: ________ · Why (2–3 sentences): ________
