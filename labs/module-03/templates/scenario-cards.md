# Fine-tuning decision workshop: four scenario cards (Lab 04 Part 3)

> 30 minutes, teams of 3–4 (from the delivery schedule). Each team fills one canvas per card in
> [`fine-tuning-decision-canvas.md`](fine-tuning-decision-canvas.md), then presents one decision and
> the evaluation that would overturn it. The scenarios are illustrative, and the numbers are synthetic.

---

## Card A — Workshop repair assistant

Dealer technicians in 30 countries ask how to fix a fault: "FC-ADB-0412 on an LH-40 built in March".

- **Content:** 3,000 technical service bulletins and repair guides, in English and German; about 40
  are added or changed every week; withdrawn bulletins must stop being used at once.
- **Users:** need the bulletin number and the steps, and they check the source before they touch a vehicle.
- **Volume and latency:** 8,000 questions a day; 5 seconds is acceptable.
- **Labelled data:** none yet, apart from the bulletins themselves.

## Card B — Warranty claim coding

Every warranty claim carries a technician's free-text "complaint, cause, correction". The warranty
team assigns one of **380 damage codes** before the claim can be paid.

- **Labelled data:** 200,000 historical claims with their final code.
- **Volume and latency:** 25,000 claims a month; coding runs inside the Dealer Workshop Portal
  submission, with a **300 ms** budget.
- **Change:** about ten codes are added or retired per year.
- **Cost:** must stay well below the cost of the manual coding it replaces.

## Card C — Goodwill decision letters

After a goodwill decision (POL-GDW-003), the regional warranty team writes to the customer.

- **Problem:** letters vary in tone and structure, and some describe goodwill as "warranty", which
  the policy forbids.
- **Languages:** 9, with the same structure in each.
- **Volume:** about 40 letters a day; each is reviewed by a person before it is sent.
- **Examples:** 60 approved letters that reviewers rated "exemplary".

## Card D — Engineering specification assistant

Engineers ask about values in specifications: "What coolant flow raises FC-EAX-1310 on the ED-26?"

- **Content:** 1,200 Confluence specifications (`ENG-SPEC-…`) with tables of exact values, part
  codes and engineering change numbers, in mixed German and English jargon; specs change weekly.
- **Labelled data:** 400 question-and-answer pairs written by engineers.
- **Observation so far:** dense retrieval misses queries made only of part codes, and when the right
  passage is retrieved the model sometimes confuses two similar internal abbreviations.
- **Latency:** 10 seconds is fine; correctness and the exact citation matter most.

---

<details>
<summary>Trainer notes (reveal after the presentations)</summary>

| Card | Reference decision | Key reasons | What would overturn it |
| ---- | ------------------ | ----------- | ---------------------- |
| A | RAG (hybrid retrieval, citations, access by dealer region) | Knowledge gap; weekly changes; provenance and withdrawal | Recall@4 below 0.9 on a labelled workshop set even after hybrid search and re-ranking |
| B | Few-shot on the small tier first; fine-tune a small open-weight classifier if cost or the 300 ms budget fails | Behaviour gap with a fixed label set and abundant labels | The small-tier few-shot baseline meets accuracy and p95 latency on a held-out month |
| C | Prompt engineering: a template plus examples per language, and a rule check for the word "warranty" | Behaviour gap that a template fixes; low volume; a person reviews every letter | Reviewers still reject more than 10 % for tone after two prompt iterations |
| D | Hybrid: BM25 + vector retrieval for the codes; consider RAFT-style fine-tuning only if retrieval is right but answers misread the jargon | Knowledge gap plus a domain-language gap; weekly changes need RAG | Answer accuracy is already above 95 % once the right passages are retrieved |

</details>
