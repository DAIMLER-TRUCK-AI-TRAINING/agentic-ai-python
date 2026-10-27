# Lab 02 demo — Document intelligence: OCR and multilingual embeddings

A uv project that runs [Lab 02](../../lab-02-ocr-multilingual-embeddings.md) from the terminal. It
holds the same code as the solution notebook, one module per part, with the notebook's check cells
turned into offline tests. The notebooks stay the main lab path.

## Run it

```bash
# (terminal) - from labs/module-01/uv-projects/lab-02-ocr-multilingual-embeddings
uv sync                          # creates .venv/ with the pinned packages
uv run lab02 status
uv run lab02 accuracy            # one part
uv run lab02 all                 # every part, in order
uv run lab02 all --provider openai
uv run pytest                    # the offline tests
```

| Step | Lab guide part | Calls a model? | Without keys |
| ---- | -------------- | -------------- | ------------ |
| `tesseract` | Part 1: generate the samples, plain Tesseract, word-level confidence | no | runs |
| `accuracy` | Part 2: CER for psm 3, psm 4 and preprocessing, then regex field extraction | no | runs |
| `vision` | Part 3: OpenAI and Claude vision with the same schema, and the no-null schema failure | yes | skipped with a note |
| `cloud-ocr` | Part 4: adds the instructor's Azure results from `data/cloud_ocr/` to the CER table | no | prints a note if there are no results |
| `similarity` | Part 5: English-only and multilingual embeddings, the 6 × 6 matrix, the traps | no | runs |
| `matching` | Part 6: ticket-to-KB matching per language, escalation, image → KB pipeline, optional OpenAI embeddings | optional | runs |

## What it needs

- **uv** 0.9 or later, and the repo-root `.env` (`cp labs/setup/.env.example .env`).
- **Tesseract 5.x** on `PATH` (see `labs/setup/README.md`). Without it, the Tesseract steps skip,
  and the pipeline in Part 6 uses the ground-truth text instead of OCR.
- **The two Sentence Transformers models, cached:**
  `uv run python ../../../setup/prefetch_models.py --module 1`.

`data/samples/` is generated on first use by `data/make_samples.py`. The Azure demo script stays in
the course venv (`labs/module-01/instructor/`); this project only reads its results.
