# Module 1 labs as uv projects

The three Module 1 labs as small, standalone [uv](https://docs.astral.sh/uv/) projects: the same
code as the solution notebooks, run from the terminal one lab part at a time. Use them for a live
demo, or if you prefer scripts to Jupyter. The notebooks in `../notebooks/` and `../solutions/` stay
the main lab path, and nothing here changes them.

| Project | Lab | Command | Steps |
| ------- | --- | ------- | ----- |
| [`lab-01-exploring-llm-behaviour/`](lab-01-exploring-llm-behaviour/README.md) | [Lab 01](../lab-01-exploring-llm-behaviour.md) | `uv run lab01` | `first-calls` · `temperature` · `hallucination` · `context` |
| [`lab-02-ocr-multilingual-embeddings/`](lab-02-ocr-multilingual-embeddings/README.md) | [Lab 02](../lab-02-ocr-multilingual-embeddings.md) | `uv run lab02` | `tesseract` · `accuracy` · `vision` · `cloud-ocr` · `similarity` · `matching` |
| [`lab-03-use-cases-model-selection/`](lab-03-use-cases-model-selection/README.md) | [Lab 03](../lab-03-use-cases-model-selection.md) | `uv run lab03` | `scenarios` · `mapping` · `cost` · `decide` |

Every command also takes `all` (every step in order), `status` (which `.env` was loaded and which
providers are ready, never the key values) and `--provider openai|anthropic` to call one provider
only.

## Quick start

```bash
# (terminal) - from the repo root, once
cp labs/setup/.env.example .env          # then add your OpenAI and Anthropic keys

# (terminal) - then, per lab
cd labs/module-01/uv-projects/lab-01-exploring-llm-behaviour
uv sync                                  # .venv/ inside the project, from uv.lock
uv run lab01 status
uv run lab01 all
uv run pytest                            # offline tests, no keys needed
```

You need uv 0.9 or later. Each project pins Python 3.12 in `.python-version`; uv downloads it if
you do not have it. Lab 02 also needs Tesseract 5.x on `PATH` (see `labs/setup/README.md`).

## How they relate to the notebooks

| | Notebooks | uv projects |
| --- | --- | --- |
| Code | starter with `TODO`s, plus the solution | the solution, one module per lab part |
| Checks | check cells print ✅ / ⏳ / ⚠️ | offline `pytest` tests of the same contracts, plus the guide's deterministic numbers |
| Packages | the course venv, `labs/setup/requirements.txt` | one `.venv/` per project, with the same pins in `pyproject.toml` and a committed `uv.lock` |
| Settings and data | repo-root `.env`, `labs/module-01/data/` | the same files, read in place |
| Plots | thumbnails and a heatmap | tables only |
| Offline | each provider and model is skipped, with a note | the same fallbacks |

Each project is self-contained, so you can `cd` into one and run it without the others. They share
nothing but the data: `src/lab0N/common.py` (the `.env` loading, both SDK clients and the print
helpers) is the same file in all three. Lab 01 and Lab 02 install CPU-only PyTorch on Linux and
Windows from the PyTorch index, which avoids the much larger CUDA wheels; macOS uses PyPI.

## Cleaning up

```bash
# (terminal) - from labs/module-01/uv-projects
rm -rf */.venv                           # uv sync recreates them from uv.lock
rm -rf ../outputs                        # comparison sheets written by `lab03 decide`
```
