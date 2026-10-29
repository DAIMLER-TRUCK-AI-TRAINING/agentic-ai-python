# Lab environment setup

> **Course:** Agentic AI Systems Engineering (Python track)
> **Used by:** every lab in `labs/module-01/` … `labs/module-05/`

You set this up **once**, before Module 1. It gives you one Python virtual
environment for the whole course, a `.env` file with your keys and model IDs,
the Tesseract OCR binary, cached local models, and a check script that tells
you what is missing.

| File | What it is |
| ---- | ---------- |
| [`requirements.txt`](requirements.txt) | Every Python package the course uses, pinned and grouped by module |
| [`.env.example`](.env.example) | The environment variables the labs read: names only, no values |
| [`check_env.py`](check_env.py) | Prints a ✅ / ⚠️ / ❌ table of Python, packages, keys, Tesseract, Redis and local models |
| [`prefetch_models.py`](prefetch_models.py) | Downloads the open models the labs run locally (about 830 MB for Module 1) |
| [`llm.py`](llm.py) | Provider factory used from Module 2 Lab 04 on: `get_chat_model()`, `get_client()`, `get_model_name()`; from Module 3, also LlamaIndex models (`get_llama_llm()`), embeddings (`get_embeddings()`, `get_llama_embed_model()`) and the Azure providers |

---

## 1. Python 3.12 or 3.13 and the course venv

```bash
# (terminal) - from the repo root
python3.13 --version               # 3.12.x or 3.13.x (use python3.12 if that is what you have)
python3.13 -m venv .venv           # name the version: plain `python3` may be 3.14
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python --version                   # confirm the venv is 3.12 or 3.13
python -m pip install --upgrade pip
pip install -r labs/setup/requirements.txt
```

The install pulls PyTorch, so expect a few minutes and roughly 2 GB on disk.

> **Why not Python 3.14.** `guardrails-ai` (Module 5) declares
> `Requires-Python <3.14`, so on 3.14 pip skips it and stops with
> `No matching distribution found for guardrails-ai==0.11.0`. Older
> `guardrails-ai` releases do install on 3.14, but they pin `click<=8.2.0`,
> which conflicts with `huggingface-hub` 1.x (`click>=8.4.2`). Use 3.12 or 3.13
> until `guardrails-ai` supports 3.14.

> **Why two SDKs are pinned below their newest major version.** `openai` is
> held on 2.x because `llama-index-llms-openai` and `guardrails-ai` still
> require `openai<3`; `anthropic` is held on 1.x because `langchain-anthropic`
> still requires `anthropic<2`. One environment has to serve the direct-SDK
> labs and the framework labs, so the pins follow the frameworks. Everything
> the labs call exists in these versions.

## 2. Jupyter or VS Code

Either works. Use the `.venv` interpreter as the kernel.

```bash
# (terminal) - from the repo root, venv active
python -m ipykernel install --user --name agentic-ai --display-name "Agentic AI (.venv)"
jupyter lab labs/
```

In **VS Code**, open the repo folder, open any notebook, click **Select
Kernel → Python Environments** and pick `.venv`.

## 3. System packages

| Package | Needed for | Install | Fallback if missing |
| ------- | ---------- | ------- | ------------------- |
| **Tesseract OCR 5.x** | Module 1 Lab 02 (OCR) | macOS: `brew install tesseract` · Ubuntu/Debian: `sudo apt-get install tesseract-ocr` · Windows: the UB Mannheim installer, then add it to `PATH` | Lab 02 skips the Tesseract cells with a printed note; the AI-assisted OCR and embeddings parts still run |
| Redis 7+ (optional) | Module 4 episodic memory | `docker run -d --name course-redis -p 6379:6379 redis:7` | The labs switch to an in-memory store and say so |

English language data (`eng`) ships with Tesseract and is all the OCR lab needs.

## 4. Your `.env`

```bash
# (terminal) - from the repo root
cp labs/setup/.env.example .env
```

Open `.env` and fill in the two API keys the bootcamp provides. Leave the model
IDs as they are unless your instructor says otherwise:

| Variable | Default | Used by |
| -------- | ------- | ------- |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | (provided) | Every lab |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | `gpt-6.1-sol` / `claude-opus-5-5` | Every lab |
| `OPENAI_SMALL_MODEL` / `ANTHROPIC_SMALL_MODEL` | `gpt-6-luna` / `claude-haiku-5-5` | Module 1 Lab 01 (temperature sweep), Lab 03 (tier comparison), later routing labs |
| `LLM_PROVIDER` | `openai` | Module 2 Labs 04–05 and Phase 2 labs, through `llm.py` |
| `LANGSMITH_*`, `LANGFUSE_*`, `TAVILY_API_KEY`, `REDIS_URL` | empty | Modules 2–5; each lab has a fallback |
| `AZURE_VISION_*` | empty | Instructor only (Module 1 cloud OCR demo) |
| `AZURE_OPENAI_*`, `ANTHROPIC_FOUNDRY_*` | empty | Module 3 onwards, only with `LLM_PROVIDER=azure-openai` or `azure-anthropic`: deployment names, not model IDs |
| `EMBEDDING_PROVIDER`, `ST_MODEL`, `RERANK_MODEL` | `local`, the multilingual MiniLM, `ms-marco-MiniLM-L-6-v2` | Module 3 onwards: local embeddings and re-ranking, no key needed |
| `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_CLOUD`, `PINECONE_REGION` | empty, empty, `aws`, `us-east-1` | Module 3 Lab 03 (optional; Chroma otherwise) |

`.env` is git-ignored. Never paste a key into a notebook cell, a commit or a chat.

## 5. Cache the local models

```bash
# (terminal) - from the repo root, venv active
python labs/setup/prefetch_models.py --module 1
python labs/setup/prefetch_models.py --module 3   # the embedding model and the re-ranker for the RAG labs
```

This downloads `HuggingFaceTB/SmolLM2-135M` (Lab 01) and two Sentence
Transformers models (Lab 02) into the standard Hugging Face cache, plus the
`o200k_base` tiktoken encoding. Run it on a good connection before the day.

## 6. Run the check

```bash
# (terminal) - from the repo root, venv active
python labs/setup/check_env.py --module 1   # before Module 1
python labs/setup/check_env.py --module 2   # before Module 2
python labs/setup/check_env.py              # the whole course
```

**Expected** (abridged; versions and paths depend on your machine):

```
   Area                 Item                        Detail
   -------------------  --------------------------  ----------------------------------------
✅ Python               3.12.x                      /…/agentic-ai-python/.venv/bin/python
✅ Config               .env                        found and git-ignored
✅ Packages (core)      openai                      2.54.0
✅ Packages (core)      anthropic                   1.17.0
…
✅ Env (M1)             OPENAI_API_KEY              present
✅ Env (M1)             OPENAI_MODEL                gpt-6.1-sol
✅ Env (M1)             ANTHROPIC_API_KEY           present
✅ Env (M1)             ANTHROPIC_MODEL             claude-opus-5-5
ℹ️  Env (M1)             AZURE_VISION_*              not set: instructor-only, fine
✅ System               tesseract                   tesseract 5.5.x; langs: eng, osd
✅ Models (M1)          HuggingFaceTB/SmolLM2-135M  cached

0 ❌  ·  0 ⚠️   ·  scope: Module 1
```

Fix every ❌ row before the lab. A ⚠️ row has a fallback, and the lab tells
you what changes.

## Using `llm.py` (Module 2 Lab 04 onwards)

From Module 2 Lab 04 on, notebooks pick the provider from `LLM_PROVIDER`:

```python
# (notebook) - from labs/module-NN/notebooks
import sys; sys.path.append("../../setup")
from llm import describe, get_chat_model

print(describe())          # LLM_PROVIDER=openai · OPENAI_MODEL=gpt-6.1-sol · … · OPENAI_API_KEY present
llm = get_chat_model()     # ChatOpenAI or ChatAnthropic
```

`get_chat_model()` raises `ProviderConfigError` with the variable to fix when
a key or model ID is missing.

From Module 3 on, the same settings drive LlamaIndex and the embeddings, and
`LLM_PROVIDER` also accepts the Azure providers:

| `LLM_PROVIDER` | Calls | Comparison partner (`partner_provider()`) |
| -------------- | ----- | ----------------------------------------- |
| `openai` / `anthropic` | The providers' own APIs | the other one |
| `azure-openai` | Azure OpenAI v1 API on `AZURE_OPENAI_ENDPOINT` (Bearer key) | `azure-anthropic` |
| `azure-anthropic` | Claude on Microsoft Foundry, `https://<ANTHROPIC_FOUNDRY_RESOURCE>.services.ai.azure.com/anthropic` | `azure-openai` |

```python
# (notebook)
from llm import get_llama_llm, get_llama_embed_model, get_embeddings
llm = get_llama_llm()                 # OpenAIResponses or Anthropic, for LLM_PROVIDER
embed = get_llama_embed_model()       # EMBEDDING_PROVIDER: local (default), openai or azure-openai
```

Module 2 Labs 04–05 use `openai` or `anthropic` only.

## Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `❌ Python 3.11…` or `❌ Python 3.14…` | The venv was created with an unsupported interpreter | Delete `.venv`, recreate it with `python3.13 -m venv .venv` (or `python3.12`) |
| `No matching distribution found for guardrails-ai` | The venv is Python 3.14, which `guardrails-ai` does not support yet | Same fix: recreate the venv on 3.12 or 3.13 |
| `ResolutionImpossible` during `pip install` | An older pip, or another package installed into the venv first | `python -m pip install --upgrade pip`, then install into a fresh venv |
| `TesseractNotFoundError` | The binary is not on `PATH` | Install it (section 3), then restart the Jupyter kernel |
| `❌ .env missing` | The template was not copied | `cp labs/setup/.env.example .env` from the repo root |
| `AuthenticationError` in a lab | A wrong or expired key in `.env` | Fix the key, then restart the kernel: `.env` is read when the notebook starts |
| Models download during the lab | `prefetch_models.py` was not run | Run it now; it skips files already cached |
