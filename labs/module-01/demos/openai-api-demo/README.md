# OpenAI API Demo

A simple Python demo that calls the OpenAI API to generate a response using the Responses API (`01_openai_api.py`) or the Chat Completions API (`02_openai_api_completions.py`). Both scripts run unchanged on **OpenAI** or **Azure OpenAI**: set `LLM_PROVIDER` in `.env`.

## Prerequisites

- Python 3.8 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys)) **or** an Azure OpenAI resource with a model deployment and its key

---

## Setup Instructions

### Step 1 — Clone or navigate to the project directory

**Mac / Windows (PowerShell):**
```bash
cd labs/module-01/demos/openai-api-demo
```

---

### Step 2 — Create a virtual environment

**Mac (Terminal):**
```bash
python3 -m venv .venv
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

---

### Step 3 — Activate the virtual environment

**Mac (Terminal):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

> If you get a script execution error on Windows, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 4 — Install dependencies

**Mac / Windows:**
```bash
pip install openai python-dotenv
```

---

### Step 5 — Configure environment variables

Copy the example env file and fill in your values:

**Mac (Terminal):**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Then open `.env` in any text editor, set `LLM_PROVIDER`, and fill in the variables for that provider:

| `LLM_PROVIDER` | Calls | Variables to set |
| --- | --- | --- |
| `openai` (default) | OpenAI | `OPENAI_API_KEY`, `MODEL_NAME` |
| `azure` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

**OpenAI:**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your-api-key-here...
MODEL_NAME=gpt-4o
```

> Common model names: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

**Azure OpenAI:**
```
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
```

> On Azure, the model is your **deployment name** (Azure portal > your resource > Keys and Endpoint / Model deployments), not the base model name. Use the resource endpoint only, with no path.

You can keep both blocks in `.env` and switch providers by changing only `LLM_PROVIDER`.

---

### Step 6 — Run the demo

**Mac / Windows:**
```bash
python 01_openai_api.py              # Responses API
python 02_openai_api_completions.py  # Chat Completions API
```

You should see a response printed to the terminal. The first line shows which provider, base URL and model were used.

Azure OpenAI's v1 API is OpenAI-compatible, so the same `OpenAI` client works for both providers: only the key and `base_url` (`<endpoint>/openai/v1/`) differ. `llm_client.py` makes that choice in one place. `03_azure_openai_api.py` is the stand-alone Azure version of `01_openai_api.py` with the setup inlined; it always uses Azure regardless of `LLM_PROVIDER`.

---

## Project Structure

```
openai-api-demo/
├── .env                          # Your local environment variables (not committed to git)
├── .env.example                  # Template for environment variables
├── .venv/                        # Virtual environment (not committed to git)
├── llm_client.py                 # Picks OpenAI or Azure OpenAI from LLM_PROVIDER
├── 01_openai_api.py              # Responses API (OpenAI or Azure)
├── 02_openai_api_completions.py  # Chat Completions API (OpenAI or Azure)
└── 03_azure_openai_api.py        # Stand-alone Azure OpenAI example (Responses API)
```

---

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```
