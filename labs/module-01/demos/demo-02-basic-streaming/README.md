# Basic Streaming Implementation using OpenAI-Compatible SDK

This application demonstrates how to **convert a standard LLM API call into a streaming call** for a better user experience.
It is based on *Example 1: Basic Streaming Implementation* from the Sprint Master document and is implemented using **uv** for environment management.

---

## Features

* **Streaming Response:** Enables `stream=True` for real-time token-by-token output
* **Switchable Provider:** `LLM_PROVIDER=gemini | openai | azure` picks the backend; the streaming code is identical for all three
* **Progressive Display:** Prints model output as it’s generated
* **Same Setup as Non-Streaming:** Reuses existing API initialization
* **Better UX:** Immediate feedback without waiting for full completion
* **Compatible with Older OpenAI SDKs:** Works even when `.stream()` helper is not available

---

## Project Structure

```bash
demo-2-basic-streaming/
├── main.py                      # Streaming implementation
├── .env.example                # Template for environment variables
├── .env                        # Your environment variables (copy of .env.example)
├── .gitignore                  # Ignore sensitive/config files
└── README.md                   # Documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-2-basic-streaming
cd demo-2-basic-streaming
```

---

### 2. Create Virtual Environment

```bash
uv venv
```

Activate the virtual environment:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
uv add openai python-dotenv
```

---

### 4. Configure Environment Variables

Copy `.env.example` to `.env`, set `LLM_PROVIDER`, and fill in the variables for that provider.

```bash
cp .env.example .env
```

| `LLM_PROVIDER` | Calls | Variables to set |
| --- | --- | --- |
| `gemini` (default) | Google Gemini (OpenAI-compatible endpoint) | `GEMINI_API_KEY`, `GEMINI_BASE_URL`, `GEMINI_MODEL` |
| `openai` | OpenAI | `OPENAI_API_KEY`, `MODEL_NAME` (default `gpt-4o-mini`) |
| `azure` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

Example `.env` for Azure OpenAI:

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
```

> On Azure, `model` is the **deployment name** you created in the portal, not the base model name (for example `gpt-4o`). The code uses Azure OpenAI's v1 API, so the regular `OpenAI` client works: only the key and `base_url` (`<endpoint>/openai/v1/`) change.
>
> Azure may send a first streamed chunk with an empty `choices` list (content-filter results). The loop already skips chunks without choices, so no change is needed.

---

## Usage

Run the script using `uv`:

```bash
uv run main.py
```

---

## Example Output

```bash
=== Streaming Response ===

List comprehensions in Python provide a concise way to create lists.
Example: [x**2 for x in range(5)] → [0, 1, 4, 9, 16]
They are faster and more readable than using loops.
```

---

## Key Observations

* **Text appears progressively**, not all at once
* **Users see immediate feedback** as tokens arrive
* **Natural reading flow** similar to real-time typing
* **Overall response time remains the same**, but perceived performance is improved

---

## Key Concepts

| Step | Concept               | Description                             |
| ---- | --------------------- | --------------------------------------- |
| 1    | Initialize Client     | Same as non-streaming setup (provider chosen by `LLM_PROVIDER`) |
| 2    | Enable Streaming      | Add `stream=True` to API call           |
| 3    | Receive Stream        | Response returned as a generator        |
| 4    | Iterate Chunks        | Loop through each streamed token        |
| 5    | Display Progressively | Print chunks using `sys.stdout.flush()` |

---


