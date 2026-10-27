# Basic OpenAI API Call with OpenAI SDK

Simple Python application that demonstrates how to make a **basic API call** to **OpenAI** using the **OpenAI SDK** and `uv` environment management. The same code also runs on **Azure OpenAI**: set `LLM_PROVIDER` in `.env`.

---

## Features

- **OpenAI API Call**: Uses the OpenAI SDK to call OpenAI models
- **Switchable Provider**: `LLM_PROVIDER=openai | azure` picks the backend; the API call itself never changes
- **Environment Management**: Securely loads API keys from `.env` using `python-dotenv`
- **Structured Prompts**: Handles both system and user message roles
- **Runtime Configuration**: Customize model, token limit, and temperature
- **Error Handling**: Detects missing keys, invalid responses, and network issues
- **Token Usage Reporting**: Prints prompt, completion, and total token counts

---

## Project Structure

```bash
demo-05-openai-via-open-ai-sdk/
├── main.py                      # Main Python script
├── .env.example                 # Template for environment variables
├── .env                         # Your environment variables (copy of .env.example)
├── .gitignore                   # Ignore environment/config files
└── README.md                    # Project documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-05-openai-via-open-ai-sdk
cd demo-05-openai-via-open-ai-sdk
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

### 3. Install or Sync Dependencies

```bash
uv add openai python-dotenv
```

or

```bash
uv sync
```

---

### 4. Configure Environment Variables

Copy `.env.example` to `.env`, set `LLM_PROVIDER`, and fill in the variables for that provider.

```bash
cp .env.example .env
```

| `LLM_PROVIDER` | Calls | Variables to set |
| --- | --- | --- |
| `openai` (default) | OpenAI | `OPENAI_API_KEY`, `MODEL_NAME` (default `gpt-4o-mini`) |
| `azure` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

Example `.env` for OpenAI:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
```

Example `.env` for Azure OpenAI:

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
```

> On Azure, `model` is the **deployment name** you created in the portal, not the base model name (for example `gpt-4o`). The code uses Azure OpenAI's v1 API, so the regular `OpenAI` client works: only the key and `base_url` (`<endpoint>/openai/v1/`) change.

---

## Usage

Run the script using `uv`:

```bash
uv run main.py
```

---

## Example Output

```bash
=== Generated Answer ===

Python is a high-level, versatile programming language known for its simplicity and readability. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python is widely used for web development, data analysis, artificial intelligence, scientific computing, and automation tasks.

--- Token usage ---
Prompt tokens: 18
Completion tokens: 35
Total tokens: 53
```

---

## Key Concepts

| Step | Concept               | Description                                 |
| ---- | --------------------- | ------------------------------------------- |
| 1    | Environment Setup     | Load secrets securely using `python-dotenv` |
| 2    | Client Initialization | Configure OpenAI SDK for OpenAI or Azure OpenAI via `LLM_PROVIDER` |
| 3    | Message Construction  | Build `system` and `user` message structure |
| 4    | API Invocation        | Use `client.chat.completions.create()`      |
| 5    | Response Parsing      | Extract text and usage details              |
| 6    | Error Handling        | Manage runtime and auth issues gracefully   |

---
