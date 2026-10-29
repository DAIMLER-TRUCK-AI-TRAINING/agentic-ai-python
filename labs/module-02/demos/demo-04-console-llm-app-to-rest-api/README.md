# Converting Console LLM App to REST API with LangChain

This project demonstrates how to **transform a console-based AI application into a REST API** using **FastAPI**.
It converts the Study Buddy console app into a web service with HTTP endpoints. The answer comes from a **LangChain LCEL chain** (`prompt | model | StrOutputParser()`), and `LLM_PROVIDER` picks the model: **OpenAI**, **Anthropic Claude**, **Azure OpenAI** or **Claude on Microsoft Foundry**, each called with its own API key. The endpoints never change.
It is implemented using **uv** for environment management.

---

## Features

- **POST Endpoint:** `/query` accepts a question and returns the AI answer
- **Health Check:** `/health` reports the service status and the configured provider and model
- **LangChain Chain:** `ChatPromptTemplate | chat model | StrOutputParser()`, with a Study Buddy system prompt
- **Switchable Provider:** `LLM_PROVIDER=openai | anthropic | azure-openai | azure-anthropic`; `init_chat_model()` returns a `ChatOpenAI` or a `ChatAnthropic`
- **Async Endpoint:** `await chain.ainvoke()` keeps the server free for other requests while the model answers
- **Pydantic Validation:** Automatic request/response validation; an empty prompt is rejected with 422
- **Error Handling:** Provider errors become 502 (wrong key, model or deployment, rate limit), an unreachable endpoint 503
- **Auto-Generated Documentation:** Interactive Swagger UI at `/docs`

---

## Project Structure

```bash
demo-04-console-llm-app-to-rest-api/
├── main.py                      # FastAPI application
├── .env.example                 # Template for environment variables
├── .env                         # Your environment variables (copy of .env.example)
├── .gitignore                   # Ignore sensitive/config files
├── pyproject.toml               # Project dependencies
├── uv.lock                      # Lock file
└── README.md                    # Documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-04-console-llm-app-to-rest-api
cd demo-04-console-llm-app-to-rest-api
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
uv add "fastapi[standard]" langchain langchain-openai langchain-anthropic "anthropic<2" python-dotenv
```

or

```bash
uv sync
```

**Dependencies:**

- `fastapi[standard]` - Web framework, with Uvicorn and the `fastapi` CLI
- `langchain` - `init_chat_model()`, the one factory for every provider
- `langchain-openai` / `langchain-anthropic` - `ChatOpenAI` and `ChatAnthropic`
- `python-dotenv` - Loads the keys from `.env`

---

### 4. Configure Environment Variables

Copy `.env.example` to `.env`, set `LLM_PROVIDER`, and fill in the variables for that provider. You need only the provider you run.

```bash
cp .env.example .env
```

| `LLM_PROVIDER` | Calls | Variables to set |
| --- | --- | --- |
| `openai` (default) | OpenAI API | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| `anthropic` | Claude API | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `azure-openai` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| `azure-anthropic` | Claude on Microsoft Foundry | `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_FOUNDRY_DEPLOYMENT` |

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-opus-5-5
```

> The variable names and provider values match the course's repo-root `.env` (`labs/setup/.env.example`). Without a local `.env`, `load_dotenv()` picks up that file, so the demo runs with the keys you set for the labs. On Azure, the model is the **deployment name** you created, not the base model name.
>
> Demos 1 and 2 call only Azure, so there `LLM_PROVIDER=openai | anthropic` means Azure OpenAI or Claude on Foundry. Here those two values mean the providers' own APIs, and the Azure ones are `azure-openai | azure-anthropic`.

---

## Usage

Run the server using `uv`:

```bash
uv run uvicorn main:app --reload --port 8000
```

or with the FastAPI CLI:

```bash
uv run fastapi dev main.py
```

The server will start at: http://localhost:8000, with the Swagger UI at http://localhost:8000/docs

Switch the model for one run without editing `.env` (a value in the shell wins over `.env`):

```bash
LLM_PROVIDER=azure-anthropic uv run uvicorn main:app --port 8000
```

Ask a question from another terminal:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a list in Python?"}'
```

---

## Example Output

On startup the server logs the model it built:

```bash
INFO - Provider: anthropic | model: claude-opus-5-5
```

On Azure the line also shows the base URL, for example `https://<resource>.services.ai.azure.com/anthropic`.

### `/health` Response (200 OK):

```json
{
  "status": "healthy",
  "service": "llm-query-api",
  "version": "1.0.0",
  "provider": "anthropic",
  "model": "claude-opus-5-5"
}
```

### `/query` Success Response (200 OK):

The wording varies from run to run and from model to model.

```json
{
  "provider": "anthropic",
  "model": "claude-opus-5-5",
  "answer": "A list in Python is an ordered, changeable collection of items written in square brackets ... Example: fruits = [\"apple\", \"banana\", \"cherry\"]"
}
```

### Error Responses:

| Status | When | `detail` |
| --- | --- | --- |
| 422 | Missing or empty `prompt` | Pydantic's validation error |
| 502 | The provider refused the call: 401 wrong key, 404 wrong model or deployment, 429 rate limit | `LLM provider error 401: ...` |
| 503 | The endpoint or resource name is wrong, or the network is down | `Could not reach the LLM provider: ...` |

> The 502 `detail` passes the provider's message through, which helps while you learn. In production, log it and return a generic message instead.

---

## How the Provider Switch Works

| `LLM_PROVIDER` | `init_chat_model(..., model_provider=)` | `base_url` | `model=` |
| --- | --- | --- | --- |
| `openai` | `"openai"` → `ChatOpenAI` | OpenAI's default | `OPENAI_MODEL` |
| `anthropic` | `"anthropic"` → `ChatAnthropic` | Anthropic's default | `ANTHROPIC_MODEL` |
| `azure-openai` | `"openai"` → `ChatOpenAI` | `<endpoint>/openai/v1/` (Azure OpenAI v1 API) | your Azure OpenAI deployment |
| `azure-anthropic` | `"anthropic"` → `ChatAnthropic` | `https://<resource>.services.ai.azure.com/anthropic` | your Foundry deployment |

Everything after `build_model()` (the prompt, the parser, the chain and both endpoints) is the same for all four.

> `anthropic` is held below 2.0 in `pyproject.toml`: 1.x sends the key as `x-api-key`, the header the Foundry endpoint reads. Claude also gets `max_tokens=4096`; otherwise LangChain asks for the model's maximum.

---

## Console App vs REST API

| | Console app | REST API (this demo) |
| --- | --- | --- |
| Input | `input()` in a loop | `POST /query` with a JSON body, validated by Pydantic |
| Output | `print()` | A JSON `AnswerResponse` |
| Errors | A traceback ends the program | HTTP status codes; the server keeps running |
| Concurrency | One user | Many clients; `ainvoke()` awaits the model without blocking |
| Clients | The terminal | Browsers, mobile apps, `curl`, the Streamlit app in Demo 12 |

---

## Key Concepts

| Step | Concept            | Description                          |
| ---- | ------------------ | ------------------------------------ |
| 1    | Environment Setup  | Load the provider settings from `.env` with `python-dotenv` |
| 2    | Model Initialization | `init_chat_model()` builds `ChatOpenAI` or `ChatAnthropic` for `LLM_PROVIDER` |
| 3    | LCEL Chain         | `prompt \| model \| StrOutputParser()`; the parser returns plain text on every provider |
| 4    | FastAPI Setup      | Creating web application instance    |
| 5    | CORS Middleware    | Enabling cross-origin requests       |
| 6    | Pydantic Models    | Defining request/response schemas    |
| 7    | Health Check       | Adding monitoring endpoint           |
| 8    | Async POST Endpoint | `await chain.ainvoke()`; provider errors mapped to 502 and 503 |
| 9    | Auto Documentation | Using FastAPI's built-in Swagger UI  |

---

## Summary

This project demonstrates how to convert a **console-based LLM application into a REST API** using **FastAPI** and **LangChain**.
One `build_model()` function covers **OpenAI**, **Anthropic Claude**, **Azure OpenAI** and **Claude on Microsoft Foundry**, so the chain and the endpoints stay the same whichever API key you use.
It shows the transformation from terminal input/output to HTTP endpoints, making the application **accessible to web, mobile, and any HTTP client**.
