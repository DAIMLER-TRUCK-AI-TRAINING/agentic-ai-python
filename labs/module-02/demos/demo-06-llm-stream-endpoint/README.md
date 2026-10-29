# LLM Streaming Endpoint with Server-Sent Events (SSE)

This project demonstrates how to implement **streaming AI responses** using **FastAPI** and **Server-Sent Events (SSE)** with **multi-provider support**.
It shows how to create a real-time, word-by-word streaming experience similar to ChatGPT's typing effect.

---

## Features

- **Multi-Provider Support:** Works with OpenAI, Gemini, and other compatible APIs
- **Streaming Endpoint:** `/query/stream` for real-time token-by-token responses
- **Non-Streaming Endpoint:** `/query` for comparison
- **Server-Sent Events (SSE):** Standard protocol for server-to-client streaming
- **Better User Experience:** Users see responses immediately as they're generated
- **CORS Support:** Enable browser-based clients
- **Error Handling:** Graceful streaming error messages
- **Auto-Generated Documentation:** Interactive Swagger UI at `/docs`

---

## Project Structure

```bash
demo-08-llm-stream-endpoint/
├── main.py                      # FastAPI application with streaming
├── .env                        # Environment variables
├── .gitignore                  # Ignore sensitive/config files
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Lock file
└── README.md                   # Documentation
```

---

## Setup

### 1. Navigate to Project Directory

```bash
cd demo-08-llm-stream-endpoint
```

---

### 2. Install Dependencies

```bash
uv sync
```

This will automatically:

- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Set up the project environment

---

### 3. Configure Environment Variables

Create or update the `.env` file with your preferred provider:

**For OpenAI:**

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

**For Gemini:**

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

**Note**:

- The `LLM_PROVIDER` variable determines which configuration is used
- Model names may change over time; refer to your provider's latest documentation

## Usage

Run the application using `uv`:

```bash
uv run uvicorn main:app --reload --port 8000
```

The server will start at: http://localhost:8000

---

## Example Output

### Request:

```json
POST /query/stream
{
  "prompt": "What is AI?"
}
```

### Response (SSE):

```
data: Artificial

data:  Intelligence

data:  (

data: AI

data: )

data:  is

data:  the

data:  simulation

data:  of

data:  human

data:  intelligence

data:  processes

data:  by

data:  machines

data: .


```

---

## Key Concepts

| Step | Concept               | Description                      |
| ---- | --------------------- | -------------------------------- |
| 1    | Initialize Client     | Same as non-streaming setup      |
| 2    | Enable Streaming      | Add `stream=True` to API call    |
| 3    | Receive Stream        | Response returned as a generator |
| 4    | Iterate Chunks        | Loop through each streamed token |
| 5    | Display Progressively | Send chunks using SSE format     |

---

## Summary

This project demonstrates how to implement **streaming AI responses** using **FastAPI**, **Server-Sent Events (SSE)**, and **Gemini API**.
It improves **user experience** by showing results in real time, creating a **ChatGPT-style typing effect** while maintaining the same total processing time.
Perfect for building **real-time AI applications** and **interactive chat interfaces**.

curl -X POST http://127.0.0.1:8000/query/stream \
 -H "Content-Type: application/json" \
 -d '{"prompt": "What is the generator in python?"}'
