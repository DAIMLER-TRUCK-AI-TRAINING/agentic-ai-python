# Claude Messages API with an Anthropic API Key

Simple Python application that demonstrates the **Claude Messages API** using the **Anthropic SDK**, an **Anthropic API key** and `uv` environment management. It makes three calls: a single message, a multi-turn follow-up and a streamed answer. [Demo 7](../demo-07-claude-via-azure-foundry/README.md) runs exactly the same calls on **Microsoft Foundry** with an Azure key.

---

## Features

- **Messages API Call**: `client.messages.create()` with a system prompt, a user message and `max_tokens`
- **Content Blocks**: Reads the answer from the `text` blocks of `response.content` (a thinking block comes first)
- **Multi-turn Conversation**: Sends the history back, because the API is stateless
- **Streaming**: `client.messages.stream()` prints text as it is generated
- **Environment Management**: Securely loads the API key from `.env` using `python-dotenv`
- **Error Handling**: Clear messages for a missing key and for API errors (401, 404, 429)
- **Token Usage Reporting**: Prints the stop reason, input tokens and output tokens of every call

---

## Project Structure

```bash
demo-06-claude-via-anthropic-api/
├── main.py                      # Main Python script
├── .env.example                 # Template for environment variables
├── .env                         # Your environment variables (copy of .env.example)
├── .gitignore                   # Ignore environment/config files
├── pyproject.toml               # Project and dependencies
├── uv.lock                      # Locked versions (anthropic 2.3.0)
└── README.md                    # Project documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-06-claude-via-anthropic-api
cd demo-06-claude-via-anthropic-api
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
uv add anthropic python-dotenv
```

or

```bash
uv sync
```

---

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and add your key from [console.anthropic.com](https://console.anthropic.com) (**API keys**).

```bash
cp .env.example .env
```

| Variable | Required | Value |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | yes | Your Claude API key |
| `MODEL_NAME` | no | A Claude model ID: `claude-opus-5-5` (default), `claude-sonnet-5-5`, or `claude-haiku-5-5` (cheapest) |

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MODEL_NAME=claude-opus-5-5
```

> `load_dotenv()` uses the nearest `.env`: this folder first, then the parent folders. Without a local `.env`, the demo picks up the repo-root `.env` from the labs, which already has `ANTHROPIC_API_KEY` (`MODEL_NAME` then defaults to `claude-opus-5-5`).

---

## Usage

Run the script using `uv`:

```bash
uv run main.py
```

One run makes three short requests. On `claude-opus-5-5` that came to about 260 input and 290 output tokens, under one cent.

---

## Example Output

From one run on `claude-opus-5-5`; your text and token counts will differ.

```bash
INFO - Provider: anthropic | base URL: https://api.anthropic.com | model: claude-opus-5-5
INFO - --- 1. Single call ---
INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO - Content blocks: ['thinking', 'text']
INFO - Python programming is writing software in Python, a high-level, interpreted language created by Guido van Rossum and first released in 1991, known for its clean, readable syntax. It supports multiple paradigms, including procedural, object-oriented, and functional programming, and has a large standard library and ecosystem of third-party packages. Python is widely used in web development, data science, machine learning, automation, scripting, and education.
INFO - Stop reason: end_turn | Input tokens: 33, Output tokens: 141
INFO - --- 2. Multi-turn ---
INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO - ```python
print("Hello, World!")
```

This line uses Python's built-in `print()` function to display the text `Hello, World!` on the screen.
INFO - Stop reason: end_turn | Input tokens: 187, Output tokens: 49
INFO - --- 3. Streaming ---
INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
Three popular Python web frameworks are **Django**, a full-featured "batteries-included" framework; **Flask**, a lightweight and flexible microframework; and **FastAPI**, a modern, high-performance framework for building APIs with type hints.
INFO - Stop reason: end_turn | Input tokens: 40, Output tokens: 97
```

The multi-turn call has 187 input tokens against 33 for the first: the whole history, including the first answer, is sent again.

---

## Claude Messages API vs OpenAI Chat Completions (Demo 5)

| | Claude Messages API | OpenAI Chat Completions |
| --- | --- | --- |
| Call | `client.messages.create()` | `client.chat.completions.create()` |
| System prompt | top-level `system=` parameter | a message with `"role": "system"` |
| Output limit | `max_tokens` is **required** | optional |
| Answer | `response.content`: a list of blocks (`thinking`, `text`, `tool_use`, …) | `response.choices[0].message.content`: a string |
| Why it stopped | `response.stop_reason` (`end_turn`, `max_tokens`, `refusal`, …) | `response.choices[0].finish_reason` |
| Usage | `usage.input_tokens`, `usage.output_tokens` | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens` |
| Streaming | `client.messages.stream()` → `stream.text_stream` | `stream=True` → iterate the chunks |

---

## Key Concepts

| Step | Concept               | Description                                 |
| ---- | --------------------- | ------------------------------------------- |
| 1    | Environment Setup     | Load the key securely using `python-dotenv` |
| 2    | Client Initialization | `Anthropic(api_key=...)` targets `https://api.anthropic.com` |
| 3    | Single Call           | `system`, `messages` and `max_tokens`; read the `text` blocks, the stop reason and the usage |
| 4    | Multi-turn            | Resend the history; pass the previous `response.content` back unchanged, thinking blocks included |
| 5    | Streaming             | `messages.stream()` yields text as it arrives; `get_final_message()` returns the full `Message` |
| 6    | Error Handling        | `APIStatusError` carries the HTTP status: 401 wrong key, 404 unknown model, 429 rate limit |

> Current Claude models think before they answer, so the first block is usually `thinking`, not `text`. Read text by block type rather than `response.content[0].text`. Claude Opus 5.5 always thinks; `output_config={"effort": "low"}` makes it think less. The temperature parameter is not available: on the current API version a request that sends `temperature` returns a 400 error.

---
