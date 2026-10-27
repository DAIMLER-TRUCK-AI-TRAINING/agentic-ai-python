# Basic OpenAI API Call with Gemini via OpenAI SDK

Simple Python application that demonstrates how to make a **basic OpenAI-compatible API call** to **Google Gemini** using the **OpenAI SDK** and `uv` environment management.

---

## Features

* **OpenAI-Compatible API Call**: Uses the OpenAI SDK to call Gemini via `base_url`
* **Environment Management**: Securely loads API keys from `.env` using `python-dotenv`
* **Structured Prompts**: Handles both system and user message roles
* **Runtime Configuration**: Customize model, token limit, and temperature
* **Error Handling**: Detects missing keys, invalid responses, and network issues
* **Token Usage Reporting**: Prints prompt, completion, and total token counts

---

## Project Structure

```bash
demo-1-llm-via-openai-sdk/
├── main.py                      # Main Python script
├── .env                         # Environment variables
├── .gitignore                   # Ignore environment/config files
└── README.md                    # Project documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-1-llm-via-openai-sdk
cd demo-1-llm-via-openai-sdk
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

Change the filename from `.envbackup` to `.env`, and include the appropriate keys within it.

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-2.0-flash
```

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

A list comprehension is a compact way to create lists in Python.
Example: [x**2 for x in range(5)] → [0, 1, 4, 9, 16]

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
| 2    | Client Initialization | Configure OpenAI SDK with Gemini `base_url` |
| 3    | Message Construction  | Build `system` and `user` message structure |
| 4    | API Invocation        | Use `client.chat.completions.create()`      |
| 5    | Response Parsing      | Extract text and usage details              |
| 6    | Error Handling        | Manage runtime and auth issues gracefully   |

---


