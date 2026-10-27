# Basic OpenAI API Call with OpenAI SDK

Simple Python application that demonstrates how to make a **basic API call** to **OpenAI** using the **OpenAI SDK** and `uv` environment management.

---

## Features

- **OpenAI API Call**: Uses the OpenAI SDK to call OpenAI models
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
├── .env                         # Environment variables
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

Change the filename from `.envbackup` to `.env`, and include the appropriate keys within it.

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
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
| 2    | Client Initialization | Configure OpenAI SDK with API key           |
| 3    | Message Construction  | Build `system` and `user` message structure |
| 4    | API Invocation        | Use `client.chat.completions.create()`      |
| 5    | Response Parsing      | Extract text and usage details              |
| 6    | Error Handling        | Manage runtime and auth issues gracefully   |

---
