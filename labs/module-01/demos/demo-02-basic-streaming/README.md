# Basic Streaming Implementation using OpenAI-Compatible SDK

This application demonstrates how to **convert a standard LLM API call into a streaming call** for a better user experience.
It is based on *Example 1: Basic Streaming Implementation* from the Sprint Master document and is implemented using **uv** for environment management.

---

## Features

* **Streaming Response:** Enables `stream=True` for real-time token-by-token output
* **Progressive Display:** Prints model output as it’s generated
* **Same Setup as Non-Streaming:** Reuses existing API initialization
* **Better UX:** Immediate feedback without waiting for full completion
* **Compatible with Older OpenAI SDKs:** Works even when `.stream()` helper is not available

---

## Project Structure

```bash
demo-2-basic-streaming/
├── main.py                      # Streaming implementation
├── .env                        # Environment variables
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
| 1    | Initialize Client     | Same as non-streaming setup             |
| 2    | Enable Streaming      | Add `stream=True` to API call           |
| 3    | Receive Stream        | Response returned as a generator        |
| 4    | Iterate Chunks        | Loop through each streamed token        |
| 5    | Display Progressively | Print chunks using `sys.stdout.flush()` |

---


