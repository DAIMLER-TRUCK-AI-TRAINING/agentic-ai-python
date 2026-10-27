# Run Your First Local LLM with Ollama

This application consists of a  simple **Python script** that connects to a **local Ollama server** to query the **Llama 3.1 8B** model using the **OpenAI-compatible SDK**.

---

## Features

* **Local Inference:** Model runs entirely on your local machine
* **OpenAI SDK Compatibility:** Use the familiar `openai` library with Ollama's local endpoint
* **Model Verification:** Script checks for available local models
* **Privacy-First:** No external internet connectivity required after the initial model download

---

## Project Structure

```bash

demo-3-test-local-model/
├── main.py                  # Main script to connect and query
├── .gitignore               # Ignore system/config files
└── README.md                # Documentation

```

---

## Setup

### 1. Ensure Ollama is Running

Make sure **Ollama** is installed and running on your system.

**Linux:**

```bash
ollama serve
```

**macOS:**

On macOS, Ollama typically runs as a background service. Start it with:

```bash
ollama serve
```

**Windows:**

On Windows, Ollama runs as a background service. 

Alternatively, if Ollama is installed as a desktop application, you can start it from the Start menu and it will run in the background. Then use Command Prompt or PowerShell to interact with it.

Check the ollama model (works on all platforms):

```bash
ollama list
```

### 2. Pull the Model

Open your terminal and pull the required model:

```bash
ollama pull llama3.1:8b
```

For lesser memory space, pull the below model​

```bash
ollama pull llama3.2:3b
```
---

### 3. Create and Initialize Project

```bash
uv init demo-3-test-local-model
cd demo-3-test-local-model
```

---

### 4. Create Virtual Environment

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

### 5. Install Dependencies

```bash
uv add openai
```

---

## Usage

Run the test script using:

```bash
uv run main.py
```

---

## Example Output

```bash
Available models:
 - llama3.1:8b

Testing Llama 3.1 8B:

Reinforcement learning is when an AI learns by trial and error.
It takes actions, receives rewards or penalties, and gradually improves to achieve goals.
```

---

## Key Implementation Points

| Step | Concept          | Description                                                    |
| :--- | :--------------- | :------------------------------------------------------------- |
| 1    | Client Setup     | Point the `openai.OpenAI` client to Ollama's local URL         |
| 2    | Model Check      | Use `client.models.list()` to verify Ollama's local catalog    |
| 3    | Basic Generation | Use `client.chat.completions.create()` for a simple query      |
| 4    | Local Speed      | Note the fast response time from the model running in memory   |

---


