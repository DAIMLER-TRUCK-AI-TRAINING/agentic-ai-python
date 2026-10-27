# Building a Privacy-First Document Analyzer 

A Python application that demonstrates how to **analyze sensitive, local documents entirely offline** using the **Llama 3.1 8B** model via **Ollama**.

---

## Features

* **100% Offline Processing:** All analysis happens locally; **no data ever leaves your machine**.
* **Privacy Compliant:** Ideal for highly regulated data (e.g., HIPAA, GDPR, internal corporate reports).
* **Multi-Analysis:** Supports generating summaries, extracting key points, and determining sentiment.
* **OpenAI SDK Integration:** Uses the standard `openai` library to interface with Ollama.

---

## Conceptual Flow

1.  **Load Document:** Read confidential text from the local filesystem.
2.  **Process Locally:** Send the content to the running **Ollama** service (no internet required).
3.  **Extract Insights:** **Llama 3.1 8B** generates the requested analysis (summary/key points).
4.  **Cleanup:** Output is printed to the terminal, and the temporary file is deleted.

---

## Project Structure

```bash

demo-4-private-document-analyzer
├── main.py                       # Main script
├── .gitignore                    # Ignore sensitive/config files
└── README.md                     # Documentation

```

---

## Setup

### 1. Ensure Ollama is Ready

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

### 2. Create and Initialize Project

```bash
uv init demo-4-private-document-analyzer
cd demo-4-private-document-analyzer
```

---

### 3. Create Virtual Environment

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

### 4. Install Dependencies

```bash
uv add openai
```

---

## Usage

Run the analyzer script:

```bash
uv run main.py
```

---

## Example Output

```bash
============================================================
PRIVACY-FIRST DOCUMENT ANALYZER
============================================================
Processing locally using Ollama and Llama 3.1 8B

Analyzing 'sample_doc.txt' for summary...

SUMMARY:
------------------------------------------------------------
The report confirms workplace compliance with safety norms but recommends ergonomic improvements. The overall conclusion is low risk, provided that standing desks and monitor adjustments are implemented.

Analyzing 'sample_doc.txt' for key_points...

KEY_POINTS:
------------------------------------------------------------
• Workplace complies with current safety norms.
• Ergonomic improvements are highly recommended.
• Implement standing desks for employees.
• Monitor height adjustments are advised.
• The overall risk level is assessed as low.

Analyzing 'confidential.txt' for sentiment...

SENTIMENT:
------------------------------------------------------------
The document maintains a professional, objective, and moderately positive tone. It is fact-based with constructive recommendations for improvement.

✓ Analysis complete — all processing was local.
```

---

## Summary

This example successfully demonstrates how to build a **privacy-preserving document analyzer** using **Ollama** and **Llama 3.1 8B**. By keeping all computation and data exchange **strictly local**, you achieve complete data confidentiality, making this solution viable for sensitive and regulated industries.

