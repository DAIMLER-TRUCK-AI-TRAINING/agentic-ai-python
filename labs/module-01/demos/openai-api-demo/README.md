# OpenAI API Demo

A simple Python demo that calls the OpenAI API to generate a response using the Responses API.

## Prerequisites

- Python 3.8 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

---

## Setup Instructions

### Step 1 — Clone or navigate to the project directory

**Mac / Windows (PowerShell):**
```bash
cd labs/module-04/openai-api-demo
```

---

### Step 2 — Create a virtual environment

**Mac (Terminal):**
```bash
python3 -m venv .venv
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

---

### Step 3 — Activate the virtual environment

**Mac (Terminal):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

> If you get a script execution error on Windows, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 4 — Install dependencies

**Mac / Windows:**
```bash
pip install openai python-dotenv
```

---

### Step 5 — Configure environment variables

Copy the example env file and fill in your values:

**Mac (Terminal):**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

Then open `.env` in any text editor and update the values:

```
OPENAI_API_KEY=sk-...your-api-key-here...
MODEL_NAME=gpt-4o
```

> Common model names: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

---

### Step 6 — Run the demo

**Mac / Windows:**
```bash
python 01_openai_api.py
```

You should see a response about the latest AI trends printed to the terminal.

---

## Project Structure

```
openai-api-demo/
├── .env              # Your local environment variables (not committed to git)
├── .env.example      # Template for environment variables
├── .venv/            # Virtual environment (not committed to git)
└── 01_openai_api.py  # Main demo script
```

---

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```
