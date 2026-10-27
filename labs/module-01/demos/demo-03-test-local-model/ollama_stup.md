# Ollama Setup Guide

This guide walks you through installing **Ollama** and setting up your local environment to run **Llama 3.1 8B** on your machine.

---

## Objective

Set up **Ollama** to run LLMs locally, verify installation, and test a model using both the **command line** and **Python**.

---

## Conceptual Flow

1. **Install Ollama:** Download and install based on your operating system  
2. **Download Model:** Pull the Llama 3.1 8B model  
3. **Verify Installation:** Check that Ollama is running correctly  
4. **Test via Command Line:** Run a basic query  
5. **Test via Python:** Use OpenAI SDK to call the local model  

---

## Step-by-Step Installation

### macOS / Linux

```
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service (runs in background)
ollama serve

# In a new terminal, pull the Llama 3.1 8B model
ollama pull llama3.1:8b

# Test the model from the command line
ollama run llama3.1:8b "What is machine learning?"

```

### Windows

```
# Download installer from https://ollama.com/download
# Run OllamaSetup.exe (it will start the Ollama service automatically)

# Open Command Prompt or PowerShell
ollama pull llama3.1:8b

# Test from the command line
ollama run llama3.1:8b "What is machine learning?"

```
### Verification
- Check that Ollama is running:
```
ollama list
You should see:

NAME              ID             SIZE
llama3.1:8b       abc123         4.7GB
```

### Key Observations
- Model download is approximately 4.7 GB (one-time download)

- The first run may be slow while the model loads

- Subsequent runs are faster due to in-memory caching

- No API keys or internet are required after setup


### Summary
- You have successfully installed Ollama, downloaded Llama 3.1 8B, and verified it runs locally.
- You can now integrate it with Python for offline LLM applications.


