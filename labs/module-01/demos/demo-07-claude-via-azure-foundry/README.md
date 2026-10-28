# Claude Messages API with an Azure Key (Microsoft Foundry)

Simple Python application that demonstrates the **Claude Messages API** on **Microsoft Foundry**: a Claude model deployed in your Azure resource, called with the **Anthropic SDK**, an **Azure key** and `uv` environment management. The calls are exactly the same as in [Demo 6](../demo-06-claude-via-anthropic-api/README.md); only the client changes.

---

## Features

- **Foundry Client**: `AnthropicFoundry` from the Anthropic SDK, with your Azure resource key and resource name
- **Same Messages API**: `client.messages.create()` and `client.messages.stream()`, unchanged from Demo 6
- **Multi-turn Conversation**: Sends the history back, because the API is stateless
- **Streaming**: Prints text as it is generated
- **Environment Management**: Securely loads the Azure key from `.env` using `python-dotenv`
- **Error Handling**: Clear messages for missing settings and for API errors (401, 404, 429)
- **Token Usage Reporting**: Prints the stop reason, input tokens and output tokens of every call

---

## Project Structure

```bash
demo-07-claude-via-azure-foundry/
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

### 0. Deploy Claude in Microsoft Foundry

In the [Microsoft Foundry portal](https://ai.azure.com), deploy a Claude model (for example `claude-opus-5-5`) from the model catalogue into a Foundry resource. From the deployment's details, note three values:

| Value | Looks like | Goes into |
| --- | --- | --- |
| Endpoint | `https://<resource>.services.ai.azure.com/anthropic/` | `ANTHROPIC_FOUNDRY_RESOURCE`: the `<resource>` part only |
| Key | a long string | `ANTHROPIC_FOUNDRY_API_KEY` |
| Deployment name | often the model ID, for example `claude-opus-5-5` | `ANTHROPIC_FOUNDRY_DEPLOYMENT` |

---

### 1. Create and Initialize Project

```bash
uv init demo-07-claude-via-azure-foundry
cd demo-07-claude-via-azure-foundry
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

Copy `.env.example` to `.env` and fill in the three values from step 0.

```bash
cp .env.example .env
```

```bash
ANTHROPIC_FOUNDRY_API_KEY=your_azure_foundry_key_here
ANTHROPIC_FOUNDRY_RESOURCE=your-resource-name
ANTHROPIC_FOUNDRY_DEPLOYMENT=your_deployment_name_here
```

> `ANTHROPIC_FOUNDRY_API_KEY` and `ANTHROPIC_FOUNDRY_RESOURCE` are the names the SDK itself reads, so `AnthropicFoundry()` with no arguments also works. On Foundry, `model` is the **deployment name**, not necessarily the model ID.

---

## Usage

Run the script using `uv`:

```bash
uv run main.py
```

---

## Example Output

The output has the same shape as [Demo 6's](../demo-06-claude-via-anthropic-api/README.md#example-output). Only the first line and the request URLs change:

```bash
INFO - Provider: azure foundry | base URL: https://your-resource-name.services.ai.azure.com/anthropic/ | model: claude-opus-5-5
INFO - --- 1. Single call ---
INFO - HTTP Request: POST https://your-resource-name.services.ai.azure.com/anthropic/v1/messages "HTTP/1.1 200 OK"
INFO - Content blocks: ['thinking', 'text']
INFO - <the answer>
INFO - Stop reason: end_turn | Input tokens: <n>, Output tokens: <n>
...
```

A wrong deployment name gives `ERROR - Foundry API error 404: ...`, a wrong key gives a 401, and a misspelt resource name gives `ERROR - Could not reach https://<resource>...`.

---

## Claude API (Demo 6) vs Microsoft Foundry (Demo 7)

| | Demo 6: Claude API | Demo 7: Microsoft Foundry |
| --- | --- | --- |
| Client | `Anthropic(api_key=...)` | `AnthropicFoundry(api_key=..., resource=...)` |
| Key | Anthropic API key (console.anthropic.com) | Azure resource key |
| Endpoint | `https://api.anthropic.com` | `https://<resource>.services.ai.azure.com/anthropic/` |
| `model=` | a Claude model ID | your deployment name |
| Billing | your Anthropic account | your Azure subscription |
| Messages, streaming, tool use, thinking | ✅ | ✅ same code |
| Models API, Message Batches | ✅ | not available: the SDK removes `client.models` and `client.messages.batches` |

> **Keyless option:** instead of a key, `AnthropicFoundry` accepts a Microsoft Entra ID token provider (`pip install azure-identity`):
>
> ```python
> from azure.identity import DefaultAzureCredential, get_bearer_token_provider
>
> token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
> client = AnthropicFoundry(azure_ad_token_provider=token_provider, resource="your-resource-name")
> ```

---

## Key Concepts

| Step | Concept               | Description                                 |
| ---- | --------------------- | ------------------------------------------- |
| 1    | Environment Setup     | Load the Azure key, resource and deployment securely using `python-dotenv` |
| 2    | Client Initialization | `AnthropicFoundry` builds the endpoint from the resource name |
| 3    | Single Call           | `system`, `messages` and `max_tokens`; read the `text` blocks, the stop reason and the usage |
| 4    | Multi-turn            | Resend the history; pass the previous `response.content` back unchanged, thinking blocks included |
| 5    | Streaming             | `messages.stream()` yields text as it arrives; `get_final_message()` returns the full `Message` |
| 6    | Error Handling        | `APIStatusError` carries the HTTP status: 401 wrong key, 404 wrong deployment, 429 rate limit; `APIConnectionError` for a misspelt resource name |

---
