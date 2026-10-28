# LangChain Classic on Azure: OpenAI and Claude Models

Simple Python application that demonstrates **LangChain classic**, an **LCEL chain** and the classic tool-calling agent (`AgentExecutor`, from the `langchain-classic` package), on a model hosted in **Azure**. The model is either **Azure OpenAI** or **Claude on Microsoft Foundry**, both called with an **Azure API key**, and `LLM_PROVIDER` switches between them without changing the chain or the agent. [Demo 2](../demo-02-langchain-agent-azure/README.md) runs the same task on the new LangChain agent, `create_agent()`.

---

## Features

- **One Model Factory**: `init_chat_model()` returns a `ChatOpenAI` or a `ChatAnthropic` pointed at your Azure resource
- **Switchable Provider**: `LLM_PROVIDER=openai | anthropic` picks the model; the chain and the agent never change
- **LCEL Chain**: `prompt | model | StrOutputParser()` in one runnable
- **Classic Agent**: `create_tool_calling_agent()` + `AgentExecutor` with two service desk tools (`get_incident`, `search_kb`)
- **Tool Trace**: Prints every tool call and its result from `intermediate_steps`
- **Environment Management**: Securely loads the Azure keys from `.env` using `python-dotenv`
- **Error Handling**: Clear messages for missing settings, API errors (401, 404, 429) and an unreachable endpoint

---

## Project Structure

```bash
demo-01-langchain-classic-azure/
├── main.py                      # Main Python script
├── .env.example                 # Template for environment variables
├── .env                         # Your environment variables (copy of .env.example)
├── .gitignore                   # Ignore environment/config files
├── pyproject.toml               # Project and dependencies
├── uv.lock                      # Locked versions
└── README.md                    # Project documentation
```

---

## Setup

### 0. Deploy the Models in Azure

You need only the provider you run:

| `LLM_PROVIDER` | In Azure | Values to note |
| --- | --- | --- |
| `openai` | An **Azure OpenAI** resource with a model deployment | Endpoint `https://<resource>.openai.azure.com`, key, deployment name |
| `anthropic` | A **Microsoft Foundry** resource with a Claude deployment (for example `claude-opus-5-5`) | Resource name from `https://<resource>.services.ai.azure.com`, key, deployment name |

---

### 1. Create and Initialize Project

```bash
uv init demo-01-langchain-classic-azure
cd demo-01-langchain-classic-azure
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
uv add langchain langchain-classic langchain-openai langchain-anthropic "anthropic<2" python-dotenv
```

or

```bash
uv sync
```

---

### 4. Configure Environment Variables

Copy `.env.example` to `.env`, set `LLM_PROVIDER`, and fill in the variables for that provider.

```bash
cp .env.example .env
```

| `LLM_PROVIDER` | Calls | Variables to set |
| --- | --- | --- |
| `openai` (default) | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| `anthropic` | Claude on Microsoft Foundry | `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_FOUNDRY_DEPLOYMENT` |

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_FOUNDRY_API_KEY=your_azure_foundry_key_here
ANTHROPIC_FOUNDRY_RESOURCE=your-resource-name
ANTHROPIC_FOUNDRY_DEPLOYMENT=claude-opus-5-5
```

> On Azure, `model` is the **deployment name** you created, not the base model name. Without a local `.env`, `load_dotenv()` picks up the repo-root `.env` from the labs, which has no Azure settings, so the demo stops with `... not found in environment variables!`.

---

## Usage

Run the script using `uv`:

```bash
uv run main.py
```

Switch the model for one run without editing `.env` (a value in the shell wins over `.env`):

```bash
LLM_PROVIDER=anthropic uv run main.py
```

---

## Example Output

The shape of a run on Claude. The tool results come from the demo's own data, but the wording, the model's search query and the number of agent loops vary from run to run.

```bash
INFO - Provider: anthropic on Azure | base URL: https://your-resource-name.services.ai.azure.com/anthropic | deployment: claude-opus-5-5
INFO - --- 1. LCEL chain ---
INFO - HTTP Request: POST https://your-resource-name.services.ai.azure.com/anthropic/v1/messages "HTTP/1.1 200 OK"
INFO - <a two-sentence answer about VPNs and cached passwords>
INFO - --- 2. Classic agent (AgentExecutor) ---
INFO - HTTP Request: POST https://your-resource-name.services.ai.azure.com/anthropic/v1/messages "HTTP/1.1 200 OK"
INFO - HTTP Request: ...                                   (one request per agent loop)
INFO - Tool call: get_incident({'incident_id': 'INC-104301'}) -> {'status': 'In progress', 'priority': 'P3', 'team': 'Network Ops', 'summary': 'VPN keeps rejecting credentials after a password change'}
INFO - Tool call: search_kb({'query': '<the model's query>'}) -> [{'kb_id': 'KB-0042', 'text': 'VPN client rejects credentials after a password change. ...'}]
INFO - Answer: <names INC-104301, its status and KB-0042>
```

With `LLM_PROVIDER=openai` the first line shows `https://<resource>.openai.azure.com/openai/v1/`, and the requests go to `/openai/v1/chat/completions` (or `/openai/v1/responses`, see below).

---

## How the Provider Switch Works

| | `openai` | `anthropic` |
| --- | --- | --- |
| `init_chat_model(..., model_provider=)` | `"openai"` → `ChatOpenAI` | `"anthropic"` → `ChatAnthropic` |
| `base_url` | `<endpoint>/openai/v1/` (Azure OpenAI v1 API) | `https://<resource>.services.ai.azure.com/anthropic` |
| Key sent as | `Authorization: Bearer <key>` | `x-api-key: <key>` |
| `model=` | your Azure OpenAI deployment | your Foundry deployment |
| API used | Chat Completions; LangChain switches to the Responses API by itself for a `gpt-6…` deployment with tools | Claude Messages API |

> `anthropic` is held below 2.0 in `pyproject.toml`: 1.x sends the key as `x-api-key`, the header the Foundry endpoint reads, while 2.x sends `Authorization: Bearer`. Claude also gets `max_tokens=4096`; otherwise LangChain asks for the model's maximum.

---

## LangChain Classic (Demo 1) vs the New Agent (Demo 2)

| | Classic: `AgentExecutor` | New: `create_agent()` |
| --- | --- | --- |
| Package | `langchain-classic`: the pre-1.0 APIs, kept so existing code runs | `langchain` 1.x, running on LangGraph |
| Build | `create_tool_calling_agent(model, tools, prompt)` + `AgentExecutor(agent=, tools=)` | `create_agent(model, tools=, system_prompt=)` |
| Prompt | A `ChatPromptTemplate` with `{input}` and an `agent_scratchpad` placeholder | `system_prompt=`; the conversation lives in the `messages` state |
| Input → output | `{"input": ...}` → `{"output": ...}`: the raw content, a list of blocks on Claude | `{"messages": [...]}` → `{"messages": [...]}`; `.text` on the last one |
| Loop limit | `max_iterations=` | middleware such as `ModelCallLimitMiddleware` |
| Memory | none built in (wrap it in `RunnableWithMessageHistory`) | `checkpointer=` + a `thread_id` |
| Streaming | `executor.stream()` yields actions and steps | `agent.stream(stream_mode="updates")` yields each step |

---

## Key Concepts

| Step | Concept               | Description                                 |
| ---- | --------------------- | ------------------------------------------- |
| 1    | Environment Setup     | Load the Azure settings securely using `python-dotenv` |
| 2    | Model Initialization  | `init_chat_model()` builds `ChatOpenAI` or `ChatAnthropic` for your Azure resource via `LLM_PROVIDER` |
| 3    | LCEL Chain            | `prompt \| model \| StrOutputParser()`; the parser keeps only the text on both providers |
| 4    | Tools                 | `@tool` functions; the docstring and type hints become the tool schema the model sees |
| 5    | Classic Agent         | `AgentExecutor` loops model → tool → model until the model answers or `max_iterations` is reached |
| 6    | Error Handling        | `APIStatusError` from either SDK: 401 wrong key, 404 wrong deployment, 429 rate limit |

---
