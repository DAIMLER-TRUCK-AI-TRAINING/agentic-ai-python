# The New LangChain Agent on Azure: OpenAI and Claude Models

Simple Python application that demonstrates the **new LangChain agent**, `create_agent()` from LangChain 1.x (built on LangGraph), on a model hosted in **Azure**. The model is either **Azure OpenAI** or **Claude on Microsoft Foundry**, both called with an **Azure API key**, and `LLM_PROVIDER` switches between them without changing the agent. [Demo 1](../demo-01-langchain-classic-azure/README.md) runs the same task on the classic `AgentExecutor`.

---

## Features

- **One Model Factory**: `init_chat_model()` returns a `ChatOpenAI` or a `ChatAnthropic` pointed at your Azure resource
- **Switchable Provider**: `LLM_PROVIDER=openai | anthropic` picks the model; the agent never changes
- **New Agent**: `create_agent(model, tools=, system_prompt=)` with two service desk tools (`get_incident`, `search_kb`)
- **Step-by-step Streaming**: `agent.stream(stream_mode="updates")` prints each tool call, tool result and answer as it happens
- **Built-in Memory**: An `InMemorySaver` checkpointer and a `thread_id` let a follow-up question refer to "it"
- **Token Usage Reporting**: Adds up `usage_metadata` over every model call in the thread
- **Environment Management**: Securely loads the Azure keys from `.env` using `python-dotenv`
- **Error Handling**: Clear messages for missing settings, API errors (401, 404, 429) and an unreachable endpoint

---

## Project Structure

```bash
demo-02-langchain-agent-azure/
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
uv init demo-02-langchain-agent-azure
cd demo-02-langchain-agent-azure
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
uv add langchain langgraph langchain-openai langchain-anthropic "anthropic<2" python-dotenv
```

or

```bash
uv sync
```

---

### 4. Configure Environment Variables

Copy `.env.example` to `.env`, set `LLM_PROVIDER`, and fill in the variables for that provider. The file is the same as Demo 1's, so you can copy that one instead.

```bash
cp .env.example .env
```

| `LLM_PROVIDER` | Calls | Variables to set |
| --- | --- | --- |
| `openai` (default) | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| `anthropic` | Claude on Microsoft Foundry | `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_FOUNDRY_DEPLOYMENT` |

```bash
LLM_PROVIDER=openai
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
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

The shape of a run on Azure OpenAI. The tool results come from the demo's own data, but the wording, the model's search query and the number of steps vary from run to run.

```bash
INFO - Provider: openai on Azure | base URL: https://your-resource-name.openai.azure.com/openai/v1/ | deployment: <your deployment>
INFO - --- 1. Agent run (create_agent), streamed step by step ---
INFO - HTTP Request: POST https://your-resource-name.openai.azure.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
INFO - Tool call: get_incident({'incident_id': 'INC-104301'})
INFO - Tool result (get_incident): {"status": "In progress", "priority": "P3", "team": "Network Ops", "summary": "VPN keeps rejecting credentials after a password change"}
INFO - Tool call: search_kb({'query': '<the model's query>'})
INFO - Tool result (search_kb): [{"kb_id": "KB-0042", "text": "VPN client rejects credentials after a password change. ..."}]
INFO - Answer: <names INC-104301, its status and KB-0042>
INFO - --- 2. Follow-up on the same thread ---
INFO - Answer: <Network Ops, and what to tell the user in the meantime>
INFO - Thread: <n> messages, <n> model calls | Input tokens: <n>, Output tokens: <n>
```

The follow-up never names the incident: the checkpointer resends the first exchange, so the model knows what "it" is. With `LLM_PROVIDER=anthropic` the requests go to `https://<resource>.services.ai.azure.com/anthropic/v1/messages`.

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

## The New Agent (Demo 2) vs LangChain Classic (Demo 1)

| | New: `create_agent()` | Classic: `AgentExecutor` |
| --- | --- | --- |
| Package | `langchain` 1.x, running on LangGraph | `langchain-classic`: the pre-1.0 APIs, kept so existing code runs |
| Build | `create_agent(model, tools=, system_prompt=)` | `create_tool_calling_agent(model, tools, prompt)` + `AgentExecutor(agent=, tools=)` |
| Prompt | `system_prompt=`; the conversation lives in the `messages` state | A `ChatPromptTemplate` with `{input}` and an `agent_scratchpad` placeholder |
| Input → output | `{"messages": [...]}` → `{"messages": [...]}`; `.text` on the last one | `{"input": ...}` → `{"output": ...}`: the raw content, a list of blocks on Claude |
| Loop limit | middleware such as `ModelCallLimitMiddleware` | `max_iterations=` |
| Memory | `checkpointer=` + a `thread_id` | none built in (wrap it in `RunnableWithMessageHistory`) |
| Streaming | `agent.stream(stream_mode="updates")` yields each step | `executor.stream()` yields actions and steps |
| Extending it | middleware: summarisation, human-in-the-loop, guardrails | callbacks or a custom agent class |

---

## Key Concepts

| Step | Concept               | Description                                 |
| ---- | --------------------- | ------------------------------------------- |
| 1    | Environment Setup     | Load the Azure settings securely using `python-dotenv` |
| 2    | Model Initialization  | `init_chat_model()` builds `ChatOpenAI` or `ChatAnthropic` for your Azure resource via `LLM_PROVIDER` |
| 3    | Agent                 | `create_agent()` compiles a LangGraph graph: a model node and a tools node in a loop |
| 4    | Streaming             | `stream_mode="updates"` yields one update per finished step: model calls and tool runs |
| 5    | Memory                | `InMemorySaver` stores each thread's messages; the same `thread_id` continues the conversation |
| 6    | Error Handling        | `APIStatusError` from either SDK: 401 wrong key, 404 wrong deployment, 429 rate limit |

---
