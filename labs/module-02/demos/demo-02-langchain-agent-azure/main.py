"""
Demo 2: The new LangChain agent on Azure, with an OpenAI or an Anthropic model
Runs an agent built with create_agent() (LangChain 1.x, on LangGraph) on a model
hosted in Azure: Azure OpenAI, or Claude on Microsoft Foundry. LLM_PROVIDER in
.env picks the model (openai | anthropic); the agent code never changes.
Demo 1 runs the same task on the classic AgentExecutor.
"""

import os
import re
import logging
import anthropic
import openai
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def require_env(name):
    """Return an environment variable or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables!")
    return value


def build_model():
    """
    Build a LangChain chat model for the provider in LLM_PROVIDER, hosted in Azure.
    init_chat_model returns a ChatOpenAI or a ChatAnthropic; only the deployment,
    endpoint and key differ, so everything built on the model stays the same.
    """
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        # Azure OpenAI v1 API: the regular OpenAI client, base_url points at your Azure resource.
        # "model" is the deployment name you created in Azure, not the base model name.
        base_url = f"{require_env('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"
        deployment = require_env("AZURE_OPENAI_DEPLOYMENT")
        model = init_chat_model(
            deployment,
            model_provider="openai",
            base_url=base_url,
            api_key=require_env("AZURE_OPENAI_API_KEY")
        )
    elif provider == "anthropic":
        # Claude on Microsoft Foundry: the Anthropic Messages API under /anthropic on your resource.
        # "model" is the deployment name you created in Foundry.
        base_url = f"https://{require_env('ANTHROPIC_FOUNDRY_RESOURCE')}.services.ai.azure.com/anthropic"
        deployment = require_env("ANTHROPIC_FOUNDRY_DEPLOYMENT")
        model = init_chat_model(
            deployment,
            model_provider="anthropic",
            base_url=base_url,
            api_key=require_env("ANTHROPIC_FOUNDRY_API_KEY"),
            max_tokens=4096  # otherwise LangChain asks for the model's maximum
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Use 'openai' or 'anthropic'.")

    logger.info(f"Provider: {provider} on Azure | base URL: {base_url} | deployment: {deployment}")
    return model


# A tiny in-memory service desk, so the demo needs no database
INCIDENTS = {
    "INC-104301": {"status": "In progress", "priority": "P3", "team": "Network Ops",
                   "summary": "VPN keeps rejecting credentials after a password change"},
    "INC-104318": {"status": "New", "priority": "P2", "team": "Service Desk",
                   "summary": "Outgoing email stuck in the outbox since this morning"},
}
KB_ARTICLES = {
    "KB-0042": "VPN client rejects credentials after a password change. "
               "Wait for directory sync, clear the saved credential and reconnect.",
    "KB-0107": "Account locked after repeated failed sign-ins. "
               "Use the self-service portal to unlock it or ask the Service Desk to verify your identity.",
    "KB-0521": "Outgoing email stuck in the outbox. "
               "Free up mailbox space, archive old items and check the connection.",
}


@tool
def get_incident(incident_id: str) -> dict:
    """Look up a service desk incident by ID, for example INC-104301: status, priority, team and summary."""
    return INCIDENTS.get(incident_id.strip().upper(), {"error": f"{incident_id} not found"})


@tool
def search_kb(query: str) -> list[dict]:
    """Search the internal knowledge base. Returns up to two matching articles with their KB IDs."""
    words = {word for word in re.findall(r"[a-z]+", query.lower()) if len(word) > 3}
    scored = sorted(((len(words & set(re.findall(r"[a-z]+", text.lower()))), kb_id, text)
                     for kb_id, text in KB_ARTICLES.items()), reverse=True)
    hits = [{"kb_id": kb_id, "text": text} for score, kb_id, text in scored if score]
    return hits[:2] or [{"error": "no matching article"}]


TOOLS = [get_incident, search_kb]
SYSTEM_PROMPT = (
    "You are an IT service desk assistant. Use get_incident to look up incidents and search_kb "
    "to find knowledge articles. Name the incident and KB IDs you relied on. "
    "Answer in at most three sentences."
)
QUESTION = "What is the status of INC-104301, and which KB article should the engineer follow to fix it?"
FOLLOW_UP = "Which team is working on it, and what should I tell the user in the meantime?"


def log_message(message):
    """One line per message: the model's tool calls, a tool's result, or the answer."""
    if message.type == "ai" and message.tool_calls:
        for call in message.tool_calls:
            logger.info(f"Tool call: {call['name']}({call['args']})")
    elif message.type == "tool":
        logger.info(f"Tool result ({message.name}): {message.content}")
    elif message.type == "ai":
        # .text joins the text blocks, so it is a plain string on both providers
        logger.info(f"Answer: {message.text}")


def streamed_run(agent, config):
    # stream_mode="updates" yields each step (a model call or a tool run) as soon as it finishes
    inputs = {"messages": [{"role": "user", "content": QUESTION}]}
    for update in agent.stream(inputs, config, stream_mode="updates"):
        for step in update.values():
            for message in (step or {}).get("messages", []):
                log_message(message)


def follow_up(agent, config):
    # The checkpointer kept the first exchange under the same thread_id,
    # so the agent knows that "it" is INC-104301
    result = agent.invoke({"messages": [{"role": "user", "content": FOLLOW_UP}]}, config)
    log_message(result["messages"][-1])
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    input_tokens = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai_messages)
    output_tokens = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai_messages)
    logger.info(f"Thread: {len(result['messages'])} messages, {len(ai_messages)} model calls | "
                f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")


def main():
    # 1. Environment setup: load provider settings from .env file
    load_dotenv()
    # 2. Model initialization: one LangChain chat model, Azure OpenAI or Claude on Foundry
    model = build_model()
    # 3. The agent: model, tools and system prompt; the checkpointer gives it memory
    agent = create_agent(model, tools=TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "demo-02"}}
    try:
        # 4. A tool-calling run, streamed step by step
        logger.info("--- 1. Agent run (create_agent), streamed step by step ---")
        streamed_run(agent, config)
        # 5. A follow-up question on the same thread
        logger.info("--- 2. Follow-up on the same thread ---")
        follow_up(agent, config)
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        # 401: wrong key · 404: wrong deployment name · 429: rate limit
        logger.error(f"Azure API error {error.status_code}: {error.message}")
        raise SystemExit(1)
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        logger.error("Could not reach Azure: check AZURE_OPENAI_ENDPOINT or ANTHROPIC_FOUNDRY_RESOURCE")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
