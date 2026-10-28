"""
Demo 1: LangChain classic on Azure, with an OpenAI or an Anthropic model
Runs an LCEL chain and the classic tool-calling agent (AgentExecutor, from the
langchain-classic package) on a model hosted in Azure: Azure OpenAI, or Claude on
Microsoft Foundry. LLM_PROVIDER in .env picks the model (openai | anthropic); the
chain and agent code never changes.
Demo 2 runs the same task on the new LangChain agent, create_agent().
"""

import os
import re
import logging
import anthropic
import openai
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

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


def output_text(output):
    """The classic agent returns the model's raw content: a string, or a list of blocks (Claude)."""
    if isinstance(output, str):
        return output
    return "".join(block.get("text", "") for block in output
                   if isinstance(block, dict) and block.get("type") == "text")


def lcel_chain(model):
    # LCEL: a prompt, the model and an output parser piped into one runnable
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an IT service desk assistant. Answer in at most two sentences."),
        ("human", "{question}")
    ])
    chain = prompt | model | StrOutputParser()
    # StrOutputParser keeps only the text, so the answer is a plain string on both providers
    answer = chain.invoke({"question": "Why can a VPN reject credentials right after a password change?"})
    logger.info(answer)


def classic_agent(model):
    # The classic agent prompt needs an agent_scratchpad slot: on every loop,
    # AgentExecutor puts the tool calls so far and their results there
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    agent = create_tool_calling_agent(model, TOOLS, prompt)
    executor = AgentExecutor(agent=agent, tools=TOOLS, max_iterations=5, return_intermediate_steps=True)
    result = executor.invoke({"input": QUESTION})

    for action, observation in result["intermediate_steps"]:
        logger.info(f"Tool call: {action.tool}({action.tool_input}) -> {observation}")
    logger.info(f"Answer: {output_text(result['output'])}")


def main():
    # 1. Environment setup: load provider settings from .env file
    load_dotenv()
    # 2. Model initialization: one LangChain chat model, Azure OpenAI or Claude on Foundry
    model = build_model()
    try:
        # 3. An LCEL chain: prompt | model | parser, one model call
        logger.info("--- 1. LCEL chain ---")
        lcel_chain(model)
        # 4. The classic agent: create_tool_calling_agent + AgentExecutor
        logger.info("--- 2. Classic agent (AgentExecutor) ---")
        classic_agent(model)
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        # 401: wrong key · 404: wrong deployment name · 429: rate limit
        logger.error(f"Azure API error {error.status_code}: {error.message}")
        raise SystemExit(1)
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        logger.error("Could not reach Azure: check AZURE_OPENAI_ENDPOINT or ANTHROPIC_FOUNDRY_RESOURCE")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
