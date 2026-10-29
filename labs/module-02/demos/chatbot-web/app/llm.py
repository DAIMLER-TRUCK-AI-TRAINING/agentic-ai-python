"""
LangChain chat model for the web chatbot.

LLM_PROVIDER picks one of four providers. init_chat_model returns a ChatOpenAI or a
ChatAnthropic; only the model name, endpoint and key differ, so the streaming,
memory and guardrail code never changes.

    openai            OpenAI API                    OPENAI_API_KEY, OPENAI_MODEL
    anthropic         Claude API                    ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    azure-openai      Azure OpenAI                  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
    azure-anthropic   Claude on Microsoft Foundry   ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_RESOURCE, ANTHROPIC_FOUNDRY_DEPLOYMENT

The model is built on the first chat turn, so the UI, /healthz and the tests run
without a key.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("openai", "anthropic", "azure-openai", "azure-anthropic")

# provider → variable holding the model ID (the deployment name on Azure)
MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "azure-openai": "AZURE_OPENAI_DEPLOYMENT",
    "azure-anthropic": "ANTHROPIC_FOUNDRY_DEPLOYMENT",
}

# Claude requires an output cap; replies here are a few sentences
CLAUDE_MAX_TOKENS = 2048

# List prices in USD per 1M tokens, copied from labs/module-02/data/model_prices.json
# (as of 2026-10-27). Only used for the cost estimate in the UI; a model missing here
# shows its cost as "—". Azure deployments match when named after their model.
PRICES_PER_1M = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-5-5": {"input": 0.1, "output": 0.5},
    "gpt-6-astra": {"input": 10.0, "output": 50.0},
    "gpt-6.1-sol": {"input": 2.0, "output": 10.0},
    "gpt-6.1-terra": {"input": 1.0, "output": 5.0},
    "gpt-6-luna": {"input": 0.1, "output": 0.5},
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
}


class ProviderConfigError(ValueError):
    """LLM_PROVIDER is unknown, or a variable it needs is not set."""


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ProviderConfigError(f"{name} not found in environment variables!")
    return value


def get_provider() -> str:
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ProviderConfigError(
            f"Unsupported LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}."
        )
    return provider


def describe() -> dict:
    """Provider, model and prices for /healthz and the UI. Never raises, never shows a key."""
    try:
        provider = get_provider()
    except ProviderConfigError as exc:
        return {"provider": None, "model": None, "price_per_1m": None, "config_error": str(exc)}
    model = os.getenv(MODEL_ENV[provider]) or None
    return {"provider": provider, "model": model, "price_per_1m": PRICES_PER_1M.get(model or "")}


def _openai_options() -> dict:
    # stream_usage: the last streamed chunk carries the real token counts
    options: dict = {"stream_usage": True}
    # Reasoning models think before the first token; a low effort shortens time-to-first-token
    if effort := os.getenv("OPENAI_REASONING_EFFORT"):
        options["reasoning_effort"] = effort
    return options


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Build the chat model for LLM_PROVIDER on first use, then reuse it."""
    provider = get_provider()
    model_name = _require_env(MODEL_ENV[provider])

    if provider == "openai":
        model = init_chat_model(
            model_name,
            model_provider="openai",
            api_key=_require_env("OPENAI_API_KEY"),
            **_openai_options(),
        )
    elif provider == "anthropic":
        model = init_chat_model(
            model_name,
            model_provider="anthropic",
            api_key=_require_env("ANTHROPIC_API_KEY"),
            max_tokens=CLAUDE_MAX_TOKENS,
        )
    elif provider == "azure-openai":
        # Azure OpenAI v1 API: the regular OpenAI client, base_url points at your Azure resource
        model = init_chat_model(
            model_name,
            model_provider="openai",
            base_url=f"{_require_env('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/",
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            **_openai_options(),
        )
    else:
        # Claude on Microsoft Foundry: the Anthropic Messages API under /anthropic on your resource
        model = init_chat_model(
            model_name,
            model_provider="anthropic",
            base_url=f"https://{_require_env('ANTHROPIC_FOUNDRY_RESOURCE')}.services.ai.azure.com/anthropic",
            api_key=_require_env("ANTHROPIC_FOUNDRY_API_KEY"),
            max_tokens=CLAUDE_MAX_TOKENS,
        )

    print(f"[llm] provider={provider} model={model_name}")
    return model
