"""Provider factory for the course labs.

Phase 2 labs (Modules 3–5) import this module so that one notebook runs on
either provider. Phase 1 labs call the OpenAI and Claude SDKs directly
instead, because comparing the two APIs is the lesson there.

    import sys; sys.path.append("../../setup")  # from labs/module-NN/notebooks
    from llm import get_chat_model, get_client, get_model_name

    llm = get_chat_model()               # LangChain chat model for LLM_PROVIDER
    llm = get_chat_model("anthropic")    # or force one provider
    client = get_client()                # raw openai.OpenAI / anthropic.Anthropic

Configuration comes from the repo-root ``.env`` (see ``.env.example``):
``LLM_PROVIDER`` picks ``openai`` or ``anthropic``; ``OPENAI_MODEL`` /
``ANTHROPIC_MODEL`` (or ``*_SMALL_MODEL`` with ``small=True``) pick the model.
Nothing here hard-codes a model ID.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, get_args

from dotenv import find_dotenv, load_dotenv

if TYPE_CHECKING:
    import anthropic
    import openai
    from langchain_core.language_models.chat_models import BaseChatModel

Provider = Literal["openai", "anthropic"]

# provider -> (key variable, model variable, small-model variable)
_ENV: dict[str, tuple[str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_SMALL_MODEL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_MODEL"),
}

load_dotenv(find_dotenv(usecwd=True))


class ProviderConfigError(RuntimeError):
    """Raised when the provider, its API key or its model ID is not configured."""


def get_provider(provider: str | None = None) -> Provider:
    """Return the provider to use: the argument, else ``LLM_PROVIDER``, else ``openai``."""
    name = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if name not in get_args(Provider):
        raise ProviderConfigError(f"LLM_PROVIDER must be one of {get_args(Provider)}, got {name!r}")
    return name  # type: ignore[return-value]


def _require(variable: str, provider: str) -> str:
    value = os.getenv(variable, "").strip()
    if not value:
        raise ProviderConfigError(
            f"{variable} is not set for provider {provider!r}. "
            "Add it to the repo-root .env (template: labs/setup/.env.example)."
        )
    return value


def get_model_name(provider: str | None = None, *, small: bool = False) -> str:
    """Return the configured model ID for ``provider`` (the small tier if ``small``)."""
    name = get_provider(provider)
    _, model_var, small_var = _ENV[name]
    return _require(small_var if small else model_var, name)


def get_client(provider: str | None = None) -> openai.OpenAI | anthropic.Anthropic:
    """Return the raw SDK client for ``provider``; the SDK reads the key from the environment."""
    name = get_provider(provider)
    _require(_ENV[name][0], name)
    if name == "openai":
        import openai

        return openai.OpenAI()
    import anthropic

    return anthropic.Anthropic()


def get_chat_model(provider: str | None = None, *, small: bool = False, **kwargs: Any) -> BaseChatModel:
    """Return a LangChain chat model for ``provider``.

    Extra keyword arguments go to the underlying chat model class. For OpenAI the
    Responses API is the default (``use_responses_api=True``), matching the labs.
    Do not pass ``temperature`` for Claude: current Claude models accept only
    the default sampling settings.
    """
    from langchain.chat_models import init_chat_model

    name = get_provider(provider)
    _require(_ENV[name][0], name)
    if name == "openai":
        kwargs.setdefault("use_responses_api", True)
    return init_chat_model(get_model_name(name, small=small), model_provider=name, **kwargs)


def describe() -> str:
    """One-line summary of the active configuration, safe to print (no secrets)."""
    name = get_provider()
    key_var, model_var, small_var = _ENV[name]
    key_state = "present" if os.getenv(key_var) else "MISSING"
    return (
        f"LLM_PROVIDER={name} · {model_var}={os.getenv(model_var) or '—'} · "
        f"{small_var}={os.getenv(small_var) or '—'} · {key_var} {key_state}"
    )
