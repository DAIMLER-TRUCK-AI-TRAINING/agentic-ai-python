"""Provider factory for the course labs.

Module 2 Labs 04–05 and the Phase 2 labs (Modules 3–5) import this module so
that one notebook runs on either provider. Module 1 and Module 2 Labs 01–03
call the OpenAI and Claude SDKs directly instead, because comparing the two
APIs is the lesson there.

    import sys; sys.path.append("../../setup")  # from labs/module-NN/notebooks
    from llm import get_chat_model, get_client, get_model_name

    llm = get_chat_model()               # LangChain chat model for LLM_PROVIDER
    llm = get_chat_model("anthropic")    # or force one provider
    client = get_client()                # raw openai.OpenAI / anthropic.Anthropic

    # Module 3 onwards: LlamaIndex models and embeddings from the same settings
    from llm import get_embeddings, get_llama_embed_model, get_llama_llm

Configuration comes from the repo-root ``.env`` (see ``.env.example``):
``LLM_PROVIDER`` picks one of four providers, in two API families:

=================  ===========  ==========================================
``LLM_PROVIDER``   API family   Model variable (``*_SMALL_*`` with small)
=================  ===========  ==========================================
``openai``         OpenAI       ``OPENAI_MODEL``
``anthropic``      Claude       ``ANTHROPIC_MODEL``
``azure-openai``   OpenAI       ``AZURE_OPENAI_DEPLOYMENT`` on ``AZURE_OPENAI_ENDPOINT``
``azure-anthropic`` Claude      ``ANTHROPIC_FOUNDRY_DEPLOYMENT`` on ``ANTHROPIC_FOUNDRY_RESOURCE``
=================  ===========  ==========================================

The Azure providers serve Module 3 onwards; Module 2 Labs 04–05 use
``openai`` or ``anthropic``. ``EMBEDDING_PROVIDER`` (``local``, ``openai`` or
``azure-openai``) picks the embedding model separately, because Claude has no
embedding model of its own. Nothing here hard-codes a chat model ID.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Literal, get_args

from dotenv import find_dotenv, load_dotenv

if TYPE_CHECKING:
    import anthropic
    import openai
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM

Provider = Literal["openai", "anthropic", "azure-openai", "azure-anthropic"]
EmbeddingProvider = Literal["local", "openai", "azure-openai"]

# provider -> (key variable, model variable, small-model variable)
_ENV: dict[str, tuple[str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_SMALL_MODEL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_MODEL"),
    "azure-openai": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_SMALL_DEPLOYMENT"),
    "azure-anthropic": (
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_DEPLOYMENT",
        "ANTHROPIC_FOUNDRY_SMALL_DEPLOYMENT",
    ),
}
# Azure providers also need to know where the resource is.
_LOCATION = {"azure-openai": "AZURE_OPENAI_ENDPOINT", "azure-anthropic": "ANTHROPIC_FOUNDRY_RESOURCE"}
# The other API family on the same platform: what a comparison cell runs next to LLM_PROVIDER.
_PARTNER = {
    "openai": "anthropic",
    "anthropic": "openai",
    "azure-openai": "azure-anthropic",
    "azure-anthropic": "azure-openai",
}
DEFAULT_LOCAL_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLAUDE_MAX_TOKENS = 4096  # otherwise LangChain asks for the model's maximum output

load_dotenv(find_dotenv(usecwd=True))


class ProviderConfigError(RuntimeError):
    """Raised when the provider, its API key or its model ID is not configured."""


def get_provider(provider: str | None = None) -> Provider:
    """Return the provider to use: the argument, else ``LLM_PROVIDER``, else ``openai``."""
    name = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if name not in get_args(Provider):
        raise ProviderConfigError(f"LLM_PROVIDER must be one of {get_args(Provider)}, got {name!r}")
    return name  # type: ignore[return-value]


def get_family(provider: str | None = None) -> Literal["openai", "anthropic"]:
    """The API family a provider speaks: ``openai`` (OpenAI, Azure OpenAI) or ``anthropic`` (Claude)."""
    return "anthropic" if get_provider(provider).endswith("anthropic") else "openai"


def partner_provider(provider: str | None = None) -> Provider:
    """The other API family on the same platform, for side-by-side comparison cells."""
    return _PARTNER[get_provider(provider)]  # type: ignore[return-value]


def is_configured(provider: str | None = None) -> bool:
    """True if the provider's key, model and (for Azure) resource are all set."""
    name = get_provider(provider)
    needed = [_ENV[name][0], _ENV[name][1], *([_LOCATION[name]] if name in _LOCATION else [])]
    return all(os.getenv(variable, "").strip() for variable in needed)


def _require(variable: str, provider: str) -> str:
    value = os.getenv(variable, "").strip()
    if not value:
        raise ProviderConfigError(
            f"{variable} is not set for provider {provider!r}. "
            "Add it to the repo-root .env (template: labs/setup/.env.example)."
        )
    return value


def get_model_name(provider: str | None = None, *, small: bool = False) -> str:
    """Return the configured model ID (or Azure deployment name) for ``provider``."""
    name = get_provider(provider)
    _, model_var, small_var = _ENV[name]
    return _require(small_var if small else model_var, name)


def get_base_url(provider: str | None = None) -> str | None:
    """The API base URL for an Azure provider; ``None`` for the providers' own endpoints."""
    name = get_provider(provider)
    if name == "azure-openai":
        return f"{_require('AZURE_OPENAI_ENDPOINT', name).rstrip('/')}/openai/v1/"
    if name == "azure-anthropic":
        return f"https://{_require('ANTHROPIC_FOUNDRY_RESOURCE', name)}.services.ai.azure.com/anthropic"
    return None


def get_client(provider: str | None = None) -> openai.OpenAI | anthropic.Anthropic:
    """Return the raw SDK client for ``provider``; keys are read from the environment."""
    name = get_provider(provider)
    key = _require(_ENV[name][0], name)
    if name == "openai":
        import openai

        return openai.OpenAI()
    if name == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    if name == "azure-openai":
        import openai

        # Azure OpenAI v1 API: the regular client, pointed at your resource; model= is the deployment.
        return openai.OpenAI(api_key=key, base_url=get_base_url(name))
    import anthropic

    return anthropic.AnthropicFoundry(api_key=key, resource=_require("ANTHROPIC_FOUNDRY_RESOURCE", name))


def get_chat_model(provider: str | None = None, *, small: bool = False, **kwargs: Any) -> BaseChatModel:
    """Return a LangChain chat model for ``provider``.

    Extra keyword arguments go to the underlying chat model class. For OpenAI the
    Responses API is the default (``use_responses_api=True``), matching the labs;
    on Azure OpenAI LangChain picks the API itself. Do not pass ``temperature``
    for Claude: current Claude models accept only the default sampling settings.
    """
    from langchain.chat_models import init_chat_model

    name = get_provider(provider)
    key = _require(_ENV[name][0], name)
    model = get_model_name(name, small=small)
    if name == "openai":
        kwargs.setdefault("use_responses_api", True)
        return init_chat_model(model, model_provider="openai", **kwargs)
    if name == "anthropic":
        return init_chat_model(model, model_provider="anthropic", **kwargs)
    family = get_family(name)
    if family == "anthropic":
        kwargs.setdefault("max_tokens", CLAUDE_MAX_TOKENS)
    return init_chat_model(model, model_provider=family, base_url=get_base_url(name), api_key=key, **kwargs)


def get_llama_llm(provider: str | None = None, *, small: bool = False, **kwargs: Any) -> LLM:
    """Return a LlamaIndex LLM for ``provider``, from the same settings as ``get_chat_model``.

    OpenAI and Azure OpenAI use the Responses API (``OpenAIResponses``); Claude and
    Claude on Foundry use ``Anthropic``. Extra keyword arguments go to that class.
    Azure deployments with names LlamaIndex does not know get a 128K context
    window (OpenAI) or a 200K one (Claude) for its prompt sizing.
    """
    name = get_provider(provider)
    key = _require(_ENV[name][0], name)
    model = get_model_name(name, small=small)
    if get_family(name) == "openai":
        from llama_index.llms.openai import OpenAIResponses
        from llama_index.llms.openai.utils import openai_modelname_to_contextsize

        if name == "azure-openai":
            kwargs.setdefault("api_base", get_base_url(name))
            kwargs.setdefault("api_key", key)
            try:
                openai_modelname_to_contextsize(model)
            except ValueError:
                kwargs.setdefault("context_window", 128_000)
        return OpenAIResponses(model=model, **kwargs)

    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.anthropic.utils import ANTHROPIC_MODELS

    kwargs.setdefault("max_tokens", CLAUDE_MAX_TOKENS)
    if name == "azure-anthropic":
        kwargs.setdefault("base_url", get_base_url(name))
        kwargs.setdefault("api_key", key)
        if model not in ANTHROPIC_MODELS:
            return _foundry_claude_class()(model=model, **kwargs)
    return Anthropic(model=model, **kwargs)


def _foundry_claude_class():
    """``Anthropic`` for a Foundry deployment whose name is not a Claude model ID."""
    from llama_index.core.llms import LLMMetadata
    from llama_index.llms.anthropic import Anthropic

    class FoundryClaude(Anthropic):
        """Claude on Foundry under a custom deployment name: fixed metadata, no temperature."""

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=200_000,
                num_output=self.max_tokens,
                is_chat_model=True,
                model_name=self.model,
                is_function_calling_model=True,
            )

        @property
        def _model_kwargs(self) -> dict[str, Any]:
            kwargs = dict(super()._model_kwargs)
            kwargs.pop("temperature", None)  # current Claude models reject sampling settings
            return kwargs

    return FoundryClaude


# --------------------------------------------------------------------------- embeddings


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Return the embedding provider: the argument, else ``EMBEDDING_PROVIDER``, else ``local``."""
    name = (provider or os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower()
    if name not in get_args(EmbeddingProvider):
        raise ProviderConfigError(f"EMBEDDING_PROVIDER must be one of {get_args(EmbeddingProvider)}, got {name!r}")
    return name  # type: ignore[return-value]


def get_embedding_model_name(provider: str | None = None) -> str:
    """The embedding model ID (or Azure deployment) that ``provider`` uses."""
    name = get_embedding_provider(provider)
    if name == "local":
        return os.getenv("ST_MODEL", "").strip() or DEFAULT_LOCAL_EMBEDDING
    if name == "openai":
        return _require("OPENAI_EMBEDDING_MODEL", name)
    return _require("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", name)


def embedding_slug(provider: str | None = None) -> str:
    """A short, index-name-safe ID for the embedding model, for example ``local-paraphrase-multilingual``.

    Store it with every index: vectors from two embedding models must never share one.
    """
    name = get_embedding_provider(provider)
    model = get_embedding_model_name(name).split("/")[-1].lower()
    model = re.sub(r"-minilm.*$", "", model) if name == "local" else model
    return re.sub(r"[^a-z0-9-]+", "-", f"{name}-{model}").strip("-")[:40]


def _local_embeddings_class():
    """``LocalEmbeddings``: Sentence Transformers behind LangChain's ``Embeddings`` interface."""
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class LocalEmbeddings(Embeddings):
        """A local Sentence Transformers model; unit-length vectors, so cosine = inner product."""

        def __init__(self, model_name: str | None = None) -> None:
            self.model_name = model_name or get_embedding_model_name("local")
            self.model = SentenceTransformer(self.model_name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    return LocalEmbeddings


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Return a LangChain embeddings object for ``EMBEDDING_PROVIDER`` (default ``local``)."""
    name = get_embedding_provider(provider)
    if name == "local":
        return _local_embeddings_class()()
    from langchain_openai import OpenAIEmbeddings

    if name == "openai":
        _require("OPENAI_API_KEY", name)
        return OpenAIEmbeddings(model=get_embedding_model_name(name))
    return OpenAIEmbeddings(
        model=get_embedding_model_name(name),
        base_url=get_base_url("azure-openai"),
        api_key=_require("AZURE_OPENAI_API_KEY", name),
        check_embedding_ctx_length=False,  # the deployment name is not a tiktoken model name
    )


def get_llama_embed_model(provider: str | None = None) -> BaseEmbedding:
    """Return a LlamaIndex embedding model for ``EMBEDDING_PROVIDER`` (default ``local``)."""
    name = get_embedding_provider(provider)
    if name == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=get_embedding_model_name(name), normalize=True)
    from llama_index.embeddings.openai import OpenAIEmbedding

    if name == "openai":
        _require("OPENAI_API_KEY", name)
        return OpenAIEmbedding(model=get_embedding_model_name(name))
    # LlamaIndex validates model= against OpenAI's names; model_name= carries the Azure deployment.
    return OpenAIEmbedding(
        model="text-embedding-3-small",
        model_name=get_embedding_model_name(name),
        api_base=get_base_url("azure-openai"),
        api_key=_require("AZURE_OPENAI_API_KEY", name),
    )


def describe() -> str:
    """One-line summary of the active configuration, safe to print (no secrets, no endpoints)."""
    name = get_provider()
    key_var, model_var, small_var = _ENV[name]
    key_state = "present" if os.getenv(key_var) else "MISSING"
    line = (
        f"LLM_PROVIDER={name} · {model_var}={os.getenv(model_var) or '—'} · "
        f"{small_var}={os.getenv(small_var) or '—'} · {key_var} {key_state}"
    )
    if name in _LOCATION:
        line += f" · {_LOCATION[name]} {'present' if os.getenv(_LOCATION[name]) else 'MISSING'}"
    return line
