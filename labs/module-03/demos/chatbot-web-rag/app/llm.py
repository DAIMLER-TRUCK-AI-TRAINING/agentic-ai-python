"""
LangChain models for the RAG chatbot: one chat model and one embedding model.

LLM_PROVIDER picks the chat model. init_chat_model returns a ChatOpenAI or a
ChatAnthropic; only the model name, endpoint and key differ, so the streaming,
memory, retrieval and guardrail code never changes.

    openai            OpenAI API                    OPENAI_API_KEY, OPENAI_MODEL
    anthropic         Claude API                    ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    azure-openai      Azure OpenAI                  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
    azure-anthropic   Claude on Microsoft Foundry   ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_RESOURCE, ANTHROPIC_FOUNDRY_DEPLOYMENT

EMBEDDING_PROVIDER picks the embedding model separately, because Claude has no
embedding model of its own:

    local (default)   Sentence Transformers on this machine: no key, multilingual   ST_MODEL (optional)
    openai            OpenAI embeddings                                            OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
    azure-openai      Azure OpenAI embeddings                                      AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
                                                                                   AZURE_OPENAI_EMBEDDING_DEPLOYMENT

Both models are built on first use, so the UI, /healthz and the tests run
without a key.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from threading import Lock

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("openai", "anthropic", "azure-openai", "azure-anthropic")
EMBEDDING_PROVIDERS = ("local", "openai", "azure-openai")

# provider → variable holding the model ID (the deployment name on Azure)
MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "azure-openai": "AZURE_OPENAI_DEPLOYMENT",
    "azure-anthropic": "ANTHROPIC_FOUNDRY_DEPLOYMENT",
}

# Claude requires an output cap; replies here are a few sentences
CLAUDE_MAX_TOKENS = 2048

# The course's local embedding model: multilingual, so a German question finds an English
# document and the other way round
DEFAULT_LOCAL_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OPENAI_EMBEDDING = "text-embedding-3-small"


class ProviderConfigError(ValueError):
    """A provider is unknown, or a variable it needs is not set."""


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ProviderConfigError(f"{name} not found in environment variables!")
    return value


def _azure_openai_base_url() -> str:
    # Azure OpenAI v1 API: the regular OpenAI client, base_url points at your Azure resource
    return f"{_require_env('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"


# ── Chat model ────────────────────────────────────────────────────────────────

def get_provider() -> str:
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ProviderConfigError(
            f"Unsupported LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}."
        )
    return provider


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
        model = init_chat_model(
            model_name,
            model_provider="openai",
            base_url=_azure_openai_base_url(),
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


# ── Embedding model ───────────────────────────────────────────────────────────

def get_embedding_provider() -> str:
    provider = (os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower()
    if provider not in EMBEDDING_PROVIDERS:
        raise ProviderConfigError(
            f"Unsupported EMBEDDING_PROVIDER '{provider}'. Use one of: {', '.join(EMBEDDING_PROVIDERS)}."
        )
    return provider


def get_embedding_model_name() -> str:
    provider = get_embedding_provider()
    if provider == "local":
        return os.getenv("ST_MODEL") or DEFAULT_LOCAL_EMBEDDING
    if provider == "openai":
        return os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_OPENAI_EMBEDDING
    return _require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def embedding_slug() -> str:
    """A short, collection-name-safe ID of the embedding model, e.g. 'local-paraphrase-multilingual-minilm-l12-v2'.

    Vectors from two embedding models must never share a collection, so the
    Chroma collection is named after it.
    """
    model = get_embedding_model_name().split("/")[-1]
    slug = re.sub(r"[^a-z0-9-]+", "-", f"{get_embedding_provider()}-{model}".lower())
    return slug.strip("-")[:48]


class LocalEmbeddings(Embeddings):
    """A Sentence Transformers model behind LangChain's Embeddings interface."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # imported here: torch loads only when used

        # On the CPU: the server embeds from worker threads, which the Apple GPU backend does not always survive
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Unit-length vectors, so cosine similarity is a plain dot product
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_embeddings_lock = Lock()


@lru_cache(maxsize=1)
def _build_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    provider = get_embedding_provider()
    model_name = get_embedding_model_name()
    if provider == "local":
        embeddings: Embeddings = LocalEmbeddings(model_name)
    elif provider == "openai":
        embeddings = OpenAIEmbeddings(model=model_name, api_key=_require_env("OPENAI_API_KEY"))
    else:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            base_url=_azure_openai_base_url(),
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            check_embedding_ctx_length=False,  # the deployment name is not a tiktoken model name
        )
    print(f"[llm] embeddings={provider} model={model_name}")
    return embeddings


def get_embeddings() -> Embeddings:
    """Build the embedding model for EMBEDDING_PROVIDER on first use, then reuse it."""
    # Two first requests at once would otherwise load the local model twice
    with _embeddings_lock:
        return _build_embeddings()


# ── Status ────────────────────────────────────────────────────────────────────

def describe() -> dict:
    """Providers and models for /healthz and the UI. Never raises, never shows a key."""
    info: dict = {}
    try:
        provider = get_provider()
        info.update(provider=provider, model=os.getenv(MODEL_ENV[provider]) or None)
    except ProviderConfigError as exc:
        info.update(provider=None, model=None, config_error=str(exc))
    try:
        info["embeddings"] = {"provider": get_embedding_provider(), "model": get_embedding_model_name()}
    except ProviderConfigError as exc:
        info["embeddings"] = None
        info.setdefault("config_error", str(exc))
    return info
