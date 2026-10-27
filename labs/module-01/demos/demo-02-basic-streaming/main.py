"""
Demo 2: Basic Streaming Response
Basic streaming implementation using OpenAI-compatible SDK.
The provider is chosen with LLM_PROVIDER in .env (gemini | openai | azure).
"""

import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI

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
        raise ValueError(f"{name} not found in .env file")
    return value


def build_client():
    """
    Build the OpenAI SDK client and model name for the provider in LLM_PROVIDER.
    All three providers speak the OpenAI API; only key, base_url and model differ.
    """
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()

    if provider == "gemini":
        # Gemini's OpenAI-compatible endpoint
        client = OpenAI(
            api_key=require_env("GEMINI_API_KEY"),
            base_url=require_env("GEMINI_BASE_URL")
        )
        model = require_env("GEMINI_MODEL")
    elif provider == "openai":
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    elif provider == "azure":
        # Azure OpenAI v1 API: same OpenAI client, base_url points at your Azure resource.
        # "model" is the deployment name you created in Azure, not the base model name.
        endpoint = require_env("AZURE_OPENAI_ENDPOINT")
        client = OpenAI(
            api_key=require_env("AZURE_OPENAI_API_KEY"),
            base_url=f"{endpoint.rstrip('/')}/openai/v1/"
        )
        model = require_env("AZURE_OPENAI_DEPLOYMENT")
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. Use 'gemini', 'openai' or 'azure'."
        )

    logger.info(f"Provider: {provider} | base URL: {client.base_url} | model: {model}")
    return client, model


def main():
    # 1. Environment setup: load provider settings from .env file
    load_dotenv()

    # 2. Client initialization: OpenAI SDK configured for the chosen provider
    client, model = build_client()



    logger.info("\n=== Streaming Response ===\n")

    # 3. Stream the response (classic pattern)
    response_stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Explain what list comprehensions are in Python with an example."}
        ],
        stream=True            # Enable streaming mode
    )

    # 4. Log chunks progressively
    try:
        for chunk in response_stream:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    logger.info(delta.content)
    except KeyboardInterrupt:
        logger.warning("\n\n[Interrupted by user]")

if __name__ == "__main__":
    main()