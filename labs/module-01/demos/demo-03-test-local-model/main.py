"""
Demo 3: Test Local Model
Test local model using OpenAI-compatible SDK.
Runs on local Ollama by default; set LLM_PROVIDER=openai or azure in .env
to run the same call on OpenAI or Azure OpenAI instead.
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def require_env(name):
    """Return an environment variable or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables!")
    return value


def build_client():
    """
    Build the OpenAI SDK client and model name for the provider in LLM_PROVIDER.
    All three providers speak the OpenAI API; only key, base_url and model differ.
    """
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

    if provider == "ollama":
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama"   # Placeholder (Ollama ignores this)
        )
        model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
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
            f"Unsupported LLM_PROVIDER '{provider}'. Use 'ollama', 'openai' or 'azure'."
        )

    logger.info(f"Provider: {provider} | base URL: {client.base_url} | model: {model}")
    return client, model


def main():
    # 1. Configure OpenAI client for the chosen provider (local Ollama by default)
    load_dotenv()
    client, model = build_client()

    try:
        # 2. API invocation: call chat.completions.create()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Explain reinforcement learning in simple terms."}
            ]
        )
        logger.info(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error while running model: {e}")


if __name__ == "__main__":
    main()