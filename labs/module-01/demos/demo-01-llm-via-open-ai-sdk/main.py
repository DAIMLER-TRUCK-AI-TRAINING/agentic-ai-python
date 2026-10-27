"""
Demo 1: Basic OpenAI API Call
Uses the OpenAI SDK to call Google Gemini, OpenAI or Azure OpenAI.
The provider is chosen with LLM_PROVIDER in .env (gemini | openai | azure).
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI  # per the document's pattern

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
    logger.info("LLM is calling")
    # 3. API invocation: call chat.completions.create()
    response = client.chat.completions.create(
            model=model,   
            messages=[
                {"role": "system", "content": "You are a python expert."},
                {"role": "user", "content": "What is python programming?"}
            ]
        )

    # 5. Response extraction (per doc): access response.choices[0].message.content
    generated_text = response.choices[0].message.content        
    logger.info(generated_text)

    # 6. Print the usage statistics
    usage = response.usage
    logger.info(f"Prompt tokens: {usage.prompt_tokens}, "
                f"Completion tokens: {usage.completion_tokens}, "
                f"Total tokens: {usage.total_tokens}")  

if __name__ == "__main__":
    main()
