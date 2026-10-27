from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def require_env(name):
    """Return an environment variable or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables / .env file")
    return value


def get_client():
    """
    Return (client, model) for the provider selected by LLM_PROVIDER in .env:
      openai (default) -> OpenAI API, model from MODEL_NAME
      azure            -> Azure OpenAI, model is your AZURE_OPENAI_DEPLOYMENT name
    Both use the same OpenAI client, so the calling code does not change.
    """
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "openai":
        # The SDK defaults to https://api.openai.com/v1 (override with OPENAI_BASE_URL)
        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    elif provider == "azure":
        # Azure OpenAI v1 API: same OpenAI client, just point base_url at your Azure resource.
        # The model is your deployment name, not the base model name.
        endpoint = require_env("AZURE_OPENAI_ENDPOINT")  # e.g. https://<your-resource>.openai.azure.com
        client = OpenAI(
            api_key=require_env("AZURE_OPENAI_API_KEY"),  # Azure portal > your resource > Keys and Endpoint
            base_url=f"{endpoint.rstrip('/')}/openai/v1/",
        )
        model = require_env("AZURE_OPENAI_DEPLOYMENT")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Use 'openai' or 'azure'.")

    print(f"[{provider}] base URL: {client.base_url} | model: {model}")
    return client, model
