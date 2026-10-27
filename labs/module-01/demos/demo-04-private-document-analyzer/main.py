"""
Demo 4: Private Document Analyzer
Uses the OpenAI-compatible client to call the local Ollama model.
Runs on local Ollama by default; set LLM_PROVIDER=openai or azure in .env
to run the same analysis on OpenAI or Azure OpenAI instead (the document
then leaves your machine).
"""


import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
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
    if provider != "ollama":
        logger.warning("Not a local model: the document text will be sent to this provider.")
    return client, model


# 1. Configure the provider (local Ollama by default)
load_dotenv()
client, model = build_client()

def analyze_document_privately(file_path, analysis_type="summary"):
    """
    Analyze a document using a local LLM for complete privacy.

    Args:
        file_path: Path to document file
        analysis_type: Type of analysis ('summary', 'key_points', 'sentiment')
    Returns:
        Analysis results as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    # 2. Add the prompt to the messages
    '''Prompt is the instruction for the model to follow'''

    prompts = {
        "summary": "Provide a concise 3-sentence summary of this document.",
    }

    '''The messages is the list of messages to send to the model'''
    messages = [
        {"role": "user", "content": f"{prompts[analysis_type]}\n\nDocument:\n{content}"}
    ]

    logger.info(f"Analyzing: {analysis_type}...")

    # 3. API invocation: call chat.completions.create()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    logger.info("Private Document Analyzer\n")
    
    # Use existing document file
    doc_file = "sample._doc.txt"
    
    result = analyze_document_privately(doc_file, "summary")
    logger.info(f"\nSUMMARY:")
    logger.info(result)
    logger.info("\nAnalysis complete!")
