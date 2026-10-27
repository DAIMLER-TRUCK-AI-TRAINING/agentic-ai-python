"""
Demo 1: Basic OpenAI API Call
Uses the OpenAI-compatible client to call Google Gemini via the OpenAI SDK.
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


def main():
    # 1. Environment setup: load API key, base_url, and model from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables!")
    base_url = os.getenv("GEMINI_BASE_URL")
    model = os.getenv("GEMINI_MODEL")
    # 2. Client initialization: OpenAI SDK configured for Gemini via base_url
    client = OpenAI(
        api_key=api_key,
        base_url= base_url
    )
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
