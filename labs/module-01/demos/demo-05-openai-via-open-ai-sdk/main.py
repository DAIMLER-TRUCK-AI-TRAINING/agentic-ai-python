"""
Demo 5: Basic OpenAI API Call
Uses the OpenAI SDK to call OpenAI models.
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
    # 1. Environment setup: load API key and model from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables!")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    logger.info(f"Using model: {model}")
    
    base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
    logger.info(f"Using base URL: {base_url}")   
    # 2. Client initialization: OpenAI SDK configured for OpenAI
    client = OpenAI(
        api_key=api_key,
    )

    # base_url is optional if using default OpenAI endpoint
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url=base_url
    # )
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
