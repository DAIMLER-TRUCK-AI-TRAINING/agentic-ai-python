"""
Demo 2: Basic Streaming Response
Basic streaming implementation using OpenAI-compatible SDK.
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

def main():
    # 1. Environment setup: load API key, base_url, and model from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    base_url = os.getenv("GEMINI_BASE_URL")
    model = os.getenv("GEMINI_MODEL")

    # 2. Client initialization: OpenAI SDK configured for Gemini via base_url
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )



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