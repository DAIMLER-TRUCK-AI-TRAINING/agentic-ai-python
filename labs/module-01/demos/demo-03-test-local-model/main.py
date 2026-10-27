"""
Demo 3: Test Local Model
Test local model using OpenAI-compatible SDK.
"""

import logging
from dotenv import load_dotenv
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Configure OpenAI client for local Ollama
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"   # Placeholder (Ollama ignores this)
    )

    try:
        # 2. API invocation: call chat.completions.create()
        response = client.chat.completions.create(
            model="llama3.2:1b",
            messages=[
                {"role": "user", "content": "Explain reinforcement learning in simple terms."}
            ]
        )
        logger.info(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error while running model: {e}")


if __name__ == "__main__":
    main()