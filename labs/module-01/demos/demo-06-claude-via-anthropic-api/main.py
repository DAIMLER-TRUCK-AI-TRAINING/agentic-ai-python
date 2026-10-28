"""
Demo 6: Claude Messages API with an Anthropic API key
Uses the Anthropic SDK to call Claude directly on the Claude API (api.anthropic.com).
Demo 7 makes exactly the same calls through Microsoft Foundry with an Azure key;
only build_client() differs.
"""

import os
import logging
from dotenv import load_dotenv
from anthropic import Anthropic, APIConnectionError, APIStatusError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a python expert. Answer in at most three sentences."
QUESTION = "What is python programming?"


def require_env(name):
    """Return an environment variable or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables!")
    return value


def build_client():
    """
    Build the Anthropic SDK client for the Claude API.
    The key comes from console.anthropic.com; the model is a Claude model ID.
    """
    client = Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    model = os.getenv("MODEL_NAME", "claude-opus-5-5")
    logger.info(f"Provider: anthropic | base URL: {client.base_url} | model: {model}")
    return client, model


def response_text(message):
    """Join the text blocks: content is a list of blocks, not a single string."""
    return "".join(block.text for block in message.content if block.type == "text")


def log_usage(message):
    """Claude reports input and output tokens separately; there is no total field."""
    usage = message.usage
    logger.info(f"Stop reason: {message.stop_reason} | "
                f"Input tokens: {usage.input_tokens}, Output tokens: {usage.output_tokens}")


def single_call(client, model):
    # system is a top-level parameter, not a message role; max_tokens is required
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": QUESTION}]
    )
    # Current models think before answering, so expect a thinking block before the text
    logger.info(f"Content blocks: {[block.type for block in message.content]}")
    logger.info(response_text(message))
    log_usage(message)
    return message


def multi_turn(client, model, first):
    # The API is stateless: send the whole conversation back, with the assistant's
    # previous content passed back unchanged (thinking blocks included)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": first.content},
            {"role": "user", "content": "Show me a one-line example."}
        ]
    )
    logger.info(response_text(message))
    log_usage(message)


def streaming(client, model):
    # messages.stream() yields text as it is generated; get_final_message() returns
    # the same Message object that messages.create() would have returned
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Name three popular Python web frameworks."}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print(flush=True)
        message = stream.get_final_message()
    log_usage(message)


def main():
    # 1. Environment setup: load the key and model from .env
    load_dotenv()
    # 2. Client initialization: Anthropic SDK pointed at the Claude API
    client, model = build_client()
    try:
        # 3. A single call: system prompt, user message, content blocks and usage
        logger.info("--- 1. Single call ---")
        first = single_call(client, model)
        # 4. A follow-up turn: resend the history
        logger.info("--- 2. Multi-turn ---")
        multi_turn(client, model, first)
        # 5. Streaming: text arrives as it is generated
        logger.info("--- 3. Streaming ---")
        streaming(client, model)
    except APIStatusError as error:
        # 401: wrong key · 404: unknown model · 429: rate limit
        logger.error(f"Claude API error {error.status_code}: {error.message}")
        raise SystemExit(1)
    except APIConnectionError:
        logger.error(f"Could not reach {client.base_url}: check your network or proxy")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
