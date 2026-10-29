"""
Demo 05: Vision / Multimodal Understanding
The other half of multimodal AI: instead of text -> image (Demo 04), here we do
image -> text. A multimodal chat model reads an image alongside a text question
and answers in natural language (Guide 06 §7.1).

Point it at any public image URL, or a local file (auto base64-encoded).
"""

import os
import base64
import logging
import mimetypes
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI()
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

# A public image works out of the box. Override with a URL or local path in .env.
DEFAULT_IMAGE = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)
IMAGE_SOURCE = os.getenv("IMAGE_SOURCE", DEFAULT_IMAGE)

QUESTION = "Describe this image in two sentences, then list any safety concerns."


def to_image_url(source: str) -> str:
    """Pass through an http(s) URL; base64-encode a local file as a data URL."""
    if source.startswith(("http://", "https://")):
        return source
    mime = mimetypes.guess_type(source)[0] or "image/jpeg"
    with open(source, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def ask_about_image(image_source: str, question: str) -> str:
    """Send text + image in one message; the model reasons over both."""
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": to_image_url(image_source)}},
            ],
        }],
    )
    return resp.choices[0].message.content


def main() -> None:
    logger.info(f"Vision model: {VISION_MODEL}")
    logger.info(f"Image: {IMAGE_SOURCE}")
    logger.info(f"Question: {QUESTION}\n")

    logger.info("Analyzing image...")
    logger.info("\n=== Answer ===\n")
    logger.info(ask_about_image(IMAGE_SOURCE, QUESTION))


if __name__ == "__main__":
    main()
