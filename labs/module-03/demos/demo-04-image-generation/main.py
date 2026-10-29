"""
Demo 04: Text-to-Image Generation
Generate an image from a text prompt with the OpenAI Images API (gpt-image-1),
and demonstrate the prompt-engineering habit from Guide 06 §9: build a prompt
from explicit visual dimensions (subject, setting, lighting, style).

The output PNG is written next to this script.
"""

import os
import base64
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")


def build_prompt() -> str:
    """Compose a strong image prompt from explicit visual dimensions.

    Pattern: subject, setting, composition, lighting, color/mood, style, detail.
    """
    parts = [
        "A red fox curled asleep",          # subject + pose
        "in a snowy pine forest",            # setting
        "wide establishing shot",            # composition
        "soft golden-hour backlight",        # lighting
        "cool blue shadows",                 # color / mood
        "watercolor illustration",           # style / medium
        "delicate paper texture, highly detailed",  # detail cues
    ]
    return ", ".join(parts)


def generate(prompt: str, size: str = "1024x1024", quality: str = "medium") -> bytes:
    """Call the Images API and return the decoded PNG bytes."""
    logger.info("Generating image...")
    result = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size,        # 1024x1024 | 1536x1024 (landscape) | 1024x1536 (portrait) | auto
        quality=quality,  # low | medium | high
        n=1,
    )
    return base64.b64decode(result.data[0].b64_json)


def main() -> None:
    prompt = build_prompt()
    logger.info(f"Prompt:\n  {prompt}\n")

    image_bytes = generate(prompt)

    out = f"output_{datetime.now():%Y%m%d_%H%M%S}.png"
    with open(out, "wb") as f:
        f.write(image_bytes)
    logger.info(f"Saved -> {out}  ({len(image_bytes) // 1024} KB)")
    logger.info(
        "\nTip: change ONE dimension in build_prompt() at a time "
        "(e.g. style or lighting) to learn what each does."
    )


if __name__ == "__main__":
    main()
