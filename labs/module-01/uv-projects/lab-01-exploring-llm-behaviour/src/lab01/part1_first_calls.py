"""Part 1 — the same prompt to both providers, and the shape of each response."""

from __future__ import annotations

import time

import pandas as pd

from .common import ANTHROPIC_MODEL, OPENAI_MODEL, available, claude_client, claude_text, openai_client, show, step

PROMPT = "In two sentences, what does an L1 service-desk analyst do?"


def ask_openai(prompt: str, *, effort: str = "low", max_output_tokens: int = 2_000) -> str:
    """Send one prompt to OPENAI_MODEL and return the visible text."""
    response = openai_client().responses.create(
        model=OPENAI_MODEL, input=prompt, reasoning={"effort": effort}, max_output_tokens=max_output_tokens
    )
    return response.output_text.strip()


def ask_claude(prompt: str, *, effort: str = "low", max_tokens: int = 2_000) -> str:
    """Send one prompt to ANTHROPIC_MODEL and return the visible text."""
    message = claude_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    return claude_text(message).strip()


ASK = {"openai": ask_openai, "anthropic": ask_claude}


def run() -> None:
    shapes = []
    if available("openai"):
        step(f"OpenAI Responses API · {OPENAI_MODEL}")
        started = time.perf_counter()
        response = openai_client().responses.create(
            model=OPENAI_MODEL,
            input=PROMPT,
            reasoning={"effort": "low"},  # GPT-6 models reason before answering; low effort keeps this quick
            max_output_tokens=2_000,  # the cap covers reasoning tokens AND visible text
        )
        elapsed = time.perf_counter() - started
        usage = response.usage
        print(response.output_text, "\n")
        print(
            f"status={response.status} · input={usage.input_tokens} · output={usage.output_tokens} "
            f"(reasoning {usage.output_tokens_details.reasoning_tokens}) · {elapsed:.1f}s · id={response.id}"
        )
        shapes.append(
            {
                "provider": "OpenAI Responses",
                "output items": [item.type for item in response.output],
                "text": "response.output_text",
                "finished": f"status={response.status}",
                "usage": "input_tokens · output_tokens · output_tokens_details.reasoning_tokens",
            }
        )

    if available("anthropic"):
        step(f"Claude Messages API · {ANTHROPIC_MODEL}")
        started = time.perf_counter()
        message = claude_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2_000,  # required; the cap covers thinking AND visible text
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": PROMPT}],
        )
        elapsed = time.perf_counter() - started
        usage = message.usage
        print(claude_text(message), "\n")
        print(
            f"stop_reason={message.stop_reason} · input={usage.input_tokens} · output={usage.output_tokens} "
            f"· {elapsed:.1f}s · id={message.id}"
        )
        shapes.append(
            {
                "provider": "Claude Messages",
                "output items": [block.type for block in message.content],
                "text": "text blocks in message.content",
                "finished": f"stop_reason={message.stop_reason}",
                "usage": "input_tokens · output_tokens (thinking counts as output)",
            }
        )

    if shapes:
        step("The two response shapes side by side")
        show(pd.DataFrame(shapes), index=False)
