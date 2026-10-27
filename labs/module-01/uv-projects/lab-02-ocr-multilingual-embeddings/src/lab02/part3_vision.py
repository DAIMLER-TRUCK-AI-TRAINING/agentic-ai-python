"""Part 3 — AI-assisted OCR: the same image and schema to OpenAI and Claude vision, then a deliberate failure."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from .common import ANTHROPIC_MODEL, OPENAI_MODEL, active_providers, claude_client, openai_client, show, step
from .part1_tesseract import CLEAN_INVOICE, SAMPLES_DIR, SCANNED_INVOICE, ground_truth, tesseract_ok
from .part2_accuracy import InvoiceFields, field_matches, field_report, tesseract_fields

EXTRACTION_PROMPT = (
    "Extract the invoice fields from this scanned invoice. Copy values exactly as printed. "
    "Use null for any field that is not printed; never guess."
)


def image_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def extract_openai(path: Path, schema: type[BaseModel] = InvoiceFields) -> tuple[BaseModel, float, int]:
    """(fields, seconds, total tokens) from the OpenAI Responses API with structured output."""
    started = time.perf_counter()
    response = openai_client().responses.parse(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": EXTRACTION_PROMPT},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64(path)}", "detail": "high"},
                ],
            }
        ],
        text_format=schema,
        reasoning={"effort": "low"},
        max_output_tokens=4_000,
    )
    return response.output_parsed, time.perf_counter() - started, response.usage.total_tokens


def extract_claude(path: Path, schema: type[BaseModel] = InvoiceFields) -> tuple[BaseModel, float, int]:
    """(fields, seconds, total tokens) from the Claude Messages API with structured output."""
    started = time.perf_counter()
    message = claude_client().messages.parse(
        model=ANTHROPIC_MODEL,
        max_tokens=4_000,
        output_config={"effort": "low"},
        output_format=schema,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64(path)}},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )
    usage = message.usage
    return message.parsed_output, time.perf_counter() - started, usage.input_tokens + usage.output_tokens


EXTRACT = {"openai": extract_openai, "anthropic": extract_claude}


class InvoiceFieldsNoNull(InvoiceFields):
    """A schema someone got wrong: po_number is a required string, so "not printed" cannot be said."""

    po_number: str = Field(description="Purchase-order number")


def run() -> None:
    providers = active_providers()
    if not providers:
        print("ℹ️  AI-assisted OCR needs a model: skipping this part.")
        return
    truth = ground_truth()

    step("Same image, same InvoiceFields schema, both providers")
    vision_results = {}
    for provider in providers:
        for name in (CLEAN_INVOICE, SCANNED_INVOICE):
            fields, seconds, tokens = EXTRACT[provider](SAMPLES_DIR / name)
            vision_results[(provider, name)] = fields
            print(f"{provider:>9} · {name:<20} {seconds:5.1f}s · {tokens:>5} tokens · {fields.model_dump()}")

    step("Field accuracy: Tesseract + regex versus the vision models (1 = field correct)")
    comparison = {}
    for name in (CLEAN_INVOICE, SCANNED_INVOICE):
        expected = truth[name]["fields"]
        if tesseract_ok():
            comparison[("tesseract+regex", name)] = field_report(tesseract_fields()[name], expected)
        for provider in providers:
            comparison[(provider, name)] = field_report(vision_results[(provider, name)], expected)
    accuracy = pd.DataFrame(comparison).astype(int)
    accuracy.loc["fields correct"] = accuracy.sum()
    show(accuracy)

    step(f"Deliberate failure: {SCANNED_INVOICE} has no PO number, but this schema makes po_number a required string")
    forced = []
    for provider in providers:
        strict, _, _ = EXTRACT[provider](SAMPLES_DIR / SCANNED_INVOICE, schema=InvoiceFieldsNoNull)
        forced.append(
            {
                "provider": provider,
                "po_number (nullable schema)": vision_results[(provider, SCANNED_INVOICE)].po_number,
                "po_number (no-null schema)": strict.po_number,
                "no-null value correct?": field_matches("po_number", strict.po_number, None),
            }
        )
    show(pd.DataFrame(forced), index=False)
