"""Part 3 — hallucination probes on both providers, what code can verify, and verifiable grounding."""

from __future__ import annotations

import importlib.metadata
import json
import re
import textwrap

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from .common import (
    ANTHROPIC_MODEL,
    DATA_DIR,
    OPENAI_MODEL,
    active_providers,
    claude_client,
    openai_client,
    show,
    step,
)
from .part1_first_calls import ASK

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"'<>\])]+")
ARTICLE_PATH = DATA_DIR / "kb" / "KB-0042.md"
QUESTIONS = {
    "answerable": "What should I do first if the VPN client rejects my new password?",
    "unanswerable": "What is the Service Desk's out-of-hours phone number?",
}


def load_probes() -> list[dict]:
    return json.loads((DATA_DIR / "hallucination_probes.json").read_text(encoding="utf-8"))


def run_probes(probes: list[dict], providers: list[str]) -> list[dict]:
    """Every probe on every provider; prints each answer next to the ground truth."""
    answers = []
    for probe in probes:
        row = {"id": probe["id"], "trigger": probe["trigger"]}
        for provider in providers:
            row[provider] = ASK[provider](probe["prompt"])
        answers.append(row)
        print(f"\n=== {probe['id']} · {probe['trigger']}\nPROMPT  {probe['prompt']}\nTRUTH   {probe['ground_truth']}")
        for provider in providers:
            print(f"--- {provider}\n" + textwrap.fill(row[provider], width=110, max_lines=8, placeholder=" […]"))
    return answers


def doi_exists(doi: str) -> bool | None:
    """True or False from the doi.org handle API; None if doi.org cannot be reached."""
    try:
        reply = httpx.get(f"https://doi.org/api/handles/{doi.rstrip('.,;')}", timeout=5)
    except httpx.HTTPError:
        return None
    return reply.status_code == 200


def latest_pypi_version(package: str) -> str | None:
    try:
        return httpx.get(f"https://pypi.org/pypi/{package}/json", timeout=5).json()["info"]["version"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def verify_claims(answers: list[dict], providers: list[str]) -> None:
    """Check what code can check: do the cited DOIs exist (H3), and is the stated version current (H5)?"""
    step("What code can verify: cited DOIs (H3) and the latest SDK version (H5)")
    citations = next((r for r in answers if r["id"] == "H3"), {})
    for provider in providers:
        dois = DOI_PATTERN.findall(citations.get(provider, ""))
        verdicts = {doi: doi_exists(doi) for doi in dois}
        print(f"{provider}: {len(dois)} DOI(s) → {verdicts or 'none given'}")
    live = latest_pypi_version("anthropic")
    installed = importlib.metadata.version("anthropic")
    print(f"anthropic on PyPI now: {live or 'unreachable'} · installed in this project: {installed} (pinned on purpose)")
    versions = next((r for r in answers if r["id"] == "H5"), {})
    for provider in providers:
        print(f"{provider} says: {versions.get(provider, '')[:40]!r}")


class GroundedAnswer(BaseModel):
    supported: bool = Field(description="True only if the article states the answer")
    answer: str | None = Field(description="The answer, or null when the article does not contain it")
    quote: str | None = Field(description="One sentence copied verbatim from the article that supports the answer, or null")


def grounding_instructions(article: str) -> str:
    return (
        "Answer the question using ONLY the knowledge-base article below. If the article does not contain "
        "the answer, set supported to false and answer and quote to null. Never use outside knowledge.\n\n"
        f"<article>\n{article}\n</article>"
    )


def grounded_openai(question: str, article: str) -> GroundedAnswer:
    response = openai_client().responses.parse(
        model=OPENAI_MODEL,
        instructions=grounding_instructions(article),
        input=question,
        text_format=GroundedAnswer,  # strict JSON schema generated from the Pydantic model
        reasoning={"effort": "low"},
        max_output_tokens=2_000,
    )
    return response.output_parsed


def grounded_claude(question: str, article: str) -> GroundedAnswer:
    message = claude_client().messages.parse(
        model=ANTHROPIC_MODEL,
        max_tokens=2_000,
        system=grounding_instructions(article),
        messages=[{"role": "user", "content": question}],
        output_format=GroundedAnswer,  # the SDK turns it into output_config.format
        output_config={"effort": "low"},
    )
    return message.parsed_output


GROUNDED = {"openai": grounded_openai, "anthropic": grounded_claude}


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def verify_grounding(result: GroundedAnswer, article: str) -> str:
    """Deterministic check: a supported answer must quote the article verbatim."""
    if not result.supported:
        return "abstained"
    if result.quote and normalise(result.quote) in normalise(article):
        return "grounded (quote found)"
    return "UNVERIFIED (quote not in article)"


def grounding(providers: list[str]) -> None:
    step("Grounding: the KB-0042 article, permission to abstain and a verbatim quote checked in code")
    article = ARTICLE_PATH.read_text(encoding="utf-8")
    rows = []
    for label, question in QUESTIONS.items():
        for provider in providers:
            result = GROUNDED[provider](question, article)
            rows.append(
                {
                    "question": label,
                    "provider": provider,
                    "supported": result.supported,
                    "verdict": verify_grounding(result, article),
                    "answer": (result.answer or "")[:100],
                }
            )
            if label == "unanswerable" and result.supported:
                print(f"⚠️  {provider} claimed the article answers the unanswerable question")
    show(pd.DataFrame(rows), index=False)


def run() -> None:
    probes = load_probes()
    step("The probe set (data/hallucination_probes.json)")
    show(pd.DataFrame(probes)[["id", "trigger", "prompt"]], index=False)
    providers = active_providers()
    if not providers:
        print("ℹ️  The probes and the grounding step need a model: skipping them.")
        return
    answers = run_probes(probes, providers)
    verify_claims(answers, providers)
    grounding(providers)
