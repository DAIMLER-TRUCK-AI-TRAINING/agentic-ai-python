"""Part 4 — JSON in the prompt vs a schema the API enforces, on five change-request documents."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from .common import (
    ANTHROPIC_MODEL,
    DATA_DIR,
    OPENAI_MODEL,
    active_providers,
    available,
    claude_client,
    claude_text,
    openai_client,
    run_parallel,
    show,
    step,
)
from .part1_baseline import Group

# Five change-request documents, their gold extraction, and the target schema.
DOCS = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA_DIR / "change_requests").glob("CHG-*.txt"))}
GOLD = {
    k: v
    for k, v in json.loads((DATA_DIR / "change_requests" / "ground_truth.json").read_text(encoding="utf-8")).items()
    if k.startswith("CHG-")
}


class ChangeRequest(BaseModel):
    change_id: str = Field(description="Change ID exactly as written, e.g. CHG-20931")
    title: str = Field(description="Short title of the change, in English")
    change_type: Literal["standard", "normal", "emergency"]
    risk: Literal["low", "medium", "high"] = Field(description="Risk level as stated in the document")
    requested_by_team: Group
    region: Literal["EU", "US", "APAC"]
    affected_services: list[str] = Field(description="Host names or CI IDs the change touches, lower case")
    window_start: datetime = Field(description="Start of the CURRENT implementation window, ISO 8601 with UTC offset")
    window_end: datetime = Field(description="End of the current implementation window, ISO 8601 with UTC offset")
    cab_required: bool = Field(description="True if the document says CAB approval is needed")
    rollback_plan: str | None = Field(description="The rollback or back-out plan in one sentence, or null if the document has none")


INSTRUCTIONS = (
    "You extract change requests for the change-management database. If the document reschedules the change, "
    "use the current window. Use null for anything the document does not state; never invent a value."
)

# Approach A: ask for JSON in the prompt, then parse and validate it yourself.
PROMPT_ONLY_SYSTEM = (
    f"{INSTRUCTIONS}\nReturn ONLY a JSON object that matches this JSON Schema:\n"
    + json.dumps(ChangeRequest.model_json_schema())
)

# A reply recorded from an earlier run: valid content, invalid JSON.
RECORDED_REPLY = (
    "Here is the extracted change request:\n```json\n"
    '{"change_id": "CHG-20932", "change_type": "standard", "risk": "low"}\n'
    "```\nLet me know if you need anything else."
)


def parse_reply(text: str) -> tuple[str, ChangeRequest | None]:
    """Outcome of json.loads + schema validation on a raw model reply."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"invalid JSON ({exc.msg})", None
    try:
        return "valid", ChangeRequest.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        return f"schema error ×{exc.error_count()}: {'.'.join(map(str, first['loc']))}: {first['msg']}", None


def extract_prompt_only(provider: str, doc_id: str) -> dict:
    if provider == "openai":
        r = openai_client().responses.create(
            model=OPENAI_MODEL, instructions=PROMPT_ONLY_SYSTEM, input=DOCS[doc_id],
            reasoning={"effort": "low"}, max_output_tokens=4_000,
        )
        text = r.output_text
    else:
        m = claude_client().messages.create(
            model=ANTHROPIC_MODEL, max_tokens=4_000, system=PROMPT_ONLY_SYSTEM, output_config={"effort": "low"},
            messages=[{"role": "user", "content": DOCS[doc_id]}],
        )
        text = claude_text(m)
    outcome, _ = parse_reply(text)
    return {"provider": provider, "approach": "A: JSON in the prompt", "doc": doc_id, "outcome": outcome, "starts with": text[:25]}


def extract_native(provider: str, messages: list[dict]) -> ChangeRequest:
    """Approach B: one structured-output call; `messages` is the conversation so far (user turn last)."""
    if provider == "openai":
        r = openai_client().responses.parse(
            model=OPENAI_MODEL, instructions=INSTRUCTIONS, input=messages, text_format=ChangeRequest,
            reasoning={"effort": "low"}, max_output_tokens=4_000,
        )
        return r.output_parsed
    m = claude_client().messages.parse(
        model=ANTHROPIC_MODEL, max_tokens=4_000, system=INSTRUCTIONS, output_config={"effort": "low"},
        output_format=ChangeRequest, messages=messages,
    )
    return m.parsed_output


def native_row(provider: str, doc_id: str) -> dict:
    try:
        cr = extract_native(provider, [{"role": "user", "content": DOCS[doc_id]}])
        return {"provider": provider, "approach": "B: schema enforced", "doc": doc_id, "outcome": "valid", "record": cr}
    except Exception as exc:
        return {"provider": provider, "approach": "B: schema enforced", "doc": doc_id, "outcome": f"{exc.__class__.__name__}"}


def prefill_probe() -> str:
    """The old trick of pre-filling the assistant turn with "{" to force JSON, on ANTHROPIC_MODEL."""
    try:
        claude_client().messages.create(
            model=ANTHROPIC_MODEL, max_tokens=500, system=PROMPT_ONLY_SYSTEM,
            messages=[{"role": "user", "content": DOCS["CHG-20932"]}, {"role": "assistant", "content": "{"}],
        )
        return f"ℹ️  {ANTHROPIC_MODEL} still accepts a prefilled assistant turn."
    except anthropic.BadRequestError as exc:
        detail = exc.body["error"]["message"] if isinstance(exc.body, dict) else exc
        return f"❌ {ANTHROPIC_MODEL} → 400 BadRequestError: {detail}"


def run() -> None:
    step("CHG-20934, one of the five documents (in German)")
    print(DOCS["CHG-20934"])

    providers = active_providers()
    step("Approach A: JSON in the prompt, parsed and validated by your code")
    prompt_only = pd.DataFrame(run_parallel(extract_prompt_only, [(p, d) for p in providers for d in DOCS]))
    show(prompt_only, index=False)

    step("Deliberate failures: a recorded reply in a code fence, and the assistant-prefill trick")
    print("recorded reply →", parse_reply(RECORDED_REPLY)[0])
    if available("anthropic"):
        print(prefill_probe())

    step("Approach B: the API enforces the schema, and the SDK parses into ChangeRequest")
    native = pd.DataFrame(run_parallel(native_row, [(p, d) for p in providers for d in DOCS]))
    if native.empty:
        print("⚠️  No provider configured: skipping both approaches.")
        return
    both = pd.concat([prompt_only, native])
    valid = both.assign(valid=both.outcome.eq("valid")).pivot_table(index="provider", columns="approach", values="valid", aggfunc="sum")
    print("valid records out of", len(DOCS))
    show(valid)
    ok = native.outcome.eq("valid").all() and all(isinstance(r, ChangeRequest) for r in native.record)
    print(f"\n{'✅' if ok else '⚠️ '} approach B returned a validated ChangeRequest for every document")
