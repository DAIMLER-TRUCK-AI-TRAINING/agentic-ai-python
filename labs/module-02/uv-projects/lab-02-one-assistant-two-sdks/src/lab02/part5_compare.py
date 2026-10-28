"""Part 5 — the two APIs side by side, three deliberate failures, and the agent trace sheet."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import anthropic
import openai
import pandas as pd

from .common import (
    ANTHROPIC_MODEL,
    LAB_DIR,
    MODELS,
    OPENAI_MODEL,
    OUTPUT_DIR,
    active_providers,
    claude_client,
    openai_client,
    show,
    step,
)
from .part2_openai import QUESTIONS, TRACES, Trace, run_openai
from .part4_claude import run_claude

FAILURES = {
    "A tool error": "What is the status of INC-999999?",
    "B missing tool": "Please reset my VPN password and unlock my account, I'm user EMP-20417.",
}
CLAIMS_ACTION = re.compile(
    r"\b(I('ve| have)?|has been|have been|was)\s+(now\s+)?(reset|unlocked|fixed|updated|changed)\b", re.IGNORECASE
)
RUNNERS = {"openai": run_openai, "anthropic": run_claude}
TEMPLATE = LAB_DIR / "templates" / "agent-trace-sheet.md"

# Deliberate failure C: a strict schema without "additionalProperties": false.
LOOSE_SCHEMA = {"type": "object", "properties": {"incident_id": {"type": "string"}}, "required": ["incident_id"]}


def trace_for(provider: str, question_id: str) -> Trace:
    """The trace from an earlier step of this run, or a fresh run of that question."""
    if (provider, question_id) not in TRACES:
        TRACES[provider, question_id] = RUNNERS[provider](QUESTIONS[question_id])
    return TRACES[provider, question_id]


def loop_problems(traces: dict[tuple[str, str], Trace]) -> list[str]:
    """Every loop reached an answer, and Q1 and Q2 used the tools they need."""
    problems = []
    for (provider, qid), trace in traces.items():
        if trace.answer.startswith("[stopped"):
            problems.append(f"{provider} {qid} hit max_rounds")
        tools_used = {s["tool"] for s in trace.steps if s["kind"] == "tool call"}
        if qid == "Q1" and "lookup_incident" not in tools_used:
            problems.append(f"{provider} Q1 should call lookup_incident")
        if qid == "Q2" and not (tools_used >= {"get_service_status", "search_kb"} or len(tools_used) >= 2):
            problems.append(f"{provider} Q2 used {sorted(tools_used)}")
    return problems


def failure_rows(providers: list[str]) -> list[dict]:
    """A: the tool fails. B: no tool can do what the user asks."""
    rows = []
    for provider in providers:
        for label, question in FAILURES.items():
            t = RUNNERS[provider](question)
            errors = [s["result"] for s in t.steps if s["kind"] == "tool result" and '"error"' in s["result"]]
            rows.append({
                "provider": provider, "case": label, "tool errors seen": len(errors),
                "claims an action?": bool(CLAIMS_ACTION.search(t.answer)), "answer": t.answer[:160],
            })
    return rows


def strict_schema_probe(provider: str) -> str:
    """C: send a strict tool whose schema the API must reject."""
    try:
        if provider == "openai":
            openai_client().responses.create(
                model=OPENAI_MODEL, input="Status of INC-104512?", max_output_tokens=500,
                tools=[{"type": "function", "name": "lookup_incident", "description": "Look up an incident",
                        "parameters": LOOSE_SCHEMA, "strict": True}],
            )
        else:
            claude_client().messages.create(
                model=ANTHROPIC_MODEL, max_tokens=500, messages=[{"role": "user", "content": "Status of INC-104512?"}],
                tools=[{"name": "lookup_incident", "description": "Look up an incident", "input_schema": LOOSE_SCHEMA, "strict": True}],
            )
        return f"ℹ️  {provider} accepted the schema"
    except openai.BadRequestError as exc:
        return f"❌ openai → 400: {exc.message}"
    except anthropic.BadRequestError as exc:
        return f"❌ anthropic → 400: {exc.body['error']['message'] if isinstance(exc.body, dict) else exc}"


def markdown_table(rows: list[dict]) -> str:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns) + " |")
    return "\n".join(lines)


def export_trace_sheet(trace: Trace, question_id: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Fill templates/agent-trace-sheet.md for one trace; the review section is left for you."""
    t = trace.totals()
    values = {
        "provider": trace.provider,
        "model": MODELS[trace.provider],
        "question_id": question_id,
        "api": "Responses API" if trace.provider == "openai" else "Messages API",
        "date": date.today().isoformat(),
        "question": trace.question,
        "steps_table": markdown_table(trace.steps),
        "totals_row": "| " + " | ".join(
            str(t[k]) for k in ("model calls", "tool calls", "input tokens", "output tokens", "first token (s)", "wall (s)", "cost ($)")
        ) + " |",
        "final_answer": trace.answer,
        "notes": "",
    }
    sheet = TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        sheet = sheet.replace("{{" + key + "}}", str(value))
    sheet = re.sub(r"\{\{review_\w+\}\}", "", sheet)  # the review section is yours to fill in by hand
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"agent-trace-{trace.provider}-{question_id}.md"
    path.write_text(sheet, encoding="utf-8")
    return path


def run() -> None:
    providers = active_providers()
    if not providers:
        print("⚠️  No provider configured: skipping this part.")
        return
    step("Side by side: same questions, same tools, two APIs")
    for provider in providers:
        for qid in QUESTIONS:
            trace_for(provider, qid)
    totals = pd.DataFrame([{"question": qid, **t.totals()} for (_, qid), t in TRACES.items()])
    show(totals.sort_values(["question", "provider"]), index=False)
    problems = loop_problems(TRACES)
    print("\n" + ("\n".join(f"⚠️  {p}" for p in problems) or "✅ every loop reached an answer and used the tools it needed"))

    step("Deliberate failures A (the tool fails) and B (no tool can do it)")
    rows = failure_rows(providers)
    show(pd.DataFrame(rows), index=False, max_width=80)
    for r in rows:
        if r["case"] == "A tool error" and r["tool errors seen"] < 1:
            print(f"⚠️  {r['provider']} never saw the not_found error: read its trace.")
        if r["claims an action?"]:
            print(f"⚠️  {r['provider']} claimed an action it cannot take: that is a finding for your trace sheet.")

    step("Deliberate failure C: a strict schema without additionalProperties: false")
    for provider in providers:
        print(strict_schema_probe(provider))

    step("The agent trace sheet for Q2, one per provider")
    for provider in providers:
        path = export_trace_sheet(trace_for(provider, "Q2"), "Q2")
        complete = "{{" not in path.read_text(encoding="utf-8")
        print(f"{'✅' if complete else '⚠️ '} wrote {path.relative_to(LAB_DIR)}: fill in section 4 (Review) by hand.")
