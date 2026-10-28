"""Part 5 — the architecture review: where tokens and seconds go, two stress tests, and the written review."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from .common import LAB_DIR, OUTPUT_DIR, chat_model, main_provider, show, step
from .ops import ops_tools
from .part1_brief import CASE_BY_ID, system_prompt
from .part2_build import ARCHITECTURES, build_agent, handle, models_by_provider, ticket_message
from .part4_evaluate import evaluate, summarise

TEMPLATE = LAB_DIR / "templates" / "architecture-review.md"


class StepTimer(BaseCallbackHandler):
    """Collect (step, start, end) for every model call and tool call of one run."""

    def __init__(self):
        self.t0, self.open, self.steps = time.perf_counter(), {}, []

    def _start(self, run_id, label):
        self.open[run_id] = (label, time.perf_counter() - self.t0)

    def _end(self, run_id):
        label, start = self.open.pop(run_id, ("?", 0.0))
        self.steps.append({"step": label, "start (s)": round(start, 2), "duration (s)": round(time.perf_counter() - self.t0 - start, 2)})

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(run_id, "model call")

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._start(run_id, f"tool {serialized.get('name')}")

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id)


def context_budget(llm) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Where the first model call's input tokens go, and where one agent run's time goes (case EV-04)."""
    case = CASE_BY_ID["EV-04"]
    tool_schemas = [convert_to_openai_tool(t) for t in ops_tools.langchain_tools()]
    timer = StepTimer()
    out = build_agent(llm).invoke(
        {"messages": [{"role": "user", "content": ticket_message(case["text"], case["ticket_id"])}]}, {"callbacks": [timer]}
    )
    first_call = next(m for m in out["messages"] if m.type == "ai" and m.usage_metadata)
    reported = first_call.usage_metadata["input_tokens"]
    parts = {
        "system prompt (spec + policy)": count_tokens_approximately([{"role": "system", "content": system_prompt()}]),
        "tool schemas (4 tools)": round(len(json.dumps(tool_schemas)) / 4),
        "ticket": count_tokens_approximately([{"role": "user", "content": ticket_message(case["text"], case["ticket_id"])}]),
    }
    parts["output schema + provider overhead"] = max(reported - sum(parts.values()), 0)
    budget = pd.DataFrame({"≈ tokens": parts}).assign(share=lambda d: (d["≈ tokens"] / d["≈ tokens"].sum()).round(2))
    return budget, pd.DataFrame(timer.steps), reported


def approx(obj) -> int:
    return round(len(json.dumps(obj, default=str)) / 4)


LOG_LINES = [
    f"2026-10-27T07:{m // 60:02d}:{m % 60:02d}Z vpn-gateway-eu-01 auth user=u{m % 900:04d} result=ok latency_ms={40 + m % 17}"
    for m in range(2_400)
]
for _i in (1_210, 1_640, 2_101):
    LOG_LINES[_i] = LOG_LINES[_i].replace("result=ok", "result=FAIL error=809 reason=cached_credential")
BIG_TICKET = "VPN keeps failing for me after my password change. Full client log below.\n" + "\n".join(LOG_LINES)


def condense_attachment(text: str, head: int = 5, tail: int = 5, max_hits: int = 20) -> str:
    """Keep the first and last lines and every line that looks like an error; say what was dropped."""
    lines = text.splitlines()
    hits = [line for line in lines[head:-tail] if re.search(r"FAIL|ERROR|error=", line)][:max_hits]
    dropped = len(lines) - head - tail - len(hits)
    return "\n".join(lines[:head] + hits + [f"[… {dropped:,} routine lines removed by the gateway before the model call …]"] + lines[-tail:])


def stress_rows(architectures: list[str]) -> list[dict]:
    """Stress test 1: an unbounded tool result. Stress test 2: a ticket with a pasted log."""
    with sqlite3.connect(f"file:{ops_tools.ensure_db()}?mode=ro", uri=True) as conn:
        all_rows = conn.execute("SELECT * FROM incidents").fetchall()
    capped = ops_tools.query_ops_db("SELECT * FROM incidents")
    condensed = condense_attachment(BIG_TICKET)
    rows = [
        {"input": "query_ops_db: SELECT * FROM incidents, uncapped", "rows/lines": len(all_rows), "≈ tokens": approx(all_rows)},
        {"input": "query_ops_db: same query, 50-row cap", "rows/lines": len(capped["rows"]), "≈ tokens": approx(capped)},
        {"input": "ticket with pasted log, raw", "rows/lines": len(LOG_LINES), "≈ tokens": round(len(BIG_TICKET) / 4)},
        {"input": "ticket with pasted log, condensed", "rows/lines": len(condensed.splitlines()), "≈ tokens": round(len(condensed) / 4)},
    ]
    provider = main_provider()
    if provider in models_by_provider() and architectures:
        architecture = architectures[-1]
        for label, text in (("raw", BIG_TICKET), ("condensed", condensed)):
            r = handle(provider, architecture, {"case_id": f"STRESS-{label}", "ticket_id": "INC-104699", "text": text})
            rows.append({
                "input": f"{architecture} run on the {label} ticket ({provider})", "rows/lines": "—", "≈ tokens": r.input_tokens,
                "seconds": r.seconds, "outcome": r.record.priority + (" escalate" if r.record.escalate else "") if r.record else r.error,
            })
    return rows


# The reference review. In the lab, your team writes these from your own measurements.
REVIEW = {
    "decision": "Ship the agent architecture on the cheaper provider that clears the bar: both architectures reach the "
                "eval targets after the policy layer, the agent adapts to tickets that name incidents or services, and "
                "the policy layer makes the escalation decision deterministic where the spec requires it.",
    "r1_evidence": "SELECT * returns ~437 rows; uncapped that is tens of thousands of tokens, capped at 50 rows a fraction of it",
    "r1_mitigation": "Row cap in the tool, plus 'truncated: true' so the model narrows the query instead of guessing",
    "r1_owner": "Platform Engineering",
    "r2_evidence": "A 2,400-line pasted log is ~60k tokens raw and under 1k condensed; the raw run is slower and costs more",
    "r2_mitigation": "Condense attachments before the model call (head, tail, error lines) and store the full log by reference",
    "r2_owner": "Service Desk tooling",
    "r3_evidence": "The fixed prompt (spec + policy + tool schemas) is most of every first call's input",
    "r3_mitigation": "Prompt caching on the stable prefix; move rarely used rules into a Skill loaded on demand (Lab 03)",
    "r3_owner": "Assistant owner",
    "r4_evidence": "Model calls dominate the waterfall; tools return in milliseconds",
    "r4_mitigation": "Fewer model turns (workflow for simple tickets), low reasoning effort, stream the user reply",
    "r4_owner": "Assistant owner",
    "r5_evidence": "Knowledge lookup is keyword search over 10 articles; recall will drop as the KB grows",
    "r5_mitigation": "Replace search_kb with retrieval over embeddings (Module 3) and measure precision@k",
    "r5_owner": "Knowledge management",
    "next_steps": "Module 3: RAG over the full KB and policy documents behind the same search_kb contract. "
                  "Module 4: a supervisor that routes simple tickets to the workflow and complex ones to the agent, with a HITL gate on P1.",
}


def md_table(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "_not measured_"
    table = frame.reset_index() if frame.index.name or not isinstance(frame.index, pd.RangeIndex) else frame
    cols = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.astype(object).fillna("—").to_numpy()]
    return "\n".join(lines)


def export_review(
    *,
    team: str,
    providers: list[str],
    architectures: list[str],
    tables: dict[str, pd.DataFrame | None],
    review: dict[str, str] = REVIEW,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Fill templates/architecture-review.md with the measured tables and the written review."""
    values = {
        "team": team, "date": date.today().isoformat(), "providers": ", ".join(providers) or "—",
        "architecture": ", ".join(architectures) or "—", **{k: md_table(v) for k, v in tables.items()}, **review,
    }
    sheet = TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        sheet = sheet.replace("{{" + key + "}}", str(value))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "architecture-review.md"
    path.write_text(sheet, encoding="utf-8")
    return path


def run(architectures: list[str] = ARCHITECTURES, team: str = "Reference solution") -> None:
    provider = main_provider()
    llm = chat_model(provider)
    budget = waterfall = None
    if llm is not None and "agent" in architectures:
        step(f"Where the first model call's input tokens go, and where one run's time goes ({provider}, EV-04)")
        budget, waterfall, reported = context_budget(llm)
        print(f"First model call: {reported:,} input tokens (reported by the API)")
        show(budget)
        show(waterfall, index=False)

    step("Stress tests: an unbounded tool result, and a ticket with a 2,400-line pasted log")
    stress = pd.DataFrame(stress_rows(architectures))
    show(stress.fillna("—"), index=False)

    evaluation = evaluate(tuple(architectures)) if models_by_provider() else pd.DataFrame()
    path = export_review(
        team=team, providers=list(models_by_provider()), architectures=architectures,
        tables={
            "eval_table": summarise(evaluation) if not evaluation.empty else None,
            "budget_table": budget, "stress_table": stress, "waterfall_table": waterfall,
        },
    )
    text = path.read_text(encoding="utf-8")
    print(f"\n{'✅' if '{{' not in text else '❌'} wrote {path.relative_to(LAB_DIR)} (every placeholder filled)")

    if llm is not None:
        step("The packaged reference (labs/module-02/ops_assistant/): Module 5 Lab 03 imports exactly this")
        from ops_assistant import build_assistant

        reference = build_assistant(provider, architecture="agent")
        for case in (CASE_BY_ID["EV-03"], CASE_BY_ID["EV-08"]):
            r = reference.handle(case["text"], case["ticket_id"])
            outcome = f"{r.record.priority}{' escalate' if r.record.escalate else ''} {r.record.assignment_group}" if r.record else r.error
            print(f"{case['case_id']} → {outcome} · notes: {r.policy_notes}")
