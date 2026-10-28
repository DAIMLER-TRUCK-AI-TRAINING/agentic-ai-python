"""Part 2 — two architectures behind one contract: an agent the model drives, and a workflow your code drives."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy

from .common import chat_model, main_provider, other_provider, step
from .ops import EscalationRecord, Triage, ops_tools
from .part1_brief import CASE_BY_ID, system_prompt
from .part3_policy import enforce_policy

ARCHITECTURES = ["agent", "workflow"]


@dataclass
class Result:
    provider: str
    architecture: str
    case_id: str
    record: EscalationRecord | None = None       # after enforce_policy (Part 3)
    raw: EscalationRecord | None = None          # what the model returned
    notes: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None


def ticket_message(text: str, ticket_id: str) -> str:
    return f'<ticket id="{ticket_id}">\n{text}\n</ticket>'


def count_usage(messages: list) -> tuple[int, int, int]:
    """(model calls, input tokens, output tokens) over the AI messages of one run."""
    ai = [m for m in messages if getattr(m, "type", "") == "ai" and m.usage_metadata]
    return len(ai), sum(m.usage_metadata["input_tokens"] for m in ai), sum(m.usage_metadata["output_tokens"] for m in ai)


def models_by_provider() -> dict[str, Any]:
    """The lab's provider first, then the comparison provider, if each is configured."""
    return {name: model for name in (main_provider(), other_provider()) if (model := chat_model(name)) is not None}


def build_agent(model):
    """Agent architecture: a create_agent loop the model drives.

    Read-only tools from ops_assistant, the system prompt rendered from the spec, provider-native
    structured output, and a model-call limit so a confused loop cannot run forever.
    """
    return create_agent(
        model,
        ops_tools.langchain_tools(),
        system_prompt=system_prompt(),
        response_format=ProviderStrategy(EscalationRecord),
        middleware=[ModelCallLimitMiddleware(run_limit=8, exit_behavior="error")],
    )


def run_workflow(model, text: str, ticket_id: str) -> tuple[EscalationRecord, list]:
    """Workflow architecture: your code decides the steps.

    1. Triage: one structured-output call that returns a Triage.
    2. Lookup: code calls search_kb, not the model.
    3. Escalation: one structured-output call on the ticket, the triage and the articles.
    Returns (the parsed EscalationRecord, [raw AI message of step 1, raw AI message of step 3]).
    """
    system = {"role": "system", "content": system_prompt()}
    ticket = ticket_message(text, ticket_id)
    step1 = model.with_structured_output(Triage, method="json_schema", include_raw=True).invoke(
        [system, {"role": "user", "content": ticket}]
    )
    articles = ops_tools.search_kb(text, 3)
    step3 = model.with_structured_output(EscalationRecord, method="json_schema", include_raw=True).invoke([
        system,
        {"role": "user", "content": f"{ticket}\n\n<triage>\n{step1['parsed'].model_dump_json()}\n</triage>\n\n"
                                    f"<search_kb_results>\n{json.dumps(articles)}\n</search_kb_results>\n\n"
                                    "Write the escalation record. Cite only KB IDs from the search results that apply."},
    ])
    return step3["parsed"], [step1["raw"], step3["raw"]]


_agents: dict[str, Any] = {}


def handle(provider: str, architecture: str, case: dict) -> Result:
    """The harness: run one case through one architecture, apply the policy, and record what it cost."""
    result = Result(provider, architecture, case["case_id"])
    model = models_by_provider()[provider]
    started = time.perf_counter()
    try:
        if architecture == "agent":
            if provider not in _agents:
                _agents[provider] = build_agent(model)
            out = _agents[provider].invoke({"messages": [{"role": "user", "content": ticket_message(case["text"], case["ticket_id"])}]})
            result.raw, messages = out.get("structured_response"), out["messages"]
            result.tool_calls = [c["name"] for m in messages if m.type == "ai" for c in m.tool_calls]
        else:
            result.raw, messages = run_workflow(model, case["text"], case["ticket_id"])
            result.tool_calls = ["search_kb (by code)"]
        result.model_calls, result.input_tokens, result.output_tokens = count_usage(messages)
    except Exception as exc:  # a failed run is a result to count, not a crash
        result.error = f"{exc.__class__.__name__}: {exc}"[:300]
    if result.raw is not None:
        result.record, result.notes = enforce_policy(result.raw)
    result.seconds = round(time.perf_counter() - started, 2)
    return result


def run(architectures: list[str] = ARCHITECTURES) -> None:
    provider = main_provider()
    if provider not in models_by_provider():
        print(f"⚠️  {provider} is not configured: skipping the smoke test.")
        return
    case = CASE_BY_ID["EV-02"]
    for architecture in architectures:
        step(f"Smoke test: {case['case_id']} through the {architecture} on {provider}")
        r = handle(provider, architecture, case)
        print(f"{r.model_calls} model calls · tools {r.tool_calls} · {r.input_tokens:,} in · {r.seconds}s · error={r.error}")
        print(r.raw.model_dump_json(indent=1) if r.raw else "no record")
        ok = r.error is None and isinstance(r.raw, EscalationRecord)
        print(f"{'✅' if ok else '❌'} the {architecture} returned an EscalationRecord")
