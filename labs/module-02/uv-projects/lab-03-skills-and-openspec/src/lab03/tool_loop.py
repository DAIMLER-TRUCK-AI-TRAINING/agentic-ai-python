"""A compact tool loop for both providers (Lab 02 built it step by step); Part 2 runs the Skills through it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .common import ANTHROPIC_MODEL, OPENAI_MODEL, claude_client, claude_text, openai_client


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]


def run_with_tools(provider: str, system: str, question: str, tools: list[Tool], max_rounds: int = 6) -> dict:
    """Run the tool loop on one provider; return the answer, the tool calls and the token totals."""
    by_name = {t.name: t for t in tools}
    calls, tokens_in, tokens_out = [], 0, 0

    def run(name: str, args: dict) -> str:
        calls.append((name, args))
        try:
            return json.dumps(by_name[name].fn(**args))
        except Exception as exc:
            return json.dumps({"error": f"{exc.__class__.__name__}: {exc}"})

    if provider == "openai":
        items: list = [{"role": "user", "content": question}]
        specs = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": True} for t in tools]
        for _ in range(max_rounds):
            r = openai_client().responses.create(
                model=OPENAI_MODEL, instructions=system, input=items, tools=specs,
                reasoning={"effort": "low"}, max_output_tokens=6_000,
            )
            tokens_in, tokens_out = tokens_in + r.usage.input_tokens, tokens_out + r.usage.output_tokens
            items += [i.model_dump(exclude_none=True) for i in r.output]
            pending = [i for i in r.output if i.type == "function_call"]
            if not pending:
                return {"answer": r.output_text, "calls": calls, "in": tokens_in, "out": tokens_out}
            items += [{"type": "function_call_output", "call_id": c.call_id, "output": run(c.name, json.loads(c.arguments))} for c in pending]
    else:
        messages: list = [{"role": "user", "content": question}]
        specs = [{"name": t.name, "description": t.description, "input_schema": t.parameters, "strict": True} for t in tools]
        for _ in range(max_rounds):
            m = claude_client().messages.create(
                model=ANTHROPIC_MODEL, max_tokens=6_000, system=system, tools=specs,
                output_config={"effort": "low"}, messages=messages,
            )
            tokens_in, tokens_out = tokens_in + m.usage.input_tokens, tokens_out + m.usage.output_tokens
            messages.append({"role": "assistant", "content": m.content})
            if m.stop_reason != "tool_use":
                return {"answer": claude_text(m), "calls": calls, "in": tokens_in, "out": tokens_out}
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": run(b.name, b.input)}
                for b in m.content if b.type == "tool_use"
            ]})
    return {"answer": "[stopped: too many rounds]", "calls": calls, "in": tokens_in, "out": tokens_out}
