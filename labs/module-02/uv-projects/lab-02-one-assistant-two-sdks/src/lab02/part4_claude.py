"""Part 4 — the same assistant on the Claude Messages API: tool use, streaming and the SDK's tool runner."""

from __future__ import annotations

import json

from anthropic import beta_tool

from .common import ANTHROPIC_MODEL, available, claude_client, claude_text, step
from .part1_tools import CLAUDE_TOOLS, dispatch, lookup_incident
from .part2_openai import QUESTIONS, SYSTEM, TRACES, Trace, show_steps


def run_claude(question: str, *, stream: bool = False, max_rounds: int = 6, tools: list | None = None) -> Trace:
    """The Claude tool loop, with optional streaming through the SDK's stream helper."""
    trace = Trace("anthropic", question)
    messages: list = [{"role": "user", "content": question}]
    for _ in range(max_rounds):
        request = dict(
            model=ANTHROPIC_MODEL, max_tokens=4_000, system=SYSTEM, tools=tools or CLAUDE_TOOLS,
            output_config={"effort": "low"}, messages=messages,
        )
        if stream:
            with claude_client().messages.stream(**request) as events:
                for event in events:
                    if event.type == "text":
                        trace.first_token()
                        print(event.text, end="", flush=True)
                message = events.get_final_message()
            print()
        else:
            message = claude_client().messages.create(**request)
        trace.add("model call", status=message.stop_reason, **{"in": message.usage.input_tokens, "out": message.usage.output_tokens})
        messages.append({"role": "assistant", "content": message.content})  # thinking + tool_use blocks go back in
        if message.stop_reason != "tool_use":
            trace.answer = claude_text(message)
            return trace
        results = []
        for block in (b for b in message.content if b.type == "tool_use"):
            trace.add("tool call", tool=block.name, args=json.dumps(block.input))
            output = dispatch(block.name, block.input)
            trace.add("tool result", tool=block.name, result=output[:120])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": '"error"' in output[:12]})
        messages.append({"role": "user", "content": results})
    trace.answer = f"[stopped after {max_rounds} rounds]"
    return trace


@beta_tool
def lookup_incident_tool(incident_id: str) -> str:
    """Look up an existing incident by its ID.

    Args:
        incident_id: INC- followed by 6 digits, e.g. INC-104233.
    """
    return json.dumps(lookup_incident(incident_id))


def run() -> None:
    if not available("anthropic"):
        print("⚠️  Claude is not selected or not configured: skipping this part.")
        return
    for qid, question in QUESTIONS.items():
        step(f"{qid}: {question}")
        TRACES["anthropic", qid] = trace = run_claude(question)
        print(trace.answer)

    step("Q2 streamed through the SDK's stream helper")
    TRACES["anthropic", "Q2 streamed"] = streamed = run_claude(QUESTIONS["Q2"], stream=True)
    print(f"\nfirst token after {streamed.first_token_s}s · whole answer after {streamed.steps[-1]['t (s)']}s")

    step("The trace of Q2")
    show_steps(TRACES["anthropic", "Q2"])

    step("The SDK can run the loop for you: the tool runner (beta), with a schema built from the docstring")
    runner = claude_client().beta.messages.tool_runner(
        model=ANTHROPIC_MODEL, max_tokens=4_000, system=SYSTEM, tools=[lookup_incident_tool],
        output_config={"effort": "low"}, messages=[{"role": "user", "content": QUESTIONS["Q1"]}],
    )
    final = runner.until_done()
    print(final.stop_reason, "·", "".join(b.text for b in final.content if b.type == "text"))
