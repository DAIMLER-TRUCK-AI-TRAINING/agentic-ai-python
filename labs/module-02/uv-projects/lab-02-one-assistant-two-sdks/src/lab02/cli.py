"""Command line for the Lab 02 demo: `uv run lab02 <step>`; `uv run lab02 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
# Later steps reuse the traces of earlier ones in the same run, and make any they are missing.
STEPS = {
    "tools": ("lab02.part1_tools", "Part 1 — Tools as contracts"),
    "openai": ("lab02.part2_openai", "Part 2 — The assistant on the OpenAI Responses API: function calling"),
    "streaming": ("lab02.part3_streaming", "Part 3 — Streaming responses"),
    "claude": ("lab02.part4_claude", "Part 4 — Rebuild it with the Claude Messages API: tool use"),
    "compare": ("lab02.part5_compare", "Part 5 — Compare the two APIs, break them on purpose, and fill the trace sheet"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab02",
        description="Module 2 · Lab 02 — One assistant, two SDKs: OpenAI function calling vs Claude tool use.",
        epilog="Steps: " + " · ".join(f"{name} = {title.split(' — ')[0]}" for name, (_, title) in STEPS.items()),
    )
    parser.add_argument("step", choices=[*STEPS, "all", "status"], help="the part to run; 'all' runs every part in order")
    parser.add_argument(
        "--provider",
        choices=["both", "openai", "anthropic"],
        default="both",
        help="call one provider only (default: both, side by side)",
    )
    args = parser.parse_args(argv)

    select_providers(args.provider)
    print_status()
    if args.step == "status":
        return
    for name in STEPS if args.step == "all" else [args.step]:
        module, title = STEPS[name]
        heading(title)
        importlib.import_module(module).run()
