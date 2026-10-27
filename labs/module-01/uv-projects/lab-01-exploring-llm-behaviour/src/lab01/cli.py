"""Command line for the Lab 01 demo: `uv run lab01 <step>`; `uv run lab01 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading PyTorch.
STEPS = {
    "first-calls": ("lab01.part1_first_calls", "Part 1 — Check the environment and make your first calls"),
    "temperature": ("lab01.part2_temperature", "Part 2 — Temperature and sampling"),
    "hallucination": ("lab01.part3_hallucination", "Part 3 — Hallucination triggers and grounding"),
    "context": ("lab01.part4_context", "Part 4 — Context-window edge cases"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab01",
        description="Module 1 · Lab 01 — Exploring LLM behaviour with OpenAI and Claude.",
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
