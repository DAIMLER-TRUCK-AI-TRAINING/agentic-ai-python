"""Command line for the Lab 01 demo: `uv run lab01 <step>`; `uv run lab01 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
STEPS = {
    "baseline": ("lab01.part1_baseline", "Part 1 — The labelled ticket set and a baseline to beat"),
    "techniques": ("lab01.part2_techniques", "Part 2 — Zero-shot vs few-shot vs chain-of-thought, on both providers"),
    "debug": ("lab01.part3_debug", "Part 3 — Debug the prompt, then price it"),
    "schema": ("lab01.part4_schema", "Part 4 — Enforce a JSON schema on document extraction"),
    "validate": ("lab01.part5_validate", "Part 5 — Validate the output, retry with feedback, route what is left"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab01",
        description="Module 2 · Lab 01 — Prompt engineering and structured outputs.",
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
