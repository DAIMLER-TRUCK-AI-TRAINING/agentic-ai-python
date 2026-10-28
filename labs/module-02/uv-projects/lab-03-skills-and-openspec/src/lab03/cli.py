"""Command line for the Lab 03 demo: `uv run lab03 <step>`; `uv run lab03 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
STEPS = {
    "pattern": ("lab03.part1_pattern", "Part 1 — Find the repeated prompting pattern"),
    "skills": ("lab03.part2_skills", "Part 2 — Package Skills and load them on demand"),
    "openspec": ("lab03.part3_openspec", "Part 3 — OpenSpec walkthrough"),
    "scenarios": ("lab03.part4_scenarios", "Part 4 — Complete the spec, then test it on both providers"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab03",
        description="Module 2 · Lab 03 — Designing agents with Skills and OpenSpec.",
        epilog="Steps: " + " · ".join(f"{name} = {title.split(' — ')[0]}" for name, (_, title) in STEPS.items()),
    )
    parser.add_argument("step", choices=[*STEPS, "all", "status"], help="the part to run; 'all' runs every part in order")
    parser.add_argument(
        "--provider",
        choices=["both", "openai", "anthropic"],
        default="both",
        help="call one provider only (default: both, side by side)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="the spec the scenarios step lints and tests (default: solutions/enterprise-ops-assistant.spec.md)",
    )
    args = parser.parse_args(argv)
    if args.spec and not args.spec.is_file():
        parser.error(f"--spec: no such file: {args.spec}")

    select_providers(args.provider)
    print_status()
    if args.step == "status":
        return
    for name in STEPS if args.step == "all" else [args.step]:
        module, title = STEPS[name]
        heading(title)
        part = importlib.import_module(module)
        if name == "scenarios":
            part.run(spec_path=args.spec.resolve() if args.spec else None)
        else:
            part.run()
