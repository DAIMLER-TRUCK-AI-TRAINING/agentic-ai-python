"""Command line for the Lab 03 demo: `uv run lab03 <step>`; `uv run lab03 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
STEPS = {
    "scenarios": ("lab03.part1_scenarios", "Part 1 — Frame the four scenarios"),
    "mapping": ("lab03.part2_mapping", "Part 2 — Map each scenario to an agent type and a model family"),
    "cost": ("lab03.part3_cost", "Part 3 — Estimate tokens and cost"),
    "decide": ("lab03.part4_decision", "Part 4 — Score, decide and write it up"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab03",
        description="Module 1 · Lab 03 — Mapping use cases and selecting models.",
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
        "--scenario",
        choices=["UC-1", "UC-2", "UC-3", "UC-4"],
        default="UC-1",
        help="the scenario the decide step scores (default: UC-1, the guide's §8.4 example)",
    )
    parser.add_argument("--team", default="Team name", help="your team's name for the comparison sheet")
    args = parser.parse_args(argv)

    select_providers(args.provider)
    print_status()
    if args.step == "status":
        return
    for name in STEPS if args.step == "all" else [args.step]:
        module, title = STEPS[name]
        heading(title)
        part = importlib.import_module(module)
        if name == "decide":
            part.run(scenario_id=args.scenario, team=args.team)
        else:
            part.run()
