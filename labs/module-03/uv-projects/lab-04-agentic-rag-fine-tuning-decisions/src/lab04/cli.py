"""Command line for the Lab 04 demo: `uv run lab04 <step>`; `uv run lab04 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import FAILED, PROVIDERS, heading, print_status, select_provider

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading the RAG libraries.
STEPS = {
    "tool": ("lab04.part1_tool", "Part 1 — The retriever as a tool"),
    "agent": ("lab04.part2_agent", "Part 2 — An agent that decides when and where to search"),
    "canvases": ("lab04.part3_canvases", "Part 3 — Fine-tuning decision workshop: four scenarios"),
    "critique": ("lab04.part4_critique", "Part 4 — Challenge your decisions with a second opinion"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab04",
        description="Module 3 · Lab 04 — Agentic RAG and fine-tuning decisions.",
        epilog="Steps: " + " · ".join(f"{name} = {title.split(' — ')[0]}" for name, (_, title) in STEPS.items()),
    )
    parser.add_argument("step", choices=[*STEPS, "all", "status"], help="the part to run; 'all' runs every part in order")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="run on this provider only, with no comparison (default: LLM_PROVIDER, compared with its partner)",
    )
    args = parser.parse_args(argv)

    select_provider(args.provider)
    print_status()
    if args.step == "status":
        return
    for name in STEPS if args.step == "all" else [args.step]:
        module, title = STEPS[name]
        heading(title)
        importlib.import_module(module).run()
    if FAILED:
        raise SystemExit(f"\n❌ {len(FAILED)} check(s) failed: {', '.join(FAILED)}")
