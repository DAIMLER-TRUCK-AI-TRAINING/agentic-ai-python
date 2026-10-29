"""Command line for the Lab 02 demo: `uv run lab02 <step>`; `uv run lab02 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import FAILED, PROVIDERS, heading, print_status, select_provider

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading the RAG libraries.
STEPS = {
    "metrics": ("lab02.part1_metrics", "Part 1 — Labelled queries and retrieval metrics"),
    "chunking": ("lab02.part2_chunking", "Part 2 — Fixed vs semantic chunking"),
    "hybrid": ("lab02.part3_hybrid", "Part 3 — Keyword, hybrid and re-ranked retrieval"),
    "judge": ("lab02.part4_judge", "Part 4 — Answer relevance with an LLM judge, on both providers"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab02",
        description="Module 3 · Lab 02 — Tuning and evaluating retrieval.",
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
