"""Command line for the Lab 03 demo: `uv run lab03 <step>`; `uv run lab03 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import FAILED, PROVIDERS, heading, print_status, select_provider

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading the RAG libraries.
STEPS = {
    "connectors": ("lab03.part1_connectors", "Part 1 — Connect SharePoint and Confluence"),
    "sync": ("lab03.part2_sync", "Part 2 — Keep the index in sync: retired versions, edits and deletes"),
    "access": ("lab03.part3_access", "Part 3 — Index with access metadata, and prove the trimming"),
    "assistants": ("lab03.part4_assistants", "Part 4 — One pattern, two assistants"),
    "calibrate": ("lab03.part5_calibrate", "Part 5 — Calibrate the floor and compare providers"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab03",
        description="Module 3 · Lab 03 — Enterprise connectors and use-case assistants.",
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
