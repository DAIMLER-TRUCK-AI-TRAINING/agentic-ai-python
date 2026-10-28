"""Command line for the Lab 05 demo: `uv run lab05 <step>`; `uv run lab05 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading LangChain.
STEPS = {
    "brief": ("lab05.part1_brief", "Part 1 — The brief: your spec, the contract and the evaluation set"),
    "build": ("lab05.part2_build", "Part 2 — Build it: an agent or a workflow?"),
    "policy": ("lab05.part3_policy", "Part 3 — Policy as code: what must always hold"),
    "evaluate": ("lab05.part4_evaluate", "Part 4 — Evaluate on both providers and choose"),
    "review": ("lab05.part5_review", "Part 5 — Architecture review: context-window risks and bottlenecks"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab05",
        description="Module 2 · Lab 05 — Mini-project: the Enterprise Operations Assistant.",
        epilog="Steps: " + " · ".join(f"{name} = {title.split(' — ')[0]}" for name, (_, title) in STEPS.items()),
    )
    parser.add_argument("step", choices=[*STEPS, "all", "status"], help="the part to run; 'all' runs every part in order")
    parser.add_argument(
        "--provider",
        choices=["both", "openai", "anthropic"],
        default="both",
        help="run on one provider only (default: both, LLM_PROVIDER first and the other in the evaluation)",
    )
    parser.add_argument(
        "--architecture",
        choices=["both", "agent", "workflow"],
        default="both",
        help="the architecture(s) to build, evaluate and review (default: both)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="the spec to render as the system prompt (default: your outputs/ spec if finished, else the reference)",
    )
    parser.add_argument("--team", default="Reference solution", help="the team name on the architecture review")
    args = parser.parse_args(argv)
    if args.spec and not args.spec.is_file():
        parser.error(f"--spec: no such file: {args.spec}")
    architectures = ["agent", "workflow"] if args.architecture == "both" else [args.architecture]

    select_providers(args.provider)
    print_status(llm_provider_matters=True)
    if args.step == "status":
        return
    importlib.import_module("lab05.part1_brief").use_spec(args.spec.resolve() if args.spec else None)
    for name in STEPS if args.step == "all" else [args.step]:
        module, title = STEPS[name]
        heading(title)
        part = importlib.import_module(module)
        if name in ("build", "evaluate"):
            part.run(architectures)
        elif name == "review":
            part.run(architectures, team=args.team)
        else:
            part.run()
