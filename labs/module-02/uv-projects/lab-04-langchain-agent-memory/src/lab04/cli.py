"""Command line for the Lab 04 demo: `uv run lab04 <step>`; `uv run lab04 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading LangChain.
STEPS = {
    "chains": ("lab04.part1_chains", "Part 1 — LangChain building blocks: models, prompts, LCEL and output parsers"),
    "tools": ("lab04.part2_tools", "Part 2 — Tools: web search, a guarded database query, and summarisation"),
    "agent": ("lab04.part3_agent", "Part 3 — The agent: create_agent, a trace you can read, and both providers"),
    "buffer": ("lab04.part4_buffer", "Part 4 — Conversation memory: stateless, then a checkpointed buffer"),
    "memory": ("lab04.part5_memory", "Part 5 — Window memory vs summary memory"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab04",
        description="Module 2 · Lab 04 — Building a tool-enabled LangChain agent with memory.",
        epilog="Steps: " + " · ".join(f"{name} = {title.split(' — ')[0]}" for name, (_, title) in STEPS.items()),
    )
    parser.add_argument("step", choices=[*STEPS, "all", "status"], help="the part to run; 'all' runs every part in order")
    parser.add_argument(
        "--provider",
        choices=["both", "openai", "anthropic"],
        default="both",
        help="run on one provider only (default: both, LLM_PROVIDER first and the other for the comparisons)",
    )
    args = parser.parse_args(argv)

    select_providers(args.provider)
    print_status(llm_provider_matters=True)
    if args.step == "status":
        return
    for name in STEPS if args.step == "all" else [args.step]:
        module, title = STEPS[name]
        heading(title)
        importlib.import_module(module).run()
