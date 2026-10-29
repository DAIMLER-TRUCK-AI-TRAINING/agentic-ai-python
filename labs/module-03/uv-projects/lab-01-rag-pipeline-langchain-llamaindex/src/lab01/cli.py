"""Command line for the Lab 01 demo: `uv run lab01 <step>`; `uv run lab01 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import FAILED, PROVIDERS, heading, print_status, select_provider

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading the RAG libraries.
STEPS = {
    "corpus": ("lab01.part1_corpus", "Part 1 — The corpus, and a question without retrieval"),
    "faiss": ("lab01.part2_faiss", "Part 2 — Naive RAG with LangChain: chunk, embed and index in FAISS"),
    "chain": ("lab01.part3_chain", "Part 3 — Retrieve, augment, generate, with citations"),
    "llamaindex": ("lab01.part4_llamaindex", "Part 4 — The same pipeline in LlamaIndex, on Chroma"),
    "compare": ("lab01.part5_compare", "Part 5 — Compare the two pipelines"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab01",
        description="Module 3 · Lab 01 — Building a RAG pipeline with LangChain and LlamaIndex.",
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
