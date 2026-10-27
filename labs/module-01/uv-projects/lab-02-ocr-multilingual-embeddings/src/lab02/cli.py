"""Command line for the Lab 02 demo: `uv run lab02 <step>`; `uv run lab02 --help` lists the steps."""

from __future__ import annotations

import argparse
import importlib

from .common import heading, print_status, select_providers

# step → (module that implements it, the part of the lab guide it mirrors).
# Modules are imported only when their step runs, so `status` starts without loading PyTorch.
STEPS = {
    "tesseract": ("lab02.part1_tesseract", "Part 1 — Generate the samples and run Tesseract"),
    "accuracy": ("lab02.part2_accuracy", "Part 2 — Measure accuracy, fix the layout, clean the scan"),
    "vision": ("lab02.part3_vision", "Part 3 — AI-assisted OCR with OpenAI and Claude vision"),
    "cloud-ocr": ("lab02.part4_cloud_ocr", "Part 4 — Cloud OCR comparison (instructor demo)"),
    "similarity": ("lab02.part5_similarity", "Part 5 — Multilingual embeddings and semantic similarity"),
    "matching": ("lab02.part6_matching", "Part 6 — Match multilingual tickets to KB articles"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lab02",
        description="Module 1 · Lab 02 — Document intelligence: OCR and multilingual embeddings.",
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
