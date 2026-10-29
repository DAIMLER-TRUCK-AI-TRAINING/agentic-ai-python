#!/usr/bin/env python3
"""Download the open models and tokenisers the labs run locally.

Run this once after installing the requirements, on a good connection, so that
lab time is not spent waiting for downloads:

    python labs/setup/prefetch_models.py            # every module
    python labs/setup/prefetch_models.py --module 1 # only Module 1

Files land in the standard Hugging Face and tiktoken caches, so the notebooks
find them without any extra configuration.
"""

from __future__ import annotations

import argparse
import sys

# Hugging Face repositories each module loads locally, keyed by module number.
MODELS: dict[int, list[str]] = {
    1: [
        "HuggingFaceTB/SmolLM2-135M",  # Lab 01: next-token probabilities and temperature
        "sentence-transformers/all-MiniLM-L6-v2",  # Lab 02: English-only embeddings
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Lab 02: multilingual
    ],
    3: [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # ST_MODEL: local embeddings
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # RERANK_MODEL: Labs 02-04 re-ranking
    ],
}

# tiktoken encodings used as an offline token-count approximation.
TIKTOKEN_ENCODINGS: dict[int, list[str]] = {1: ["o200k_base"]}

# Weight formats the labs never load; skipping them roughly halves the download.
IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "pytorch_model.bin",
]


def prefetch(modules: list[int]) -> int:
    """Download every model and encoding for ``modules``; return the failure count."""
    from huggingface_hub import snapshot_download

    failures = 0
    for module in modules:
        for repo_id in MODELS.get(module, []):
            print(f"[Module {module}] {repo_id} ...", end=" ", flush=True)
            try:
                snapshot_download(repo_id, ignore_patterns=IGNORE_PATTERNS)
            except Exception as exc:  # network, disk or auth problems all end up here
                failures += 1
                print(f"failed: {exc.__class__.__name__}: {exc}")
            else:
                print("ok")

        for name in TIKTOKEN_ENCODINGS.get(module, []):
            print(f"[Module {module}] tiktoken {name} ...", end=" ", flush=True)
            try:
                import tiktoken

                tiktoken.get_encoding(name)
            except Exception as exc:
                failures += 1
                print(f"failed: {exc.__class__.__name__}: {exc}")
            else:
                print("ok")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the open models the labs run locally.")
    parser.add_argument("--module", type=int, choices=sorted(MODELS), help="only this module")
    args = parser.parse_args()
    modules = [args.module] if args.module else sorted(MODELS)
    failures = prefetch(modules)
    if failures:
        print(f"\n{failures} download(s) failed. Check your connection and run the script again.")
        sys.exit(1)
    print("\nAll models are cached locally.")


if __name__ == "__main__":
    main()
