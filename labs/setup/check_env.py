#!/usr/bin/env python3
"""Check the course environment and print a ✅ / ⚠️ / ❌ table.

Run from the repo root with the course venv active:

    python labs/setup/check_env.py             # everything the course needs
    python labs/setup/check_env.py --module 1  # core + Module 1 only

Keys are reported as present or missing; their values are never printed.
Exit code 1 means at least one ❌ row: fix those before the lab starts.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import io
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SETUP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SETUP_DIR.parents[1]
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 13)  # guardrails-ai 0.11.0 declares Requires-Python <3.14

OK, WARN, FAIL, INFO = "✅", "⚠️ ", "❌", "ℹ️ "

# Import name -> distribution name, grouped by the module that first needs it.
# Module 0 is the core every module uses.
PACKAGES: dict[int, dict[str, str]] = {
    0: {
        "openai": "openai",
        "anthropic": "anthropic",
        "dotenv": "python-dotenv",
        "pydantic": "pydantic",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "nbformat": "nbformat",
        "ipykernel": "ipykernel",
    },
    1: {
        "tiktoken": "tiktoken",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "sentence_transformers": "sentence-transformers",
        "pytesseract": "pytesseract",
        "PIL": "pillow",
        "rapidfuzz": "rapidfuzz",
        "azure.ai.vision.imageanalysis": "azure-ai-vision-imageanalysis",
    },
    2: {
        "yaml": "pyyaml",
        "langchain": "langchain",
        "langchain_openai": "langchain-openai",
        "langchain_anthropic": "langchain-anthropic",
        "langchain_community": "langchain-community",
        "langsmith": "langsmith",
        "langchain_tavily": "langchain-tavily",
    },
    3: {
        "llama_index.core": "llama-index-core",
        "llama_index.llms.openai": "llama-index-llms-openai",
        "llama_index.llms.anthropic": "llama-index-llms-anthropic",
        "llama_index.embeddings.openai": "llama-index-embeddings-openai",
        "llama_index.embeddings.huggingface": "llama-index-embeddings-huggingface",
        "faiss": "faiss-cpu",
        "chromadb": "chromadb",
    },
    4: {"langgraph": "langgraph", "mcp": "mcp", "redis": "redis"},
    5: {"langfuse": "langfuse", "guardrails": "guardrails-ai"},
}

# Environment variables: (name, first module that uses it, is it a secret, severity if missing).
# Severity FAIL means the labs lose their main lesson without it; WARN means a fallback exists.
ENV_VARS: list[tuple[str, int, bool, str]] = [
    ("OPENAI_API_KEY", 1, True, FAIL),
    ("OPENAI_MODEL", 1, False, FAIL),
    ("OPENAI_SMALL_MODEL", 1, False, WARN),
    ("OPENAI_EMBEDDING_MODEL", 1, False, WARN),
    ("ANTHROPIC_API_KEY", 1, True, FAIL),
    ("ANTHROPIC_MODEL", 1, False, FAIL),
    ("ANTHROPIC_SMALL_MODEL", 1, False, WARN),
    ("LLM_PROVIDER", 2, False, WARN),
    ("LANGSMITH_TRACING", 2, False, WARN),
    ("LANGSMITH_API_KEY", 2, True, WARN),
    ("TAVILY_API_KEY", 2, True, WARN),
    ("REDIS_URL", 4, False, WARN),
    ("LANGFUSE_PUBLIC_KEY", 5, True, WARN),
    ("LANGFUSE_SECRET_KEY", 5, True, WARN),
    ("LANGFUSE_HOST", 5, False, WARN),
]

# Every lab calls both providers, so their keys and main models are checked whatever --module says.
ALWAYS_CHECKED = {"OPENAI_API_KEY", "OPENAI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"}

FALLBACK_NOTES = {
    "OPENAI_SMALL_MODEL": "Lab 01 skips the temperature sweep",
    "OPENAI_EMBEDDING_MODEL": "Lab 02 skips the OpenAI embedding comparison",
    "ANTHROPIC_SMALL_MODEL": "Lab 03 skips the tier comparison",
    "LLM_PROVIDER": "llm.py defaults to openai",
    "LANGSMITH_TRACING": "tracing off",
    "LANGSMITH_API_KEY": "tracing off",
    "TAVILY_API_KEY": "labs use the mock search tool",
    "REDIS_URL": "labs use the in-memory store",
    "LANGFUSE_PUBLIC_KEY": "Langfuse tracing off",
    "LANGFUSE_SECRET_KEY": "Langfuse tracing off",
    "LANGFUSE_HOST": "Langfuse tracing off",
}


@dataclass
class Row:
    """One line of the report."""

    status: str
    area: str
    item: str
    detail: str


def check_python() -> Row:
    version = platform.python_version()
    if MIN_PYTHON <= sys.version_info[:2] <= MAX_PYTHON:
        return Row(OK, "Python", version, sys.executable)
    return Row(
        FAIL,
        "Python",
        version,
        f"needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} to {MAX_PYTHON[0]}.{MAX_PYTHON[1]}: "
        f"recreate .venv with python{MAX_PYTHON[0]}.{MAX_PYTHON[1]} -m venv .venv",
    )


def check_packages(modules: list[int]) -> list[Row]:
    rows = []
    for module in modules:
        for import_name, dist in PACKAGES.get(module, {}).items():
            area = "Packages (core)" if module == 0 else f"Packages (M{module})"
            try:
                version = metadata.version(dist)
            except metadata.PackageNotFoundError:
                rows.append(Row(FAIL, area, dist, "not installed: pip install -r labs/setup/requirements.txt"))
                continue
            # Import for real (quietly): an installed but broken package must show up here.
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    importlib.import_module(import_name)
            except Exception as exc:
                rows.append(Row(FAIL, area, f"{dist} {version}", f"import failed: {exc.__class__.__name__}"))
            else:
                rows.append(Row(OK, area, dist, version))
    return rows


def load_env_file() -> Row:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return Row(FAIL, "Config", ".env", "missing: cp labs/setup/.env.example .env, then add your keys")
    try:
        from dotenv import load_dotenv
    except ImportError:
        return Row(WARN, "Config", ".env", "found, but python-dotenv is not installed")
    load_dotenv(env_path)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(env_path)], cwd=REPO_ROOT, capture_output=True
    ).returncode == 0
    if not ignored:
        return Row(FAIL, "Config", ".env", "found but NOT git-ignored: add .env to .gitignore")
    return Row(OK, "Config", ".env", "found and git-ignored")


def check_env_vars(modules: list[int]) -> list[Row]:
    rows = []
    for name, module, secret, severity in ENV_VARS:
        if module not in modules and name not in ALWAYS_CHECKED:
            continue
        value = os.getenv(name, "").strip()
        if value:
            # Secrets are only ever reported as present; model IDs are public and useful to see.
            rows.append(Row(OK, f"Env (M{module})", name, "present" if secret else value))
        else:
            note = FALLBACK_NOTES.get(name, "required by the labs")
            rows.append(Row(severity, f"Env (M{module})", name, f"missing: {note}"))
    if 1 in modules:
        azure = all(os.getenv(n) for n in ("AZURE_VISION_ENDPOINT", "AZURE_VISION_KEY"))
        rows.append(
            Row(INFO, "Env (M1)", "AZURE_VISION_*", "present (instructor demo)" if azure else "not set: instructor-only, fine")
        )
    return rows


def check_tesseract() -> Row:
    binary = shutil.which("tesseract")
    if not binary:
        return Row(FAIL, "System", "tesseract", "not found: brew install tesseract / apt-get install tesseract-ocr")
    out = subprocess.run([binary, "--version"], capture_output=True, text=True)
    first_line = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown version"
    listing = subprocess.run([binary, "--list-langs"], capture_output=True, text=True).stdout
    langs = [line.strip() for line in listing.splitlines()[1:] if line.strip()]  # line 1 is a header
    if "eng" not in langs:
        return Row(WARN, "System", "tesseract", f"{first_line}; English language data missing")
    return Row(OK, "System", "tesseract", f"{first_line}; langs: {', '.join(langs)}")


def check_redis() -> Row:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return Row(INFO, "System", "redis", "REDIS_URL not set: Module 4 uses the in-memory fallback")
    parsed = urlparse(url)
    host, port = parsed.hostname or "localhost", parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=1.5):
            pass
    except OSError as exc:
        return Row(WARN, "System", "redis", f"{host}:{port} unreachable ({exc.__class__.__name__}): in-memory fallback")
    return Row(OK, "System", "redis", f"{host}:{port} reachable")


def check_local_models(modules: list[int]) -> list[Row]:
    sys.path.insert(0, str(SETUP_DIR))
    try:
        from prefetch_models import MODELS
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        return [Row(WARN, "Models", "local models", f"cannot check: {exc}")]
    rows = []
    for module in modules:
        for repo_id in MODELS.get(module, []):
            cached = try_to_load_from_cache(repo_id, "config.json")
            if isinstance(cached, str):
                rows.append(Row(OK, f"Models (M{module})", repo_id, "cached"))
            else:
                rows.append(
                    Row(WARN, f"Models (M{module})", repo_id, "not cached: python labs/setup/prefetch_models.py")
                )
    return rows


def print_table(rows: list[Row]) -> None:
    widths = [max(len(r.area) for r in rows), max(len(r.item) for r in rows)]
    print(f"\n   {'Area':<{widths[0]}}  {'Item':<{widths[1]}}  Detail")
    print(f"   {'-' * widths[0]}  {'-' * widths[1]}  {'-' * 40}")
    for r in rows:
        print(f"{r.status} {r.area:<{widths[0]}}  {r.item:<{widths[1]}}  {r.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the course environment.")
    parser.add_argument("--module", type=int, choices=range(1, 6), help="check core + this module only")
    args = parser.parse_args()
    modules = [0, args.module] if args.module else sorted(PACKAGES)
    labs = [m for m in modules if m]

    rows = [check_python(), load_env_file()]
    rows += check_packages(modules)
    rows += check_env_vars(labs)
    if 1 in labs:
        rows.append(check_tesseract())
    if 4 in labs:
        rows.append(check_redis())
    rows += check_local_models(labs)

    print_table(rows)
    failures = sum(r.status == FAIL for r in rows)
    warnings = sum(r.status == WARN for r in rows)
    print(f"\n{failures} ❌  ·  {warnings} ⚠️   ·  scope: {'all modules' if not args.module else f'Module {args.module}'}")
    if failures:
        print("Fix the ❌ rows before the lab. ⚠️ rows have a fallback; the lab says what changes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
